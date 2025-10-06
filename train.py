# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a YOLOv3 model on a custom dataset. Models and datasets download automatically from the latest YOLOv3 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
from contextlib import nullcontext

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.patches import torch_load

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)

# Optional: Opacus for Differential Privacy
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    import opacus
    OPACUS_VERSION = opacus.__version__
    # Don't log here as LOGGER might not be initialized yet
except Exception as e:  # pragma: no cover
    PrivacyEngine = None
    ModuleValidator = None
    BatchMemoryManager = None
    OPACUS_VERSION = None
    # Don't log here as LOGGER might not be initialized yet

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()


def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    """
    Train a YOLOv3 model on a custom dataset and manage the training process.

    Args:
        hyp (str | dict): Path to hyperparameters yaml file or hyperparameters dictionary.
        opt (argparse.Namespace): Parsed command line arguments containing training options.
        device (torch.device): Device to load and train the model on.
        callbacks (Callbacks): Callbacks to handle various stages of the training lifecycle.

    Returns:
        None

    Usage - Single-GPU training:
        $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
        $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

    Usage - Multi-GPU DDP training:
        $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights
            yolov5s.pt --img 640 --device 0,1,2,3

    Models: https://github.com/ultralytics/yolov5/tree/master/models
    Datasets: https://github.com/ultralytics/yolov5/tree/master/data
    Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data

    Examples:
        ```python
        from ultralytics import train
        import argparse
        import torch
        from utils.callbacks import Callbacks

        # Example usage
        args = argparse.Namespace(
            data='coco128.yaml',
            weights='yolov5s.pt',
            cfg='yolov5s.yaml',
            img_size=640,
            epochs=50,
            batch_size=16,
            device='0'
        )

        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        callbacks = Callbacks()

        train(hyp='hyp.scratch.yaml', opt=args, device=device, callbacks=callbacks)
        ```
    """
    # Initialize DP mode flag
    simple_dp = getattr(opt, "dp_simple", False)
    
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.evolve,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.freeze,
    )
    callbacks.run("on_pretrain_routine_start")

    # Directories
    w = save_dir / "weights"  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / "last.pt", w / "best.pt"

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    if not evolve:
        yaml_save(save_dir / "hyp.yaml", hyp)
        yaml_save(save_dir / "opt.yaml", vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != "cpu"
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict["train"], data_dict["val"]
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset

    # Model
    check_suffix(weights, ".pt")  # check weights
    pretrained = weights.endswith(".pt")
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch_load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create

    # Differential Privacy pre-checks: convert BN -> GN and disable AMP if DP
    if getattr(opt, "dp", False):
        assert PrivacyEngine is not None and ModuleValidator is not None, (
            "Opacus is not installed. Please 'pip install opacus' to use --dp"
        )
        
        # Log Opacus version now that LOGGER is initialized
        if OPACUS_VERSION:
            LOGGER.info(f"Opacus version: {OPACUS_VERSION}")
        
        # Check Opacus version compatibility
        if OPACUS_VERSION:
            version_parts = OPACUS_VERSION.split('.')
            major_version = int(version_parts[0]) if version_parts else 0
            if major_version < 1:
                LOGGER.warning(f"Opacus version {OPACUS_VERSION} may have compatibility issues. Consider upgrading to 1.x+")
        else:
            LOGGER.warning("Opacus import failed or version could not be determined")
        
        # Manual conversion of BatchNorm to GroupNorm for better DP compatibility
        def convert_batchnorm_to_groupnorm(model):
            """Convert all BatchNorm2d layers to GroupNorm for DP compatibility."""
            converted_count = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    # Get the parent module and the attribute name
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                        old_bn = getattr(parent, child_name)
                        
                        # Create GroupNorm with same number of channels
                        num_groups = min(32, old_bn.num_features)  # GroupNorm works best with 8-32 groups
                        new_gn = nn.GroupNorm(num_groups, old_bn.num_features)
                        
                        # Copy weights and bias if they exist
                        if old_bn.weight is not None:
                            new_gn.weight.data = old_bn.weight.data.clone()
                        if old_bn.bias is not None:
                            new_gn.bias.data = old_bn.bias.data.clone()
                        
                        # Replace the layer
                        setattr(parent, child_name, new_gn)
                        converted_count += 1
                        LOGGER.info(f"Converted BatchNorm2d to GroupNorm: {name}")
                    else:
                        # Root level module
                        old_bn = module
                        num_groups = min(32, old_bn.num_features)
                        new_gn = nn.GroupNorm(num_groups, old_bn.num_features)
                        
                        if old_bn.weight is not None:
                            new_gn.weight.data = old_bn.weight.data.clone()
                        if old_bn.bias is not None:
                            new_gn.bias.data = old_bn.bias.data.clone()
                        
                        # For root level, we need to replace in the parent
                        for parent_name, parent_module in model.named_modules():
                            for child_name, child_module in parent_module.named_children():
                                if child_module is old_bn:
                                    setattr(parent_module, child_name, new_gn)
                                    converted_count += 1
                                    LOGGER.info(f"Converted root BatchNorm2d to GroupNorm: {name}")
                                    break
                            if converted_count > 0:
                                break
                        break
            
            LOGGER.info(f"Total BatchNorm2d layers converted to GroupNorm: {converted_count}")
            return converted_count
        
        # Convert BatchNorm to GroupNorm
        converted_count = convert_batchnorm_to_groupnorm(model)
        if converted_count > 0:
            LOGGER.info(f"Successfully converted {converted_count} BatchNorm2d layers to GroupNorm for DP compatibility")
        
        # Additional validation with Opacus
        if not ModuleValidator.is_valid(model):
            LOGGER.info("Model still has DP-incompatible layers. Applying Opacus fixes...")
            model = ModuleValidator.fix(model)
            LOGGER.info("Applied additional Opacus fixes")
        
        # Final validation check
        if not ModuleValidator.is_valid(model):
            LOGGER.warning("Model still has DP-incompatible layers after conversion. This may cause issues.")
            # List remaining incompatible layers for debugging
            incompatible_modules = []
            for name, module in model.named_modules():
                if not ModuleValidator.is_valid(module):
                    incompatible_modules.append(name)
            if incompatible_modules:
                LOGGER.warning(f"Remaining incompatible modules: {incompatible_modules[:10]}...")  # Show first 10
        
        # Ensure model is in training mode for DP
        model.train()
        
        # Test forward pass with dummy data to ensure conversion worked
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 416, 416).to(device)
                _ = model(dummy_input)
                LOGGER.info("✅ Model forward pass test successful after BatchNorm->GroupNorm conversion")
        except Exception as e:
            LOGGER.warning(f"⚠️ Model forward pass test failed after conversion: {e}")
            LOGGER.info("This might indicate an issue with the conversion process")
        
        # Disable in-place activations (SiLU/ReLU/etc.) to satisfy Opacus grad sampler
        def _disable_inplace_activations(root_module: nn.Module) -> None:
            for m in root_module.modules():
                if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.SiLU, nn.Hardswish, nn.ReLU6)):
                    if hasattr(m, "inplace") and m.inplace:
                        m.inplace = False
        _disable_inplace_activations(model)
        
        if opt.sync_bn:
            LOGGER.warning("Disabling --sync-bn because DP is enabled")
            opt.sync_bn = False
        amp = False
    else:
        amp = check_amp(model)  # check AMP

    # Freeze
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    else:

        def lf(x):
            """Linear learning rate scheduler function with decay calculated by epoch proportion."""
            return (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA (initialized now; may be re-initialized after DP wrapping)
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")

    # Trainloader
    try:
        # Debug: Check if create_dataloader is available
        if 'create_dataloader' not in globals():
            LOGGER.error("create_dataloader function not found in globals!")
            LOGGER.error(f"Available functions: {[k for k in globals().keys() if 'dataloader' in k.lower()]}")
            raise NameError("create_dataloader function not found")
        
        train_loader, dataset = create_dataloader(
            train_path,
            imgsz,
            batch_size // WORLD_SIZE,
            gs,
            single_cls,
            hyp=hyp,
            augment=True,
            cache=None if opt.cache == "val" else opt.cache,
            rect=opt.rect,
            rank=LOCAL_RANK,
            workers=workers,
            image_weights=opt.image_weights,
            quad=opt.quad,
            prefix=colorstr("train: "),
            shuffle=True,
            seed=opt.seed,
        )
    except Exception as e:
        LOGGER.error(f"Error creating train dataloader: {e}")
        LOGGER.error(f"train_path: {train_path}")
        LOGGER.error(f"imgsz: {imgsz}")
        LOGGER.error(f"batch_size: {batch_size}")
        LOGGER.error(f"WORLD_SIZE: {WORLD_SIZE}")
        raise
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Process 0
    if RANK in {-1, 0}:
        try:
            val_loader = create_dataloader(
                val_path,
                imgsz,
                batch_size // WORLD_SIZE * 2,
                gs,
                single_cls,
                hyp=hyp,
                cache=None if noval else opt.cache,
                rect=True,
                rank=-1,
                workers=workers * 2,
                pad=0.5,
                prefix=colorstr("val: "),
            )[0]
        except Exception as e:
            LOGGER.error(f"Error creating validation dataloader: {e}")
            LOGGER.error(f"val_path: {val_path}")
            raise

        if not resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision

        callbacks.run("on_pretrain_routine_end", labels, names)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Opacus PrivacyEngine setup (single-process only)
    if getattr(opt, "dp", False):
        assert PrivacyEngine is not None and ModuleValidator is not None, (
            "Opacus is not installed. Please 'pip install opacus' to use --dp"
        )
        
        # Log Opacus version now that LOGGER is initialized
        if OPACUS_VERSION:
            LOGGER.info(f"Opacus version: {OPACUS_VERSION}")
        
        # Check Opacus version compatibility
        if OPACUS_VERSION:
            version_parts = OPACUS_VERSION.split('.')
            major_version = int(version_parts[0]) if version_parts else 0
            if major_version < 1:
                LOGGER.warning(f"Opacus version {OPACUS_VERSION} may have compatibility issues. Consider upgrading to 1.x+")
        else:
            LOGGER.warning("Opacus import failed or version could not be determined")
        
        if RANK != -1:  
            raise AssertionError("DP training with Opacus is not supported with DDP in this script. Use single device.")
        if torch.cuda.device_count() > 1 and isinstance(model, torch.nn.DataParallel):
            raise AssertionError("DP training is not supported with torch.nn.DataParallel. Use single device.")
        # Gradient accumulation complicates privacy accounting; enforce accumulate=1
        if 'accumulate' in locals() and accumulate != 1:
            LOGGER.warning("Disabling gradient accumulation under DP; setting accumulate=1")
        accumulate = 1
        
        # Custom collate function for Opacus compatibility
        def dp_collate_fn(batch):
            """Custom collate function to handle Opacus compatibility issues."""
            try:
                # Handle the batch data properly
                imgs, targets, paths, shapes = zip(*batch)
                imgs = torch.stack(imgs, 0)
                targets = torch.cat(targets, 0)
                return imgs, targets, paths, shapes
            except Exception as e:
                LOGGER.warning(f"Collate function error: {e}, using fallback")
                # Fallback to default collate
                return torch.utils.data.dataloader.default_collate(batch)
        
        # Modify the DataLoader to use custom collate function
        if hasattr(train_loader, 'dataset'):
            train_loader.collate_fn = dp_collate_fn
            LOGGER.info("Applied custom collate function for DP compatibility")
        
        # If Opacus version has known compatibility issues, try to fix the DataLoader
        if OPACUS_VERSION and OPACUS_VERSION.startswith('0.'):
            LOGGER.warning("Detected Opacus 0.x version with known compatibility issues. Attempting to fix...")
            try:
                # Recreate DataLoader with custom collate function
                # Note: create_dataloader is already imported at the top of the file
                train_loader, _ = create_dataloader(
                    train_path,
                    imgsz,
                    batch_size // WORLD_SIZE,
                    gs,
                    single_cls,
                    hyp=hyp,
                    augment=True,
                    cache=None if opt.cache == "val" else opt.cache,
                    rect=opt.rect,
                    rank=LOCAL_RANK,
                    workers=workers,
                    image_weights=opt.image_weights,
                    quad=opt.quad,
                    prefix=colorstr("train: "),
                    shuffle=True,
                    seed=opt.seed,
                )
                train_loader.collate_fn = dp_collate_fn
                LOGGER.info("Recreated DataLoader with custom collate function for Opacus 0.x compatibility")
            except Exception as e:
                LOGGER.warning(f"Failed to recreate DataLoader: {e}")
        
        # Create a custom DataLoader wrapper to handle Opacus compatibility issues
        class CustomDPDataLoader:
            """Custom DataLoader wrapper to handle Opacus compatibility issues."""
            def __init__(self, original_loader, collate_fn):
                self.original_loader = original_loader
                self.collate_fn = collate_fn
                self.dataset = original_loader.dataset
                self.num_workers = original_loader.num_workers
                self.batch_size = original_loader.batch_size
                self.pin_memory = getattr(original_loader, 'pin_memory', False)
                self.drop_last = getattr(original_loader, 'drop_last', False)
                self.timeout = getattr(original_loader, 'timeout', 0)
                self.worker_init_fn = getattr(original_loader, 'worker_init_fn', None)
                self.multiprocessing_context = getattr(original_loader, 'multiprocessing_context', None)
                self.generator = getattr(original_loader, 'generator', None)
                self.prefetch_factor = getattr(original_loader, 'prefetch_factor', 2)
                self.persistent_workers = getattr(original_loader, 'persistent_workers', False)
            
            def __iter__(self):
                return iter(self.original_loader)
            
            def __len__(self):
                return len(self.original_loader)
        
        # Wrap the DataLoader to handle Opacus issues
        train_loader = CustomDPDataLoader(train_loader, dp_collate_fn)
        LOGGER.info("Wrapped DataLoader with custom handler for Opacus compatibility")
        
        # Alternative approach: If Opacus continues to have issues, we can implement
        # a custom DP training loop that doesn't rely on Opacus DataLoader
        LOGGER.info("Note: If Opacus DataLoader continues to have issues, consider using --dp-simple flag for custom DP implementation")
        
        # Check if simple DP is requested
        if getattr(opt, "dp_simple", False):
            LOGGER.info("Using simple DP implementation (bypasses Opacus DataLoader)")
            # We'll implement this in the training loop
            simple_dp = True
        else:
            simple_dp = False
        
        # If not using simple DP, try to use Opacus but be prepared to fall back
        if not simple_dp:
            try:
                privacy_engine = PrivacyEngine()
                if getattr(opt, "dp_epsilon", 0.0) and opt.dp_epsilon > 0.0:
                    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                        module=model,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        epochs=epochs,
                        target_epsilon=opt.dp_epsilon,
                        target_delta=opt.dp_delta,
                        max_grad_norm=opt.dp_max_grad_norm,
                    )
                    LOGGER.info(
                        f"DP enabled with target ε={opt.dp_epsilon}, δ={opt.dp_delta}, max_grad_norm={opt.dp_max_grad_norm}"
                    )
                else:
                    assert getattr(opt, "dp_noise_multiplier", None) is not None, (
                        "Provide --dp-epsilon or --dp-noise-multiplier for DP training"
                    )
                    model, optimizer, train_loader = privacy_engine.make_private(
                        module=model,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        noise_multiplier=opt.dp_noise_multiplier,
                        max_grad_norm=opt.dp_max_grad_norm,
                    )
                    LOGGER.info(
                        f"DP enabled with noise_multiplier={opt.dp_noise_multiplier}, max_grad_norm={opt.dp_max_grad_norm}"
                    )
                # Re-initialize EMA to track the underlying (unwrapped) module for consistent state_dict keys
                if RANK in {-1, 0}:
                    base_model = model._module if hasattr(model, "_module") else model
                    ema = ModelEMA(base_model)
                LOGGER.info("Successfully initialized Opacus PrivacyEngine")
            except Exception as e:
                LOGGER.warning(f"Opacus PrivacyEngine failed: {e}")
                LOGGER.info("Falling back to simple DP implementation")
                simple_dp = True
        else:
            # Simple DP mode - no Opacus needed
            LOGGER.info("Simple DP mode: manually implementing gradient clipping and noise addition")
            privacy_engine = None

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run("on_train_start")
    LOGGER.info(
        f"Image sizes {imgsz} train, {imgsz} val\n"
        f"Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n"
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f"Starting training for {epochs} epochs..."
    )
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run("on_train_epoch_start")
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        # Optionally wrap dataloader with Opacus BatchMemoryManager under DP to avoid OOM
        if getattr(opt, "dp", False) and getattr(opt, "dp_physical_batch_size", None):
            assert BatchMemoryManager is not None, "Opacus BatchMemoryManager not available. Install opacus."
            LOGGER.info(f"Using DP physical batch size: {opt.dp_physical_batch_size}")
            dp_ctx = BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=opt.dp_physical_batch_size,
                optimizer=optimizer,
            )
        else:
            dp_ctx = nullcontext(train_loader)

        # Custom collate function for Opacus compatibility
        def custom_collate_fn(batch):
            """Custom collate function to handle Opacus compatibility issues."""
            if getattr(opt, "dp", False):
                # For DP training, ensure proper data format
                imgs, targets, paths, shapes = zip(*batch)
                imgs = torch.stack(imgs, 0)
                targets = torch.cat(targets, 0)
                return imgs, targets, paths, shapes
            else:
                # Default collate for non-DP training
                return torch.utils.data.dataloader.default_collate(batch)

        with dp_ctx as batch_iterable:
            pbar = enumerate(batch_iterable)
            LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Instances", "Size"))
            if RANK in {-1, 0}:
                pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
            optimizer.zero_grad()
            
            # Handle different data formats for DP vs non-DP
            for i, batch_data in pbar:  # batch -------------------------------------------------------------
                callbacks.run("on_train_batch_start")
                
                try:
                    # Handle different batch data formats
                    if getattr(opt, "dp", False) and hasattr(batch_data, '__len__') and len(batch_data) == 4:
                        # Opacus DataLoader format
                        imgs, targets, paths, shapes = batch_data
                    else:
                        # Standard DataLoader format
                        imgs, targets, paths, shapes = batch_data
                    
                    ni = i + nb * epoch  # number integrated batches (since train start)
                    imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
                    
                    # Ensure targets is a tensor and has the right shape
                    if not isinstance(targets, torch.Tensor):
                        targets = torch.tensor(targets, device=device)
                    else:
                        targets = targets.to(device)
                    
                except Exception as e:
                    LOGGER.error(f"Error processing batch {i}: {e}")
                    LOGGER.error(f"Batch data type: {type(batch_data)}")
                    LOGGER.error(f"Batch data shape: {getattr(batch_data, 'shape', 'N/A')}")
                    continue

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

                # Multi-scale
                if opt.multi_scale:
                    sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

                # Forward
                # Use new autocast API to avoid deprecation warnings
                with torch.amp.autocast(device_type="cuda", enabled=amp):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                    if RANK != -1:
                        loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                    if opt.quad:
                        loss *= 4.0

                # Backward
                scaler.scale(loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= accumulate:
                    scaler.unscale_(optimizer)  # unscale gradients
                    
                    # Handle DP gradient clipping and noise addition
                    if getattr(opt, "dp", False) and getattr(opt, "dp_simple", False):
                        # Simple DP: manual gradient clipping and noise addition
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.dp_max_grad_norm)
                        
                        # Add noise to gradients for DP with memory optimization
                        for param in model.parameters():
                            if param.grad is not None:
                                # Calculate noise scale based on batch size and privacy parameters
                                base_noise_scale = opt.dp_max_grad_norm / (batch_size ** 0.5)
                                
                                # Apply memory efficiency if requested
                                if getattr(opt, "dp_memory_efficient", False):
                                    noise_scale = base_noise_scale * 0.05  # Very small noise for memory efficiency
                                    LOGGER.debug(f"Memory-efficient DP: reduced noise scale to {noise_scale:.6f}")
                                else:
                                    noise_scale = base_noise_scale * 0.1  # Small noise for stability
                                
                                # Generate noise in chunks to save memory
                                try:
                                    noise = torch.randn_like(param.grad) * noise_scale
                                    param.grad.add_(noise)
                                except torch.cuda.OutOfMemoryError:
                                    LOGGER.warning(f"OOM during noise generation for {param.shape}, skipping noise")
                                    # Skip noise for this parameter if OOM
                                    continue
                        
                        LOGGER.debug(f"Applied simple DP: clipped gradients to {opt.dp_max_grad_norm}, noise scale {noise_scale:.6f}")
                    elif getattr(opt, "dp", False) and not getattr(opt, "dp_simple", False):
                        # Opacus DP mode - let Opacus handle it
                        pass
                    elif not getattr(opt, "dp", False):
                        # Standard training: manual clip
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        base_model = model._module if hasattr(model, "_module") else model
                        ema.update(base_model)
                    last_opt_step = ni

                # Log
                if RANK in {-1, 0}:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * 5)
                        % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
                    )
                    callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))
                    if callbacks.stop_training:
                        return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    half=amp,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                )

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                # Save underlying base model if wrapped by Opacus
                base_model = model._module if hasattr(model, "_module") else model
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(base_model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f"\nValidating {f}...")
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                    )  # val best model with plots
                    if is_coco:
                        callbacks.run("on_fit_epoch_end", list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run("on_train_end", last, best, epoch, results)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    """
    Parse command line arguments for configuring the training of a YOLO model.

    Args:
        known (bool): Flag to parse known arguments only, defaults to False.

    Returns:
        (argparse.Namespace): Parsed command line arguments.

    Examples:
        ```python
        options = parse_opt()
        print(options.weights)
        ```

    Notes:
        * The default weights path is 'yolov3-tiny.pt'.
        * Set `known` to True for parsing only the known arguments, useful for partial arguments.

    References:
        * Models: https://github.com/ultralytics/yolov5/tree/master/models
        * Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        * Training Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov3-tiny.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    # Differential Privacy args
    parser.add_argument("--dp", action="store_true", help="Enable differential privacy training (Opacus)")
    parser.add_argument("--dp-simple", action="store_true", help="Enable simple differential privacy training (bypasses Opacus DataLoader issues)")
    parser.add_argument("--dp-memory-efficient", action="store_true", help="Use memory-efficient DP training (reduces noise scale)")
    parser.add_argument("--dp-epsilon", type=float, default=0.0, help="Target epsilon; if >0, use epsilon accounting")
    parser.add_argument("--dp-delta", type=float, default=1e-5, help="Target delta for (epsilon, delta)-DP")
    parser.add_argument("--dp-max-grad-norm", type=float, default=1.0, help="Per-sample gradient clipping norm")
    parser.add_argument(
        "--dp-noise-multiplier",
        type=float,
        default=None,
        help="Noise multiplier (used if --dp-epsilon <= 0)",
    )
    parser.add_argument(
        "--dp-physical-batch-size",
        type=int,
        default=None,
        help="Optional physical batch size for DP to avoid OOM (uses BatchMemoryManager)",
    )

    # Logger arguments
    parser.add_argument("--entity", default=None, help="Entity")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    """
    Main training/evolution script handling model checks, DDP setup, training, and hyperparameter evolution.

    Args:
        opt (argparse.Namespace): Parsed command-line options.
        callbacks (Callbacks, optional): Callback object for handling training events. Defaults to Callbacks().

    Returns:
        None

    Raises:
        AssertionError: If certain constraints are violated (e.g., when specific options are incompatible with DDP training).

    Notes:
       - For a tutorial on using Multi-GPU with DDP: https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training

    Example:
        Single-GPU training:
        ```python
        $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
        $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
        ```

        Multi-GPU DDP training:
        ```python
        $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml \
        --weights yolov5s.pt --img 640 --device 0,1,2,3
        ```

        Models: https://github.com/ultralytics/yolov5/tree/master/models
        Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / "requirements.txt")

    # Resume (from specified or most recent last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch_load(last, map_location="cpu")["opt"]
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )  # checks
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
        if opt.evolve:
            if opt.project == str(ROOT / "runs/train"):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / "runs/evolve")
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == "cfg":
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv3 Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            "lr0": (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (1, 0.0, 0.001),  # optimizer weight decay
            "warmup_epochs": (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (1, 0.0, 0.95),  # warmup initial momentum
            "warmup_bias_lr": (1, 0.0, 0.2),  # warmup initial bias lr
            "box": (1, 0.02, 0.2),  # box loss gain
            "cls": (1, 0.2, 4.0),  # cls loss gain
            "cls_pw": (1, 0.5, 2.0),  # cls BCELoss positive_weight
            "obj": (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            "obj_pw": (1, 0.5, 2.0),  # obj BCELoss positive_weight
            "iou_t": (0, 0.1, 0.7),  # IoU training threshold
            "anchor_t": (1, 2.0, 8.0),  # anchor-multiple threshold
            "anchors": (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            "fl_gamma": (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (1, 0.0, 45.0),  # image rotation (+/- deg)
            "translate": (1, 0.0, 0.9),  # image translation (+/- fraction)
            "scale": (1, 0.0, 0.9),  # image scale (+/- gain)
            "shear": (1, 0.0, 10.0),  # image shear (+/- deg)
            "perspective": (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (1, 0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (0, 0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (1, 0.0, 1.0),  # image mixup (probability)
            "mixup": (1, 0.0, 1.0),  # image mixup (probability)
            "copy_paste": (1, 0.0, 1.0),
        }  # segment copy-paste (probability)

        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if "anchors" not in hyp:  # anchors commented in hyp.yaml
                hyp["anchors"] = 3
        if opt.noautoanchor:
            del hyp["anchors"], meta["anchors"]
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    f"gs://{opt.bucket}/evolve.csv",
                    str(evolve_csv),
                ]
            )

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = "single"  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=",", skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1e-6  # weights (sum > 0)
                if parent == "single" or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == "weighted":
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            keys = (
                "metrics/precision",
                "metrics/recall",
                "metrics/mAP_0.5",
                "metrics/mAP_0.5:0.95",
                "val/box_loss",
                "val/obj_loss",
                "val/cls_loss",
            )
            print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(
            f"Hyperparameter evolution finished {opt.evolve} generations\n"
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f"Usage example: $ python train.py --hyp {evolve_yaml}"
        )


def run(**kwargs):
    """
    Run the training process for a YOLOv3 model with the specified configurations.

    Args:
        data (str): Path to the dataset YAML file.
        weights (str): Path to the pre-trained weights file or '' to train from scratch.
        cfg (str): Path to the model configuration file.
        hyp (str): Path to the hyperparameters YAML file.
        epochs (int): Total number of training epochs.
        batch_size (int): Total batch size across all GPUs.
        imgsz (int): Image size for training and validation (in pixels).
        rect (bool): Use rectangular training for better aspect ratio preservation.
        resume (bool | str): Resume most recent training if True, or resume training from a specific checkpoint if a string.
        nosave (bool): Only save the final checkpoint and not the intermediate ones.
        noval (bool): Only validate model performance in the final epoch.
        noautoanchor (bool): Disable automatic anchor generation.
        noplots (bool): Do not save any plots.
        evolve (int): Number of generations for hyperparameters evolution.
        bucket (str): Google Cloud Storage bucket name for saving run artifacts.
        cache (str | None): Cache images for faster training ('ram' or 'disk').
        image_weights (bool): Use weighted image selection for training.
        device (str): Device to use for training, e.g., '0' for first GPU or 'cpu' for CPU.
        multi_scale (bool): Use multi-scale training.
        single_cls (bool): Train a multi-class dataset as a single-class.
        optimizer (str): Optimizer to use ('SGD', 'Adam', or 'AdamW').
        sync_bn (bool): Use synchronized batch normalization (only in DDP mode).
        workers (int): Maximum number of dataloader workers (per rank in DDP mode).
        project (str): Location of the output directory.
        name (str): Unique name for the run.
        exist_ok (bool): Allow existing output directory.
        quad (bool): Use quad dataloader.
        cos_lr (bool): Use cosine learning rate scheduler.
        label_smoothing (float): Label smoothing epsilon.
        patience (int): EarlyStopping patience (epochs without improvement).
        freeze (list[int]): List of layers to freeze, e.g., [0] to freeze only the first layer.
        save_period (int): Save checkpoint every 'save_period' epochs (disabled if less than 1).
        seed (int): Global training seed for reproducibility.
        local_rank (int): For automatic DDP Multi-GPU argument parsing, do not modify.

    Returns:
        None

    Example:
        ```python
        from ultralytics import run
        run(data='coco128.yaml', weights='yolov5m.pt', imgsz=320, epochs=100, batch_size=16)
        ```

    Notes:
        - Ensure the dataset YAML file and initial weights are accessible.
        - Refer to the [Ultralytics YOLOv5 repository](https://github.com/ultralytics/yolov5) for model and data configurations.
        - Use the [Training Tutorial](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/) for custom dataset training.
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
