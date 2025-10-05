# gunicorn.conf.py
workers = 1
worker_class = "geventwebsocket.gunicorn.workers.GeventWebSocketWorker"
timeout = 120          # default is 30 seconds
graceful_timeout = 60
keepalive = 30
