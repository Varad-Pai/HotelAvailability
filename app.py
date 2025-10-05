import os
import datetime
import sqlite3
from contextlib import contextmanager

from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from flask_socketio import SocketIO
import psycopg2
from psycopg2.extras import DictCursor
from psycopg2 import pool as pgpool

APP_SECRET = os.environ.get("APP_SECRET", "supersecret-change-me")
DATABASE_URL = os.environ.get("DATABASE_URL")  # Set by Render when using Postgres
USE_POSTGRES = bool(DATABASE_URL)

app = Flask(__name__)
app.secret_key = APP_SECRET

# Use gevent-friendly async mode — gunicorn will use geventwebsocket worker
socketio = SocketIO(app, async_mode="gevent", cors_allowed_origins="*")

DB_PATH = "rooms.db"

# ======= Connection pooling / sqlite fallback =======
pg_pool = None
sqlite_conn = None

def init_db_pool():
    global pg_pool, sqlite_conn
    if USE_POSTGRES:
        # pool: min 1, max 10 connections
        pg_pool = pgpool.SimpleConnectionPool(1, 10, DATABASE_URL, cursor_factory=DictCursor)
    else:
        # single SQLite connection for local testing (thread safe)
        sqlite_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        sqlite_conn.row_factory = sqlite3.Row

@contextmanager
def get_conn_cursor():
    """
    Yields (conn, cur) and ensures putconn for postgres or just close cursor for sqlite.
    """
    if USE_POSTGRES:
        conn = pg_pool.getconn()
        try:
            cur = conn.cursor()
            yield conn, cur
        finally:
            # commit is caller's responsibility
            cur.close()
            pg_pool.putconn(conn)
    else:
        # sqlite single connection
        conn = sqlite_conn
        cur = conn.cursor()
        try:
            yield conn, cur
        finally:
            # do not close sqlite_conn here
            cur.close()

# ======= Initialization (safe, idempotent) =======
def init_db():
    """
    Create tables if missing; seed users, rooms, availability.
    Safe to call multiple times (it checks for existing schema).
    """
    init_db_pool()

    # Check whether users table exists; if it exists skip heavy inserts
    with get_conn_cursor() as (conn, cur):
        try:
            if USE_POSTGRES:
                cur.execute("SELECT to_regclass('public.users')")
                exists = cur.fetchone()[0]
            else:
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
                exists = cur.fetchone()
        except Exception:
            exists = None

        if exists:
            print("✅ Schema already present — skipping init_db heavy work.")
            return

    # Create tables and seed
    with get_conn_cursor() as (conn, cur):
        if USE_POSTGRES:
            # Postgres DDL
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username VARCHAR PRIMARY KEY,
                    password VARCHAR NOT NULL,
                    role VARCHAR NOT NULL CHECK (role IN ('shareholder','receptionist'))
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rooms (
                    id SERIAL PRIMARY KEY,
                    floor INTEGER NOT NULL,
                    room_no INTEGER NOT NULL,
                    UNIQUE(floor, room_no)
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS availability (
                    id SERIAL PRIMARY KEY,
                    room_id INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                    date TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('vacant','occupied')),
                    UNIQUE(room_id, date)
                );
            """)
        else:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    role TEXT CHECK(role IN ('shareholder','receptionist')) NOT NULL
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rooms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    floor INTEGER NOT NULL,
                    room_no INTEGER NOT NULL,
                    UNIQUE(floor, room_no)
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS availability (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    room_id INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('vacant','occupied')),
                    UNIQUE(room_id, date)
                );
            """)

        # seed users (idempotent)
        if USE_POSTGRES:
            cur.execute("INSERT INTO users (username,password,role) VALUES (%s,%s,%s) ON CONFLICT (username) DO NOTHING",
                        ("shareholder", "123", "shareholder"))
            cur.execute("INSERT INTO users (username,password,role) VALUES (%s,%s,%s) ON CONFLICT (username) DO NOTHING",
                        ("reception", "123", "receptionist"))
        else:
            cur.execute("INSERT OR IGNORE INTO users VALUES ('shareholder','123','shareholder')")
            cur.execute("INSERT OR IGNORE INTO users VALUES ('reception','123','receptionist')")

        # seed rooms
        for floor in range(1, 5):
            for suffix in range(1, 8):
                room_no = floor * 100 + suffix
                try:
                    if USE_POSTGRES:
                        cur.execute("INSERT INTO rooms (floor, room_no) VALUES (%s, %s) ON CONFLICT DO NOTHING", (floor, room_no))
                    else:
                        cur.execute("INSERT OR IGNORE INTO rooms (floor, room_no) VALUES (?, ?)", (floor, room_no))
                except Exception:
                    # ignore any unique constraint anomalies
                    pass

        # seed availability for next 7 days
        today = datetime.date.today()
        # fetch room ids
        if USE_POSTGRES:
            cur.execute("SELECT id FROM rooms")
            room_ids = [r[0] for r in cur.fetchall()]
        else:
            cur.execute("SELECT id FROM rooms")
            room_ids = [r[0] for r in cur.fetchall()]

        for i in range(7):
            date_str = (today + datetime.timedelta(days=i)).isoformat()
            for room_id in room_ids:
                try:
                    if USE_POSTGRES:
                        cur.execute("INSERT INTO availability (room_id, date, status) VALUES (%s,%s,'vacant') ON CONFLICT DO NOTHING",
                                    (room_id, date_str))
                    else:
                        cur.execute("INSERT OR IGNORE INTO availability (room_id, date, status) VALUES (?,?, 'vacant')",
                                    (room_id, date_str))
                except Exception:
                    pass

        conn.commit()
    print("✅ Database initialized successfully and all rooms added!")

# ======= Simple in-memory cache (short TTL) to speed GET dashboard =======
_cache = {"floors": None, "date": None, "ts": None}
CACHE_TTL_SECONDS = 2  # very short; Socket.IO updates keep clients fresh

def cache_get(date_str):
    import time
    if _cache["date"] == date_str and _cache["floors"] is not None and (time.time() - (_cache["ts"] or 0)) < CACHE_TTL_SECONDS:
        return _cache["floors"]
    return None

def cache_set(date_str, floors):
    import time
    _cache["date"] = date_str
    _cache["floors"] = floors
    _cache["ts"] = time.time()

def cache_invalidate():
    _cache["floors"] = None
    _cache["date"] = None
    _cache["ts"] = None

# ======= Helpers =======
def next_7_dates():
    today = datetime.date.today()
    return [(today + datetime.timedelta(days=i)) for i in range(7)]

def get_availability_for_date(date_str):
    # try cache first
    cached = cache_get(date_str)
    if cached is not None:
        return cached

    with get_conn_cursor() as (conn, cur):
        try:
            if USE_POSTGRES:
                cur.execute("""
                    SELECT rooms.id, rooms.floor, rooms.room_no,
                           COALESCE(availability.status, 'vacant') AS status
                    FROM rooms
                    LEFT JOIN availability
                      ON availability.room_id = rooms.id
                      AND availability.date = %s
                    ORDER BY rooms.floor ASC, rooms.room_no ASC
                """, (date_str,))
                rows = cur.fetchall()
            else:
                cur.execute("""
                    SELECT rooms.id, rooms.floor, rooms.room_no,
                           COALESCE(availability.status, 'vacant') AS status
                    FROM rooms
                    LEFT JOIN availability
                      ON availability.room_id = rooms.id
                      AND availability.date = ?
                    ORDER BY rooms.floor ASC, rooms.room_no ASC
                """, (date_str,))
                rows = cur.fetchall()
        except Exception:
            rows = []

    # normalize rows to list of tuples
    normalized = []
    for r in rows:
        # psycopg2 returns dict-like row objects; sqlite returns tuples
        try:
            normalized.append((int(r[0]), int(r[1]), int(r[2]), r[3]))
        except Exception:
            # handle row as mapping
            normalized.append((int(r["id"]), int(r["floor"]), int(r["room_no"]), r["status"]))

    floors = {1: [], 2: [], 3: [], 4: []}
    for room_id, floor, room_no, status in normalized:
        floors[int(floor)].append({"id": room_id, "room_no": room_no, "status": status or "vacant"})

    cache_set(date_str, floors)
    return floors

# ======= Routes =======
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        with get_conn_cursor() as (conn, cur):
            try:
                if USE_POSTGRES:
                    cur.execute("SELECT role FROM users WHERE username=%s AND password=%s", (username, password))
                else:
                    cur.execute("SELECT role FROM users WHERE username=? AND password=?", (username, password))
                r = cur.fetchone()
            except Exception as e:
                print("❌ DB Query Failed:", e)
                r = None
        if r:
            # row may be tuple or dict-like
            role = r[0] if isinstance(r, (list, tuple)) else (r[0] if len(r) > 0 else None)
            session["username"] = username
            session["role"] = role
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid credentials")
    if "username" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html", error=None)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))

    selected = request.args.get("date") or datetime.date.today().isoformat()

    # Ensure availability rows exist for next 7 days (idempotent)
    with get_conn_cursor() as (conn, cur):
        if USE_POSTGRES:
            insert_q = "INSERT INTO availability (room_id, date, status) VALUES (%s,%s,'vacant') ON CONFLICT DO NOTHING"
            cur.execute("SELECT id FROM rooms")
            room_ids = [r[0] for r in cur.fetchall()]
        else:
            insert_q = "INSERT OR IGNORE INTO availability (room_id, date, status) VALUES (?, ?, 'vacant')"
            cur.execute("SELECT id FROM rooms")
            room_ids = [r[0] for r in cur.fetchall()]

        today = datetime.date.today()
        for i in range(7):
            date_str = (today + datetime.timedelta(days=i)).isoformat()
            for rid in room_ids:
                try:
                    cur.execute(insert_q, (rid, date_str))
                except Exception:
                    pass
        conn.commit()

    floors = get_availability_for_date(selected)
    dates = next_7_dates()
    return render_template("dashboard.html",
                           floors=floors,
                           dates=dates,
                           selected_date=selected,
                           role=session.get("role"),
                           user=session.get("username"))

@app.post("/api/toggle")
def api_toggle():
    # only receptionist can toggle
    if "role" not in session or session["role"] != "receptionist":
        return jsonify({"ok": False, "error": "forbidden"}), 403

    data = request.get_json(force=True)
    room_id = int(data["room_id"])
    date_str = data["date"]

    with get_conn_cursor() as (conn, cur):
        try:
            if USE_POSTGRES:
                cur.execute("SELECT status FROM availability WHERE room_id=%s AND date=%s", (room_id, date_str))
            else:
                cur.execute("SELECT status FROM availability WHERE room_id=? AND date=?", (room_id, date_str))
            row = cur.fetchone()
            current = row[0] if row else "vacant"
        except Exception:
            current = "vacant"

        new_status = "vacant" if current == "occupied" else "occupied"

        # Upsert
        try:
            if USE_POSTGRES:
                cur.execute("""
                    INSERT INTO availability (room_id, date, status)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (room_id, date) DO UPDATE SET status = EXCLUDED.status
                """, (room_id, date_str, new_status))
            else:
                cur.execute("INSERT OR REPLACE INTO availability (room_id, date, status) VALUES (?, ?, ?)",
                            (room_id, date_str, new_status))
            conn.commit()
        except Exception as e:
            print("DB toggle error:", e)
            return jsonify({"ok": False, "error": "db_error"}), 500

    # invalidate cache and broadcast
    cache_invalidate()
    socketio.emit("room_updated", {"room_id": room_id, "date": date_str, "status": new_status}, broadcast=True)
    return jsonify({"ok": True, "status": new_status})

# manual init endpoint (call once on first deployment)
@app.route("/init")
def initialize():
    init_db()
    return "✅ Database initialized successfully and all rooms added!"

# ======= Run =======
if __name__ == "__main__":
    # local init for dev
    init_db()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
