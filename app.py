from flask import Flask, render_template, request, redirect, session, url_for
from flask_socketio import SocketIO
import os, datetime, sqlite3, psycopg2
from psycopg2.extras import DictCursor

APP_SECRET = "supersecret-change-me"

app = Flask(__name__)
app.secret_key = APP_SECRET
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

DB_PATH = "rooms.db"

# =====================================================
#  DATABASE CONNECTION (Postgres if available, else SQLite)
# =====================================================
def db():
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        try:
            return psycopg2.connect(db_url, cursor_factory=DictCursor)
        except Exception as e:
            print("⚠️ Postgres connection failed, falling back to SQLite:", e)
    return sqlite3.connect(DB_PATH, check_same_thread=False)


# =====================================================
#  INITIALIZE DATABASE
# =====================================================
def init_db():
    con = db()
    cur = con.cursor()

    # Try checking if 'users' table exists already
    try:
        if isinstance(con, psycopg2.extensions.connection):
            cur.execute("SELECT to_regclass('public.users')")
            exists = cur.fetchone()[0]
        else:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            exists = cur.fetchone()
        if exists:
            print("✅ Database schema already exists, skipping init.")
            con.close()
            return
    except Exception as e:
        print("⚠️ DB check failed, forcing schema creation:", e)

    # Create tables
    try:
        if isinstance(con, psycopg2.extensions.connection):
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username VARCHAR PRIMARY KEY,
                    password VARCHAR NOT NULL,
                    role VARCHAR NOT NULL CHECK (role IN ('shareholder','receptionist'))
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rooms (
                    id SERIAL PRIMARY KEY,
                    floor INT NOT NULL,
                    room_no INT NOT NULL,
                    UNIQUE(floor, room_no)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS availability (
                    id SERIAL PRIMARY KEY,
                    room_id INT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                    date VARCHAR NOT NULL,
                    status VARCHAR NOT NULL CHECK (status IN ('vacant','occupied')),
                    UNIQUE(room_id, date)
                )
            """)
        else:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    role TEXT CHECK(role IN ('shareholder','receptionist')) NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rooms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    floor INTEGER NOT NULL,
                    room_no INTEGER NOT NULL,
                    UNIQUE(floor, room_no)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS availability (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    room_id INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('vacant','occupied')),
                    UNIQUE(room_id, date)
                )
            """)

        # Add demo users
        try:
            cur.execute("INSERT INTO users VALUES ('shareholder','123','shareholder') ON CONFLICT DO NOTHING")
            cur.execute("INSERT INTO users VALUES ('reception','123','receptionist') ON CONFLICT DO NOTHING")
        except Exception:
            cur.execute("INSERT OR IGNORE INTO users VALUES ('shareholder','123','shareholder')")
            cur.execute("INSERT OR IGNORE INTO users VALUES ('reception','123','receptionist')")

        # Add rooms
        for floor in range(1, 5):
            for suffix in range(1, 8):
                room_no = floor * 100 + suffix
                try:
                    cur.execute("INSERT INTO rooms (floor, room_no) VALUES (%s,%s) ON CONFLICT DO NOTHING", (floor, room_no))
                except Exception:
                    cur.execute("INSERT OR IGNORE INTO rooms (floor, room_no) VALUES (?,?)", (floor, room_no))

        # Add availability for 7 days
        today = datetime.date.today()
        cur.execute("SELECT id FROM rooms")
        ids = [r[0] for r in cur.fetchall()]
        for i in range(7):
            date_str = (today + datetime.timedelta(days=i)).isoformat()
            for room_id in ids:
                try:
                    cur.execute("INSERT INTO availability (room_id, date, status) VALUES (%s,%s,'vacant') ON CONFLICT DO NOTHING", (room_id, date_str))
                except Exception:
                    cur.execute("INSERT OR IGNORE INTO availability (room_id, date, status) VALUES (?,?, 'vacant')", (room_id, date_str))

        con.commit()
        print("✅ Database initialized successfully and all rooms added!")
    except Exception as e:
        print("❌ DB Initialization Error:", e)
    finally:
        con.close()


# =====================================================
#  HELPERS
# =====================================================
def next_7_dates():
    today = datetime.date.today()
    return [(today + datetime.timedelta(days=i)) for i in range(7)]


def get_availability_for_date(date_str):
    con = db()
    cur = con.cursor()
    try:
        cur.execute("""
            SELECT rooms.id, rooms.floor, rooms.room_no, availability.status
            FROM rooms
            LEFT JOIN availability ON availability.room_id = rooms.id AND availability.date = %s
            ORDER BY rooms.floor ASC, rooms.room_no ASC
        """, (date_str,))
    except Exception:
        cur.execute("""
            SELECT rooms.id, rooms.floor, rooms.room_no, availability.status
            FROM rooms
            LEFT JOIN availability ON availability.room_id = rooms.id AND availability.date = ?
            ORDER BY rooms.floor ASC, rooms.room_no ASC
        """, (date_str,))
    rows = cur.fetchall()
    con.close()

    floors = {1: [], 2: [], 3: [], 4: []}
    for room_id, floor, room_no, status in rows:
        floors[floor].append({
            "id": room_id,
            "room_no": room_no,
            "status": status or "vacant"
        })
    return floors


# =====================================================
#  ROUTES
# =====================================================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        con = db()
        cur = con.cursor()
        try:
            cur.execute("SELECT role FROM users WHERE username=%s AND password=%s", (username, password))
        except Exception:
            cur.execute("SELECT role FROM users WHERE username=? AND password=?", (username, password))
        row = cur.fetchone()
        con.close()
        if row:
            session["username"] = username
            session["role"] = row[0]
            return redirect(url_for("dashboard"))
        else:
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
    floors = get_availability_for_date(selected)
    dates = next_7_dates()
    return render_template(
        "dashboard.html",
        floors=floors,
        dates=dates,
        selected_date=selected,
        role=session["role"],
        user=session["username"]
    )


@app.post("/api/toggle")
def api_toggle():
    if "role" not in session or session["role"] != "receptionist":
        return {"ok": False, "error": "forbidden"}, 403
    data = request.get_json(force=True)
    room_id = int(data["room_id"])
    date_str = data["date"]
    con = db()
    cur = con.cursor()
    try:
        cur.execute("SELECT status FROM availability WHERE room_id=%s AND date=%s", (room_id, date_str))
    except Exception:
        cur.execute("SELECT status FROM availability WHERE room_id=? AND date=?", (room_id, date_str))
    row = cur.fetchone()
    current = row[0] if row else "vacant"
    new_status = "vacant" if current == "occupied" else "occupied"
    try:
        cur.execute("UPDATE availability SET status=%s WHERE room_id=%s AND date=%s", (new_status, room_id, date_str))
    except Exception:
        cur.execute("UPDATE availability SET status=? WHERE room_id=? AND date=?", (new_status, room_id, date_str))
    con.commit()
    con.close()
    socketio.emit("room_updated", {"room_id": room_id, "date": date_str, "status": new_status})
    return {"ok": True, "status": new_status}


@app.route("/init")
def initialize():
    init_db()
    return "✅ Database initialized successfully and all rooms added!"


if __name__ == "__main__":
    if not os.environ.get("RENDER"):  # only run locally
        init_db()
    socketio.run(app, host="0.0.0.0", port=5000)

