from flask import Flask, render_template, request, redirect, session, url_for
from flask_socketio import SocketIO
import os, datetime, sqlite3
import psycopg2
from psycopg2.extras import DictCursor

APP_SECRET = "supersecret-change-me"

app = Flask(__name__)
app.secret_key = APP_SECRET
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

DB_PATH = "rooms.db"


# --- DB Connection ---
def db():
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url, cursor_factory=DictCursor)
    else:
        return sqlite3.connect(DB_PATH, check_same_thread=False)


# --- Initialize Database ---
def init_db():
    """Creates all tables and seed data (both local SQLite & Postgres)."""
    con = db()
    cur = con.cursor()
    db_url = os.environ.get("DATABASE_URL")
    is_postgres = bool(db_url)

    if is_postgres:
        # --- CREATE TABLES (Postgres syntax) ---
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
                floor INTEGER NOT NULL,
                room_no INTEGER NOT NULL,
                UNIQUE(floor, room_no)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS availability (
                id SERIAL PRIMARY KEY,
                room_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('vacant','occupied')),
                UNIQUE(room_id, date)
            )
        """)
        con.commit()

        # --- DEFAULT USERS ---
        cur.execute("INSERT INTO users (username,password,role) VALUES (%s,%s,%s) ON CONFLICT (username) DO NOTHING",
                    ("shareholder", "123", "shareholder"))
        cur.execute("INSERT INTO users (username,password,role) VALUES (%s,%s,%s) ON CONFLICT (username) DO NOTHING",
                    ("reception", "123", "receptionist"))

        # --- ROOMS ---
        for floor in range(1, 5):
            for suffix in range(1, 8):
                room_no = floor * 100 + suffix
                cur.execute("INSERT INTO rooms (floor,room_no) VALUES (%s,%s) ON CONFLICT DO NOTHING", (floor, room_no))

        # --- AVAILABILITY ---
        today = datetime.date.today()
        cur.execute("SELECT id FROM rooms")
        room_ids = [r[0] for r in cur.fetchall()]
        for i in range(7):
            date_str = (today + datetime.timedelta(days=i)).isoformat()
            for room_id in room_ids:
                cur.execute("""
                    INSERT INTO availability (room_id,date,status)
                    VALUES (%s,%s,'vacant')
                    ON CONFLICT DO NOTHING
                """, (room_id, date_str))

    else:
        # --- CREATE TABLES (SQLite syntax) ---
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
        con.commit()

        # --- DEFAULT USERS ---
        cur.execute("INSERT OR IGNORE INTO users VALUES ('shareholder','123','shareholder')")
        cur.execute("INSERT OR IGNORE INTO users VALUES ('reception','123','receptionist')")

        # --- ROOMS ---
        for floor in range(1, 5):
            for suffix in range(1, 8):
                room_no = floor * 100 + suffix
                cur.execute("INSERT OR IGNORE INTO rooms (floor, room_no) VALUES (?,?)", (floor, room_no))

        # --- AVAILABILITY ---
        today = datetime.date.today()
        cur.execute("SELECT id FROM rooms")
        room_ids = [r[0] for r in cur.fetchall()]
        for i in range(7):
            date_str = (today + datetime.timedelta(days=i)).isoformat()
            for room_id in room_ids:
                cur.execute(
                    "INSERT OR IGNORE INTO availability (room_id,date,status) VALUES (?,?,'vacant')",
                    (room_id, date_str),
                )

    con.commit()
    con.close()
    print("✅ Database initialized successfully and all rooms added!")


# --- Helper Functions ---
def next_7_dates():
    today = datetime.date.today()
    return [(today + datetime.timedelta(days=i)) for i in range(7)]


def get_availability_for_date(date_str):
    con = db()
    cur = con.cursor()
    db_url = os.environ.get("DATABASE_URL")

    if db_url:
        cur.execute("""
            SELECT rooms.id, rooms.floor, rooms.room_no,
                   COALESCE(availability.status, 'vacant') AS status
            FROM rooms
            LEFT JOIN availability ON availability.room_id = rooms.id
            AND availability.date = %s
            ORDER BY rooms.floor ASC, rooms.room_no ASC
        """, (date_str,))
    else:
        cur.execute("""
            SELECT rooms.id, rooms.floor, rooms.room_no,
                   COALESCE(availability.status, 'vacant') AS status
            FROM rooms
            LEFT JOIN availability ON availability.room_id = rooms.id
            AND availability.date = ?
            ORDER BY rooms.floor ASC, rooms.room_no ASC
        """, (date_str,))
    rows = cur.fetchall()
    con.close()

    floors = {1: [], 2: [], 3: [], 4: []}
    for room_id, floor, room_no, status in rows:
        floors[int(floor)].append({"id": room_id, "room_no": room_no, "status": status})
    return floors


# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def login():
    db_url = os.environ.get("DATABASE_URL")
    con = db()
    cur = con.cursor()

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        try:
            if db_url:
                cur.execute("SELECT role FROM users WHERE username=%s AND password=%s", (username, password))
            else:
                cur.execute("SELECT role FROM users WHERE username=? AND password=?", (username, password))
            row = cur.fetchone()
        except Exception as e:
            print("❌ DB Query Failed:", e)
            con.close()
            return render_template("login.html", error="Database error — try refreshing.")
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

    # Ensure next 7 days availability always exists
    con = db()
    cur = con.cursor()
    cur.execute("SELECT id FROM rooms")
    room_ids = [r[0] for r in cur.fetchall()]
    today = datetime.date.today()
    db_url = os.environ.get("DATABASE_URL")

    if db_url:
        insert_query = "INSERT INTO availability (room_id,date,status) VALUES (%s,%s,'vacant') ON CONFLICT DO NOTHING"
    else:
        insert_query = "INSERT OR IGNORE INTO availability (room_id,date,status) VALUES (?, ?, 'vacant')"

    for i in range(7):
        date_str = (today + datetime.timedelta(days=i)).isoformat()
        for room_id in room_ids:
            cur.execute(insert_query, (room_id, date_str))

    con.commit()
    con.close()

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
    db_url = os.environ.get("DATABASE_URL")

    if db_url:
        cur.execute("SELECT status FROM availability WHERE room_id=%s AND date=%s", (room_id, date_str))
    else:
        cur.execute("SELECT status FROM availability WHERE room_id=? AND date=?", (room_id, date_str))

    row = cur.fetchone()
    current = row[0] if row else "vacant"
    new_status = "vacant" if current == "occupied" else "occupied"

    if db_url:
        cur.execute("""
            INSERT INTO availability (room_id, date, status)
            VALUES (%s, %s, %s)
            ON CONFLICT (room_id, date)
            DO UPDATE SET status = EXCLUDED.status
        """, (room_id, date_str, new_status))
    else:
        cur.execute("""
            INSERT OR REPLACE INTO availability (room_id, date, status)
            VALUES (?, ?, ?)
        """, (room_id, date_str, new_status))

    con.commit()
    con.close()

    socketio.emit("room_updated", {"room_id": room_id, "date": date_str, "status": new_status})
    return {"ok": True, "status": new_status}


@app.route("/init")
def initialize():
    init_db()
    return "✅ Database initialized successfully and all rooms added!"


if __name__ == "__main__":
    init_db()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
