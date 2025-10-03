from flask import Flask, render_template, request, redirect, session, url_for
from flask_socketio import SocketIO, emit
import sqlite3, os, datetime
import os
import psycopg2
from psycopg2.extras import DictCursor
APP_SECRET = "supersecret-change-me"

app = Flask(__name__)
app.secret_key = APP_SECRET
# Force threading mode so no extra async deps are needed
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

DB_PATH = "rooms.db"

def db():
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        # Postgres mode
        conn = psycopg2.connect(db_url, cursor_factory=DictCursor)
        return conn
    else:
        # SQLite fallback (for local testing)
        return sqlite3.connect("rooms.db", check_same_thread=False)

def init_db():
    first_time = not os.path.exists(DB_PATH)
    con = db()
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        role     TEXT CHECK(role IN ('shareholder','receptionist')) NOT NULL
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS rooms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        floor INTEGER NOT NULL,
        room_no INTEGER NOT NULL,
        UNIQUE(floor, room_no)
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS availability (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        room_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('vacant','occupied')),
        UNIQUE(room_id, date),
        FOREIGN KEY(room_id) REFERENCES rooms(id) ON DELETE CASCADE
    )""")

    cur.execute("INSERT OR IGNORE INTO users VALUES (?,?,?)", ("shareholder","123","shareholder"))
    cur.execute("INSERT OR IGNORE INTO users VALUES (?,?,?)", ("reception","123","receptionist"))

    for floor in range(1, 5):
        for suffix in range(1, 8):
            room_no = floor * 100 + suffix
            cur.execute("INSERT OR IGNORE INTO rooms (floor, room_no) VALUES (?,?)", (floor, room_no))

    today = datetime.date.today()
    cur.execute("SELECT id FROM rooms")
    room_ids = [r[0] for r in cur.fetchall()]
    for i in range(7):
        date_str = (today + datetime.timedelta(days=i)).isoformat()
        for room_id in room_ids:
            cur.execute("""
                INSERT OR IGNORE INTO availability(room_id, date, status)
                VALUES (?,?, 'vacant')
            """, (room_id, date_str))

    con.commit()
    con.close()

def next_7_dates():
    """Return today + next 6 days as list of date objects."""
    today = datetime.date.today()
    return [(today + datetime.timedelta(days=i)) for i in range(7)]

def get_availability_for_date(date_str):
    con = db()
    cur = con.cursor()
    cur.execute("""
        SELECT rooms.id, rooms.floor, rooms.room_no, availability.status
        FROM rooms
        JOIN availability ON availability.room_id = rooms.id
        WHERE availability.date = ?
        ORDER BY rooms.floor ASC, rooms.room_no ASC
    """, (date_str,))
    rows = cur.fetchall()
    con.close()

    floors = {1:[],2:[],3:[],4:[]}
    for room_id, floor, room_no, status in rows:
        floors[floor].append({
            "id": room_id,
            "room_no": room_no,
            "status": status
        })
    return floors

@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","").strip()
        con = db()
        cur = con.cursor()
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

    con = db()
    cur = con.cursor()

    # --- NEW: Always ensure next 7 days exist in DB ---
    today = datetime.date.today()
    cur.execute("SELECT id FROM rooms")
    room_ids = [r[0] for r in cur.fetchall()]
    for i in range(7):
        date_str = (today + datetime.timedelta(days=i)).isoformat()
        for room_id in room_ids:
            cur.execute(
                "INSERT OR IGNORE INTO availability(room_id, date, status) VALUES (?,?, 'vacant')",
                (room_id, date_str)
            )
    con.commit()
    con.close()
    # --- END FIX ---

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

# --- NEW: HTTP API that toggles and then broadcasts via Socket.IO ---
@app.post("/api/toggle")
def api_toggle():
    if "role" not in session or session["role"] != "receptionist":
        return {"ok": False, "error": "forbidden"}, 403

    data = request.get_json(force=True)
    room_id = int(data["room_id"])
    date_str = data["date"]

    con = db()
    cur = con.cursor()
    cur.execute("SELECT status FROM availability WHERE room_id=? AND date=?", (room_id, date_str))
    row = cur.fetchone()
    if not row:
        cur.execute("INSERT INTO availability(room_id, date, status) VALUES (?,?, 'vacant')",
                    (room_id, date_str))
        current = "vacant"
    else:
        current = row[0]

    new_status = "vacant" if current == "occupied" else "occupied"
    cur.execute("UPDATE availability SET status=? WHERE room_id=? AND date=?", (new_status, room_id, date_str))
    con.commit()
    con.close()

    # notify all clients (shareholders & receptionist)
    socketio.emit("room_updated", {"room_id": room_id, "date": date_str, "status": new_status})
    return {"ok": True, "status": new_status}

if __name__ == "__main__":
    init_db()
    socketio.run(app, host="127.0.0.1", port=5000, debug=True)