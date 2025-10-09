import sqlite3, os

def ensure_db(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT, side TEXT, qty REAL, price REAL, lev INTEGER, fee REAL, pnl REAL, note TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS equity (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, equity REAL, pnl REAL)")
    conn.commit()
    conn.close()

def insert_trade(path, row):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("INSERT INTO trades (ts,symbol,side,qty,price,lev,fee,pnl,note) VALUES (?,?,?,?,?,?,?,?,?)",
        (row["ts"],row["symbol"],row["side"],row["qty"],row["price"],row["lev"],row["fee"],row.get("pnl",0.0),row.get("note","")))
    conn.commit(); conn.close()

def insert_equity(path, row):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("INSERT INTO equity (ts,equity,pnl) VALUES (?,?,?)",
        (row["ts"],row["equity"],row["pnl"]))
    conn.commit(); conn.close()
