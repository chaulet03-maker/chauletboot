import csv, os

def _ensure_header(path, fieldnames):
    """Upgrade header to union of old+new fields, preserving rows."""
    fields = list(fieldnames or [])
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
        return fields
    rows = []
    old = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        old = list(r.fieldnames or [])
        for row in r:
            rows.append(row)
    seen = set()
    new = []
    for k in (old or []) + fields:
        if k and k not in seen:
            seen.add(k); new.append(k)
    if new != old:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=new)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in new})
    return new

def _append(csv_dir, filename, row):
    path = os.path.join(csv_dir, filename)
    header = _ensure_header(path, list(row.keys()))
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writerow({k: row.get(k, "") for k in header})

def append_trade_csv(csv_dir, row): _append(csv_dir, "trades.csv", row)
def append_equity_csv(csv_dir, row): _append(csv_dir, "equity.csv", row)
def append_decision_csv(csv_dir, row): _append(csv_dir, "decisions.csv", row)
