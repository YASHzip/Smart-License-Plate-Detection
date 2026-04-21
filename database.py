"""
database.py — Smart License Plate Detection
Phase C: Database Integration

Provides a lightweight SQLite interface to store, query, and manage
detected license plate records.

Schema:
    detections (id, plate_number, image_path, detection_confidence,
                ocr_confidence, source, timestamp)

Usage (standalone):
    python database.py                  # Init DB and show all records
    python database.py --search MH12   # Search for plates containing MH12
    python database.py --clear          # Clear all records
"""

import sqlite3
import argparse
from datetime import datetime
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
DB_PATH  = BASE_DIR / "plates.db"


# ─── Database Initialisation ──────────────────────────────────────────────────

def init_db(db_path: str = str(DB_PATH)) -> sqlite3.Connection:
    """
    Create (or connect to) the SQLite database and ensure the
    detections table exists.
    Returns an open connection.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row          # access columns by name
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number        TEXT    NOT NULL,
            raw_ocr_text        TEXT,
            image_path          TEXT,
            detection_confidence REAL   DEFAULT 0.0,
            ocr_confidence      REAL    DEFAULT 0.0,
            source              TEXT    DEFAULT 'image',
            timestamp           TEXT    NOT NULL
        )
    """)
    conn.commit()
    return conn


# ─── CRUD Operations ──────────────────────────────────────────────────────────

def save_detection(plate_number: str,
                   raw_ocr_text: str = "",
                   image_path: str = "",
                   detection_confidence: float = 0.0,
                   ocr_confidence: float = 0.0,
                   source: str = "image",
                   db_path: str = str(DB_PATH)) -> int:
    """
    Insert a new detection record into the database.

    Args:
        plate_number:           Cleaned plate text (e.g. "MH12AB1234")
        raw_ocr_text:           Raw text returned by EasyOCR
        image_path:             Path to the source image/frame
        detection_confidence:   YOLOv5 bounding box confidence score
        ocr_confidence:         EasyOCR average confidence score
        source:                 'image', 'video', or 'webcam'
        db_path:                Path to the SQLite database file

    Returns:
        The row ID of the inserted record.
    """
    conn = init_db(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO detections
            (plate_number, raw_ocr_text, image_path,
             detection_confidence, ocr_confidence, source, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        plate_number,
        raw_ocr_text,
        image_path,
        round(detection_confidence, 4),
        round(ocr_confidence, 4),
        source,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_all(db_path: str = str(DB_PATH), limit: int = 100) -> list:
    """
    Retrieve all detection records (most recent first), up to `limit`.
    Returns a list of sqlite3.Row objects.
    """
    conn = init_db(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM detections
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def search_plate(query: str, db_path: str = str(DB_PATH)) -> list:
    """
    Search for plates containing `query` (case-insensitive).
    Returns a list of matching sqlite3.Row objects.
    """
    conn = init_db(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM detections
        WHERE plate_number LIKE ?
        ORDER BY id DESC
    """, (f"%{query.upper()}%",))
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_record_count(db_path: str = str(DB_PATH)) -> int:
    """Return the total number of detection records in the database."""
    conn = init_db(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM detections")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def clear_all(db_path: str = str(DB_PATH)) -> int:
    """Delete all detection records. Returns the number of rows deleted."""
    conn = init_db(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections")
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted


def delete_by_id(record_id: int, db_path: str = str(DB_PATH)) -> bool:
    """Delete a single record by its ID. Returns True if deleted."""
    conn = init_db(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections WHERE id = ?", (record_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# ─── Pretty-print helpers ─────────────────────────────────────────────────────

def print_rows(rows: list, title: str = "Detections"):
    """Pretty-print a list of detection rows."""
    sep = "─" * 90
    print(f"\n{'═'*90}")
    print(f"  {title}  ({len(rows)} record(s))")
    print(f"{'═'*90}")
    if not rows:
        print("  No records found.")
    else:
        print(f"  {'ID':>4}  {'Plate':<14}  {'Detect Conf':>11}  "
              f"{'OCR Conf':>8}  {'Source':<8}  {'Timestamp':<20}  Image")
        print(f"  {sep}")
        for r in rows:
            img_name = Path(r["image_path"]).name if r["image_path"] else "—"
            print(f"  {r['id']:>4}  {r['plate_number']:<14}  "
                  f"{r['detection_confidence']:>11.2%}  "
                  f"{r['ocr_confidence']:>8.2%}  "
                  f"{r['source']:<8}  {r['timestamp']:<20}  {img_name}")
    print(f"{'═'*90}\n")


# ─── Standalone CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Manage the license plate detections database.")
    parser.add_argument("--search", type=str, help="Search for a plate number substring")
    parser.add_argument("--clear",  action="store_true", help="Delete all records")
    parser.add_argument("--count",  action="store_true", help="Show total record count")
    parser.add_argument("--db",     type=str, default=str(DB_PATH), help="Path to SQLite DB file")
    args = parser.parse_args()

    # Initialise DB (creates file if not exists)
    init_db(args.db)
    print(f"[DB] Database: {args.db}")

    if args.clear:
        n = clear_all(args.db)
        print(f"[DB] Cleared {n} record(s).")
        return

    if args.count:
        print(f"[DB] Total records: {get_record_count(args.db)}")
        return

    if args.search:
        rows = search_plate(args.search, args.db)
        print_rows(rows, title=f"Search results for '{args.search}'")
    else:
        rows = get_all(args.db)
        print_rows(rows, title="All Detection Records")


if __name__ == "__main__":
    main()
