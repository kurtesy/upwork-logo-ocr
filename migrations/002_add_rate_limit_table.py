import logging
import os
import sqlite3
import sys

# This allows the script to be run from the root directory (e.g., python migrations/002_...)
# and find the 'src' module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database import SQLITE_DB_PATH, get_db_connection

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# SQL commands for the migration
CREATE_RATE_LIMIT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS request_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_identifier TEXT NOT NULL,
    timestamp REAL NOT NULL
);
"""

CREATE_RATE_LIMIT_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_request_logs_client_timestamp 
ON request_logs (client_identifier, timestamp);
"""

def apply_migration():
    """Applies the database migration to create the rate limiting table."""
    logging.info(f"Applying rate limiter migration to database: {SQLITE_DB_PATH}")
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        logging.info("Creating table 'request_logs' for rate limiting if it doesn't exist...")
        cursor.execute(CREATE_RATE_LIMIT_TABLE_SQL)

        logging.info("Creating index 'idx_request_logs_client_timestamp' if it doesn't exist...")
        cursor.execute(CREATE_RATE_LIMIT_INDEX_SQL)

        conn.commit()
        logging.info("Rate limiter migration applied successfully.")

    except sqlite3.Error as e:
        logging.error(f"An SQLite error occurred during rate limiter migration: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    apply_migration()