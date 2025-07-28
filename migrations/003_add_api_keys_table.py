import logging
import os
import sqlite3
import sys

# This allows the script to be run from the root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database import SQLITE_DB_PATH, get_db_connection
from src.config import API_KEY

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CREATE_API_KEYS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS api_keys (
    key TEXT PRIMARY KEY NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def apply_migration():
    """Applies the database migration to create the api_keys table."""
    logging.info(f"Applying api_keys migration to database: {SQLITE_DB_PATH}")
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        logging.info("Creating table 'api_keys' if it doesn't exist...")
        cursor.execute(CREATE_API_KEYS_TABLE_SQL)

        if API_KEY:
            logging.info("Checking for master API key in 'api_keys' table...")
            cursor.execute("SELECT 1 FROM api_keys WHERE key = ?", (API_KEY,))
            if cursor.fetchone() is None:
                logging.info("Inserting master API key from environment into 'api_keys' table.")
                cursor.execute(
                    "INSERT INTO api_keys (key, description) VALUES (?, ?)",
                    (API_KEY, "Master API Key from environment")
                )
            else:
                logging.info("Master API key already exists in table.")
        else:
            logging.warning("No API_KEY found in environment. No master key will be added.")

        conn.commit()
        logging.info("API keys migration applied successfully.")

    except sqlite3.Error as e:
        logging.error(f"An SQLite error occurred during api_keys migration: {e}", exc_info=True)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    apply_migration()