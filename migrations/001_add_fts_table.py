import logging
import os
import sqlite3
import sys

# This allows the script to be run from the root directory (e.g., python migrations/001_...)
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
CREATE_FTS_TABLE_SQL = """
CREATE VIRTUAL TABLE ocr_results_fts USING fts5(
    image_identifier,
    source_type,
    extracted_text,
    content='ocr_results',
    content_rowid='rowid'
);
"""

POPULATE_FTS_TABLE_SQL = """
INSERT INTO ocr_results_fts(rowid, image_identifier, source_type, extracted_text)
SELECT rowid, image_identifier, source_type, extracted_text FROM ocr_results;
"""


def apply_migration():
    """Applies the database migration to create and populate the FTS5 table."""
    logging.info(f"Applying migration to database: {SQLITE_DB_PATH}")
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Prerequisite check: Ensure the source table 'ocr_results' exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ocr_results';")
        if not cursor.fetchone():
            logging.error("Source table 'ocr_results' not found. Cannot apply FTS migration.")
            return

        # Step 1: Create the FTS table if it doesn't exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ocr_results_fts';")
        if cursor.fetchone():
            logging.info("'ocr_results_fts' table already exists. Skipping creation.")
        else:
            logging.info("Creating FTS5 virtual table 'ocr_results_fts'...")
            cursor.execute(CREATE_FTS_TABLE_SQL)
            logging.info("Table 'ocr_results_fts' created successfully.")

        # Step 2: Populate the FTS table if it's empty
        logging.info("Rebuilding FTS index to ensure it is up to date...")
        # The 'rebuild' command is an efficient way to populate or update the index.
        cursor.execute("INSERT INTO ocr_results_fts(ocr_results_fts) VALUES('rebuild');")
        conn.commit()
        logging.info("FTS index rebuild completed successfully.")

    except sqlite3.Error as e:
        logging.error(f"An SQLite error occurred: {e}")
        if conn:
            conn.rollback()
            logging.info("Transaction rolled back due to error.")
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")


if __name__ == "__main__":
    apply_migration()