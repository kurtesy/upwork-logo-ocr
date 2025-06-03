import sqlite3
import os
import logging
from fastapi import HTTPException

from .config import SQLITE_DB_PATH

logger = logging.getLogger(__name__)

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False) # check_same_thread=False for FastAPI
        conn.row_factory = sqlite3.Row # Access columns by name
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to SQLite database {SQLITE_DB_PATH}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not connect to the database: {e}")

def check_db_initialized():
    if not os.path.exists(SQLITE_DB_PATH):
        logger.warning(f"SQLite database file not found at {SQLITE_DB_PATH}. "
                       "Please run the OCR processing script to create and populate it.")