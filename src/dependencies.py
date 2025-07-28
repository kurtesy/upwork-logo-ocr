import logging
import sqlite3
from fastapi import Security, HTTPException, status, Depends
from fastapi.security import APIKeyHeader

from .config import API_KEY, API_KEY_NAME
from .database import get_db_connection

logger = logging.getLogger(__name__)
api_key_header_auth_scheme = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_admin_api_key(api_key_header: str = Security(api_key_header_auth_scheme)):
    """
    Checks if the provided API key matches the master/admin API key from config.
    This should be used to protect administrative endpoints.
    """
    if not API_KEY:
        # This case means the server itself is not configured with an API_KEY
        logger.error("Server-side API_KEY is not configured. Admin authentication cannot be performed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin API Key authentication is not configured correctly on the server."
        )
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Admin API Key"
        )

async def get_current_api_key(
    api_key_header: str = Security(api_key_header_auth_scheme),
    db: sqlite3.Connection = Depends(get_db_connection)
):
    """
    Checks if the provided API key exists in the database.
    """
    cursor = db.cursor()
    try:
        cursor.execute("SELECT 1 FROM api_keys WHERE key = ?", (api_key_header,))
        if cursor.fetchone():
            return api_key_header
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key"
            )
    except sqlite3.Error as e:
        logger.error(f"Database error during API key validation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not validate API key due to a database error."
        )
    finally:
        db.close()