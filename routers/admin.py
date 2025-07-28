import logging
import secrets
import sqlite3
from fastapi import APIRouter, Depends, HTTPException, status, Body

from src.dependencies import get_admin_api_key
from src.database import get_db_connection

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
    dependencies=[Depends(get_admin_api_key)] # Protect all routes in this router
)

@router.post("/generate-api-key", summary="Generate a new API Key")
def generate_api_key(
    description: str = Body("Generated API Key", embed=True),
    db: sqlite3.Connection = Depends(get_db_connection)
):
    """
    Generates a new secure API key, stores it in the database, and returns it.
    This endpoint is protected and requires the master API key.
    """
    new_key = secrets.token_urlsafe(32)
    try:
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO api_keys (key, description) VALUES (?, ?)",
            (new_key, description)
        )
        db.commit()
        logger.info(f"Generated and stored new API key with description: {description}")
        return {"api_key": new_key, "description": description}
    except sqlite3.IntegrityError:
        # This is extremely unlikely but good to handle.
        logger.error("Generated an API key that already exists. This is highly improbable.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not generate a unique API key. Please try again."
        )
    except sqlite3.Error as e:
        logger.error(f"Database error while generating API key: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not store new API key due to a database error."
        )
    finally:
        db.close()