import logging
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from .config import API_KEY, API_KEY_NAME

logger = logging.getLogger(__name__)
api_key_header_auth_scheme = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_current_api_key(api_key_header: str = Security(api_key_header_auth_scheme)):
    if not API_KEY:
        # This case means the server itself is not configured with an API_KEY
        logger.error("Server-side API_KEY is not configured. Authentication cannot be performed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key authentication is not configured correctly on the server."
        )
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key"
        )