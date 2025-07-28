import time
import logging
from fastapi import Request, HTTPException

from .database import get_db_connection

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    A simple rate limiter that uses SQLite to store request timestamps.
    """
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window

    def __call__(self, request: Request):
        """
        Checks if a client has exceeded the rate limit.
        This method is synchronous and should be run in a thread from an async context.
        """
        client_ip = request.client.host # type: ignore
        if not client_ip:
            # Cannot rate limit if we can't identify the client
            return

        current_time = time.time()
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 1. Delete old records for this client. This also acts as a cleanup
            # for the table over time.
            cutoff_time = current_time - self.time_window
            cursor.execute(
                "DELETE FROM request_logs WHERE client_identifier = ? AND timestamp < ?",
                (client_ip, cutoff_time)
            )

            # 2. Count recent requests for this client
            cursor.execute(
                "SELECT COUNT(*) FROM request_logs WHERE client_identifier = ?",
                (client_ip,)
            )
            request_count = cursor.fetchone()[0]

            # 3. Check if the client exceeds the max request limit
            if request_count >= self.max_requests:
                logger.warning(f"Rate limit exceeded for client: {client_ip}")
                raise HTTPException(
                    status_code=429, 
                    detail=f"Rate limit exceeded. Try again in {self.time_window} seconds."
                )

            # 4. Add the current request time to the log
            cursor.execute(
                "INSERT INTO request_logs (client_identifier, timestamp) VALUES (?, ?)",
                (client_ip, current_time)
            )
            
            conn.commit()

        except HTTPException:
            # Re-raise HTTPException to be caught by the middleware
            raise
        except Exception as e:
            # If the rate limiter fails for other reasons, it's probably better to let the request through
            # than to block legitimate users. Log the error for investigation.
            logger.error(f"Rate limiter failed for client {client_ip}: {e}", exc_info=True)
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()