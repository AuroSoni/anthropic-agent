"""Database configuration and connection pool manager."""

import os
from typing import Any

import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseManager:
    """Manages PostgreSQL connection pool lifecycle.
    
    Usage:
        db = DatabaseManager()
        
        # In FastAPI lifespan:
        async with lifespan(app):
            yield
        
        # Check health:
        health = await db.health_check()
    """
    
    def __init__(
        self,
        connection_string: str | None = None,
        min_pool_size: int = 1,
        max_pool_size: int = 5,
    ):
        """Initialize database manager.
        
        Args:
            connection_string: PostgreSQL connection URL. 
                               Defaults to DATABASE_URL env var.
            min_pool_size: Minimum connections in pool.
            max_pool_size: Maximum connections in pool.
        """
        self._connection_string = connection_string or os.getenv("DATABASE_URL")
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: asyncpg.Pool | None = None
    
    @property
    def is_configured(self) -> bool:
        """Check if database connection is configured."""
        return self._connection_string is not None
    
    async def get_pool(self) -> asyncpg.Pool | None:
        """Get or create the connection pool (lazy initialization).
        
        Returns:
            Connection pool, or None if not configured.
        """
        if not self.is_configured:
            return None
        
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._connection_string,
                min_size=self._min_pool_size,
                max_size=self._max_pool_size,
                server_settings={'timezone': 'Asia/Kolkata'}
            )
        return self._pool
    
    async def close(self) -> None:
        """Close the connection pool.
        
        Should be called on application shutdown.
        """
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
    
    async def health_check(self) -> dict[str, Any]:
        """Check database connectivity.
        
        Returns:
            Health status dict with configured, status, and message fields.
        """
        if not self.is_configured:
            return {
                "configured": False,
                "status": "not_configured",
                "message": "DATABASE_URL environment variable not set",
            }
        
        try:
            pool = await self.get_pool()
            if pool:
                async with pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                return {
                    "configured": True,
                    "status": "up",
                    "message": "Connection successful",
                }
            else:
                return {
                    "configured": True,
                    "status": "down",
                    "message": "Failed to create connection pool",
                }
        except Exception as e:
            return {
                "configured": True,
                "status": "down",
                "message": str(e),
            }


# Global database manager instance
db = DatabaseManager()

