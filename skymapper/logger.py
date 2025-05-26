import sys

from loguru import logger

# Configure loguru
logger.remove()  # Remove default handler

# Add console handler with pretty formatting
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Add file handler
logger.add(
    "skymapper.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="00:00",
    retention="7 days",
    level="INFO",
)

# Export logger as module-level logger
__all__ = ["logger"]
