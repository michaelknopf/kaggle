import logging
from rich.logging import RichHandler

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S',  # ISO 8601 timestamp
        handlers=[RichHandler(rich_tracebacks=True)]
    )
