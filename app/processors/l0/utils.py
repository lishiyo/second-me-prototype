import logging
import time
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast
import traceback

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for generic function typing
T = TypeVar('T')


def retry(
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    error_types: tuple = (Exception,),
    should_retry: Optional[Callable[[Exception], bool]] = None
) -> Callable:
    """
    Decorator for retrying a function when it raises specified exceptions.
    
    Args:
        max_retries: Maximum number of retries
        initial_backoff: Initial backoff time in seconds
        backoff_factor: Factor to increase backoff time by after each retry
        error_types: Tuple of exception types to retry on
        should_retry: Function that takes an exception and returns whether to retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            retries = 0
            backoff = initial_backoff
            
            while True:
                try:
                    return func(*args, **kwargs)
                except error_types as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    # Check if we should retry this exception
                    if should_retry and not should_retry(e):
                        logger.info(f"Not retrying due to should_retry condition: {str(e)}")
                        raise
                    
                    logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} after error: {str(e)}. "
                        f"Waiting {backoff} seconds..."
                    )
                    time.sleep(backoff)
                    backoff *= backoff_factor
        
        return wrapper
    
    return decorator


def safe_execute(
    func: Callable[..., T], 
    *args: Any, 
    default_value: Optional[Any] = None,
    log_exceptions: bool = True,
    **kwargs: Any
) -> T:
    """
    Execute a function and catch any exceptions. Return default_value if an exception occurs.
    
    Args:
        func: Function to execute
        *args: Positional arguments for func
        default_value: Value to return if an exception occurs
        log_exceptions: Whether to log exceptions
        **kwargs: Keyword arguments for func
        
    Returns:
        Result of func or default_value if an exception occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_exceptions:
            logger.error(f"Error executing {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
        return default_value


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with a specific name and level.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Only add handler if not already present
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Disable propagation to avoid duplicate logs
        logger.propagate = False
    
    return logger


def encode_url_safe(text: str) -> str:
    """
    Replace special characters in URLs to prevent chunking issues.
    
    Args:
        text: Text to encode
        
    Returns:
        Encoded text
    """
    import re
    
    def replace_url(match: re.Match) -> str:
        url = match.group(0)
        # Replace common URL special characters with placeholders
        placeholders = {
            "://": "__COLON_SLASH_SLASH__",
            ".": "__DOT__",
            "/": "__SLASH__",
            "?": "__QUESTION__",
            "&": "__AMPERSAND__",
            "=": "__EQUALS__",
            "#": "__HASH__",
            "%": "__PERCENT__",
            "+": "__PLUS__",
        }
        
        for char, placeholder in placeholders.items():
            url = url.replace(char, placeholder)
        
        return url
    
    # Find URLs in text and replace them
    url_pattern = r'https?://[^\s]+'
    return re.sub(url_pattern, replace_url, text)


def decode_url_safe(text: str) -> str:
    """
    Restore special characters in URLs.
    
    Args:
        text: Text to decode
        
    Returns:
        Decoded text
    """
    placeholders = {
        "__COLON_SLASH_SLASH__": "://",
        "__DOT__": ".",
        "__SLASH__": "/",
        "__QUESTION__": "?",
        "__AMPERSAND__": "&",
        "__EQUALS__": "=",
        "__HASH__": "#",
        "__PERCENT__": "%",
        "__PLUS__": "+",
    }
    
    result = text
    for placeholder, char in placeholders.items():
        result = result.replace(placeholder, char)
    
    return result
