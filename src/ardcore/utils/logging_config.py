"""
Centralized logging configuration for all ARD workflows.

This module provides a single place to configure logging behavior across all workflows,
including suppression of noisy third-party library logs.
"""

import logging


def setup_workflow_logging(level=logging.INFO):
    """
    Set up logging configuration for ARD workflows.
    
    This function:
    1. Configures basic logging with the specified level
    2. Suppresses noisy logs from third-party libraries
    
    Args:
        level: Logging level (default: logging.INFO)
    """
    # Configure basic logging
    logging.basicConfig(level=level)
    
    # Suppress noisy third-party library logs
    suppress_noisy_loggers()


def suppress_noisy_loggers():
    """
    Suppress verbose logs from third-party libraries that generate too much noise.
    
    This includes:
    - httpx: HTTP request/response logs from OpenAI API calls
    - openai: OpenAI client internal logs  
    - urllib3: Low-level HTTP connection logs
    - boto3/botocore: AWS SDK logs
    """
    # HTTP client libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # OpenAI client
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    # AWS SDK libraries
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    
    # Other potentially noisy libraries
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
