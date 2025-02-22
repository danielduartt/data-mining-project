import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging configuration with file and console output.
    
    Args:
        log_file: Optional custom log file path
        log_level: Logging level (default: logging.INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate default log file name if none provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"student_analysis_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    logger = logging.getLogger("StudentAnalyzer")
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger