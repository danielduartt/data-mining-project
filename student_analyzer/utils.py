import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/student_analysis.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("StudentAnalyzer")