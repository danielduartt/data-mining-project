import logging
from .data_processing import DataProcessor
from .visualization import Visualizer
from .analysis import Analyzer
from .modeling import Modeler

class DetailedStudentAnalyzer:
    def __init__(self, file_path: str):
        self.logger = logging.getLogger("StudentAnalyzer")
        self.logger.info(f"Initializing StudentAnalyzer with file: {file_path}")
        
        # Initialize modules
        self.data_processor = DataProcessor(file_path, self.logger)
        self.visualizer = Visualizer(self.logger)
        self.analyzer = Analyzer(self.data_processor, self.visualizer, self.logger)
        self.modeler = Modeler(self.data_processor, self.visualizer, self.logger)
        
        self.logger.info("Initialization completed successfully")