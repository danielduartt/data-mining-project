from student_analyzer.analyzer import DetailedStudentAnalyzer
from student_analyzer.utils import setup_logging

def main():
    logger = setup_logging()
    logger.info("Starting main execution")
    
    try:
        file_path = "data/data (1).csv"
        analyzer = DetailedStudentAnalyzer(file_path)
        
        # Pré-processamento de dados
        analyzer.data_processor.preprocess_data()
        
        # Análises
        feature_distributions = analyzer.analyzer.analyze_feature_distributions()
        academic_performance = analyzer.analyzer.analyze_academic_performance()
        
        # Modelagem
        feature_importance = analyzer.modeler.perform_feature_selection()
        cluster_analysis = analyzer.modeler.analyze_student_clusters()
        
        logger.info("Analysis completed successfully")
        return {
            "feature_distributions": feature_distributions,
            "academic_performance": academic_performance,
            "feature_importance": feature_importance,
            "cluster_analysis": cluster_analysis,
        }
    
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        return {}

if __name__ == "__main__":
    main()