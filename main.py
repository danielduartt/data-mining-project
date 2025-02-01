# main.py
from config.logging_config import setup_logging
from preprocessing.data_loader import DataLoader
from preprocessing.data_preprocessor import DataPreprocessor
from preprocessing.feature_engineering import FeatureEngineer
from analysis.feature_analyzer import FeatureAnalyzer
from analysis.performance_analyzer import PerformanceAnalyzer
from analysis.risk_analyzer import RiskAnalyzer
from analysis.correlation_analyzer import CorrelationAnalyzer
from analysis.semester_analyzer import SemesterAnalyzer
from export.report_generator import ReportGenerator
from export.data_exporter import DataExporter

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting main execution")

    try:
        file_path = "/home/tiago/college/mineracao_dados/data/data.csv"
        df = pd.read_csv(file_path, sep=";", encoding="utf-8")

        # Step 2: Inspect the dataset
        logger.info(f"Dataset shape: {df.shape}")
        logger.info("Column names:")
        logger.info(df.columns)
        logger.info("Missing values per column:")
        logger.info(df.isnull().sum())
        logger.info("Target variable distribution:")
        logger.info(df['Target'].value_counts())

        # Step 3: Handle invalid or missing values
        valid_targets = ['Dropout', 'Enrolled', 'Graduate']
        df = df[df['Target'].isin(valid_targets)]

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        # Step 4: Encode the Target variable
        target_map = {'Graduate': 1, 'Dropout': 0, 'Enrolled': 2}
        df['Target'] = df['Target'].map(target_map)

        # Step 5: Save the cleaned dataset
        cleaned_file_path = "cleaned_dataset.csv"
        df.to_csv(cleaned_file_path, index=False, sep=";", encoding="utf-8")

        logger.info(f"Cleaned dataset saved to: {cleaned_file_path}")
        # Initialize components
        file_path = "cleaned_dataset.csv"
        
        # Data loading and preprocessing
        df = DataLoader.load_data(file_path)
        preprocessor = DataPreprocessor()
        df = preprocessor.optimize_data_types(df)
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        X, y = feature_engineer.prepare_modeling_data(
            df, 
            preprocessor.numeric_features,
            preprocessor.categorical_features
        )
        
        # Analysis
        results = {}
        
        feature_analyzer = FeatureAnalyzer(df)
        results["feature_distributions"] = feature_analyzer.analyze_distributions()
        
        performance_analyzer = PerformanceAnalyzer(df)
        results["academic_performance"] = performance_analyzer.analyze_performance()
        
        risk_analyzer = RiskAnalyzer(df)
        results["risk_assessment"] = risk_analyzer.analyze_risk()
        
        correlation_analyzer = CorrelationAnalyzer(df)
        results["correlations"] = correlation_analyzer.analyze_correlations()
        
        semester_analyzer = SemesterAnalyzer(df)
        results["semester_patterns"] = semester_analyzer.analyze_patterns()
        
        # Export results
        report_generator = ReportGenerator(results)
        report_generator.generate_report()
        
        data_exporter = DataExporter(results)
        data_exporter.export_data()
        
        logger.info("Analysis completed successfully")
        return results

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.exception("Detailed error trace:")
        return {}

if __name__ == "__main__":
    results = main()