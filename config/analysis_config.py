from typing import Dict, Any

# Analysis parameters
ANALYSIS_PARAMS: Dict[str, float] = {
    "risk_threshold_high": 0.3,
    "risk_threshold_medium": 0.7,
    "min_samples_for_analysis": 100,
    "cross_validation_folds": 5,
}

# Target variable mapping
TARGET_NAMES: Dict[int, str] = {
    0: "Dropout",
    1: "Graduate",
    2: "Enrolled"
}

# Risk level definitions
RISK_LEVELS: Dict[int, str] = {
    0: "Low",
    1: "Medium",
    2: "High"
}

# Feature categories
DEMOGRAPHIC_FEATURES: list[str] = [
    "Gender",
    "Age at enrollment",
    "Scholarship holder"
]

# Analysis thresholds
CORRELATION_THRESHOLD: float = 0.5
SIGNIFICANCE_LEVEL: float = 0.05