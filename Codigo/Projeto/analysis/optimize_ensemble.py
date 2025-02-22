import optuna
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import logging
from typing import Dict

logger = logging.getLogger("StudentAnalyzer")

class EnhancedEnsembleOptimizer:
    def __init__(
        self,
        n_jobs: int = -1,
        random_state: int = 42,
        n_trials: int = 100
    ):
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_trials = n_trials
        
        # Initialize best models
        self.best_models = {}
    
    def optimize_hyperparameters(self, X, y):
        """Optimize hyperparameters for ensemble models."""
        def get_best_parameters(study):
            """Extract best parameters from study."""
            best_trial = study.best_trial
            param_keys = {
                'random_forest': [k for k in best_trial.params.keys() if k.startswith('rf_')],
                'gradient_boosting': [k for k in best_trial.params.keys() if k.startswith('gb_')],
                'adaboost': [k for k in best_trial.params.keys() if k.startswith('ab_')]
            }
            
            best_params = {
                'random_forest': {
                    k[3:]: best_trial.params[k] for k in param_keys['random_forest']
                },
                'gradient_boosting': {
                    k[3:]: best_trial.params[k] for k in param_keys['gradient_boosting']
                },
                'adaboost': {
                    k[3:]: best_trial.params[k] for k in param_keys['adaboost']
                }
            }
            
            # Add fixed parameters
            best_params['random_forest'].update({
                'criterion': 'entropy',
                'class_weight': 'balanced_subsample',
                'n_jobs': self.n_jobs,
                'random_state': self.random_state,
                'bootstrap': True,
                'warm_start': True
            })
            
            best_params['gradient_boosting'].update({
                'validation_fraction': 0.2,
                'n_iter_no_change': 20,
                'random_state': self.random_state,
                'warm_start': True
            })
            
            best_params['adaboost'].update({
                'random_state': self.random_state
            })
            
            return best_params

        def objective(trial):
            try:
                # Random Forest parameters
                rf_params = {
                    'n_estimators': trial.suggest_int('rf_n_estimators', 300, 1000),
                    'max_depth': trial.suggest_int('rf_max_depth', 15, 40),
                    'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 2, 8),
                    'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2']),
                    'max_samples': trial.suggest_float('rf_max_samples', 0.8, 0.9),
                    'criterion': 'entropy',
                    'class_weight': 'balanced_subsample',
                    'n_jobs': self.n_jobs,
                    'random_state': self.random_state,
                    'bootstrap': True,
                    'warm_start': True
                }
                
                # Gradient Boosting parameters
                gb_params = {
                    'n_estimators': trial.suggest_int('gb_n_estimators', 300, 1000),
                    'learning_rate': trial.suggest_float('gb_learning_rate', 0.01, 0.1, log=True),
                    'max_depth': trial.suggest_int('gb_max_depth', 6, 12),
                    'min_samples_split': trial.suggest_int('gb_min_samples_split', 5, 15),
                    'min_samples_leaf': trial.suggest_int('gb_min_samples_leaf', 3, 8),
                    'subsample': trial.suggest_float('gb_subsample', 0.8, 0.9),
                    'max_features': trial.suggest_float('gb_max_features', 0.8, 0.9),
                    'validation_fraction': 0.2,
                    'n_iter_no_change': 20,
                    'random_state': self.random_state,
                    'warm_start': True
                }
                
                # AdaBoost base estimator parameters
                base_estimator_params = {
                    'random_state': self.random_state
                }
                
                # Create base estimator for AdaBoost
                base_estimator = DecisionTreeClassifier(**base_estimator_params)
                
                # AdaBoost parameters
                ab_params = {
                    'n_estimators': trial.suggest_int('ab_n_estimators', 200, 500),
                    'learning_rate': trial.suggest_float('ab_learning_rate', 0.01, 0.1, log=True),
                    'base_estimator': base_estimator,
                    'random_state': self.random_state
                }
                
                # Create models
                rf = RandomForestClassifier(**rf_params)
                gb = GradientBoostingClassifier(**gb_params)
                ab = AdaBoostClassifier(**ab_params)
                
                # Cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                scores = {
                    'accuracy': [], 'precision': [], 
                    'recall': [], 'f1': [], 'roc_auc': []
                }
                
                for train_idx, val_idx in cv.split(X, y):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train and evaluate each model
                    y_preds = []
                    y_probs = []
                    
                    for model in [rf, gb, ab]:
                        model.fit(X_train, y_train)
                        y_preds.append(model.predict(X_val))
                        y_probs.append(model.predict_proba(X_val)[:, 1])
                    
                    # Average predictions (simple ensemble)
                    y_pred = np.round(np.mean(y_preds, axis=0))
                    y_prob = np.mean(y_probs, axis=0)
                    
                    # Calculate metrics
                    scores['accuracy'].append(accuracy_score(y_val, y_pred))
                    scores['precision'].append(precision_score(y_val, y_pred))
                    scores['recall'].append(recall_score(y_val, y_pred))
                    scores['f1'].append(f1_score(y_val, y_pred))
                    scores['roc_auc'].append(roc_auc_score(y_val, y_prob))
                
                # Calculate mean scores
                mean_scores = {k: np.mean(v) for k, v in scores.items()}
                
                # Log current trial performance
                logger.info(f"Trial {trial.number} scores:")
                for metric, value in mean_scores.items():
                    logger.info(f"{metric}: {value:.4f}")
                
                # Return negative mean F1 score for minimization
                return -mean_scores['f1']
                
            except Exception as e:
                logger.error(f"Error in trial: {str(e)}")
                return 0.0
        
        # Create and run study
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            callbacks=[self._log_optimization_progress],
            show_progress_bar=True
        )
        
        # Get best parameters
        try:
            best_params = get_best_parameters(study)
            logger.info(f"Best value: {-study.best_value:.4f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error extracting best parameters: {str(e)}")
            raise

    def _create_base_models(self, trial):
        """Create base models with optimized parameters."""
        # Random Forest parameters
        rf_params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 300, 1000),
            'max_depth': trial.suggest_int('rf_max_depth', 15, 40),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 2, 8),
            'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2']),
            'max_samples': trial.suggest_float('rf_max_samples', 0.8, 0.9),
            'criterion': 'entropy',
            'class_weight': 'balanced_subsample',
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'bootstrap': True,
            'warm_start': True
        }
        
        # Gradient Boosting parameters
        gb_params = {
            'n_estimators': trial.suggest_int('gb_n_estimators', 300, 1000),
            'learning_rate': trial.suggest_float('gb_learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('gb_max_depth', 6, 12),
            'min_samples_split': trial.suggest_int('gb_min_samples_split', 5, 15),
            'min_samples_leaf': trial.suggest_int('gb_min_samples_leaf', 3, 8),
            'subsample': trial.suggest_float('gb_subsample', 0.8, 0.9),
            'max_features': trial.suggest_float('gb_max_features', 0.8, 0.9),
            'validation_fraction': 0.2,
            'n_iter_no_change': 20,
            'random_state': self.random_state,
            'warm_start': True
        }
        
        # AdaBoost base estimator parameters
        base_estimator_params = {
            'random_state': self.random_state
        }
        
        # Create base estimator for AdaBoost
        base_estimator = DecisionTreeClassifier(**base_estimator_params)
        
        # AdaBoost parameters
        ab_params = {
            'n_estimators': trial.suggest_int('ab_n_estimators', 200, 500),
            'learning_rate': trial.suggest_float('ab_learning_rate', 0.01, 0.1, log=True),
            'base_estimator': base_estimator,
            'random_state': self.random_state
        }
        
        # Create models dictionary
        models = {
            'rf': RandomForestClassifier(**rf_params),
            'gb': GradientBoostingClassifier(**gb_params),
            'ab': AdaBoostClassifier(**ab_params)
        }
        
        # Create params dictionary
        params = {
            'random_forest': rf_params,
            'gradient_boosting': gb_params,
            'adaboost': ab_params
        }
        
        return models, params
    
    def _log_optimization_progress(self, study, trial):
        """Log optimization progress."""
        if trial.number % 5 == 0:
            logger.info(
                f"Trial {trial.number} completed. "
                f"Best score so far: {-study.best_value:.4f}"
            )