import os
import sys
from dataclasses import dataclass
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data arrays")
            # Extract features (all columns except last 4) and targets (last 4 columns)
            # Tree models don't need y-scaling, so we use raw unscaled targets directly.
            X_train, y_train, X_test, y_test = (
                train_array[:, :-4],
                train_array[:, -4:],
                test_array[:, :-4],
                test_array[:, -4:]
            )

            # Set MLflow tracking URI for local storage
            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            mlflow.set_experiment("Disc_Brake_Thermal_Analysis")

            with mlflow.start_run(run_name="Baseline_ExtraTrees_True_RealWorld"):
                logging.info("Initializing ExtraTreesRegressor for real-world targets")
                model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)

                logging.info("Training the model...")
                model.fit(X_train, y_train)

                logging.info("Evaluating model on test set in real-world units")
                y_pred = model.predict(X_test)
                
                # Calculate True Global R2 Score on real-world unscaled values
                global_r2 = r2_score(y_test, y_pred)
                logging.info(f"True Baseline Global R2 Score: {global_r2}")
                
                # Log metrics and parameters to MLflow
                mlflow.log_metric("global_r2_score", global_r2)
                mlflow.log_param("model", "ExtraTreesRegressor")
                mlflow.log_param("tuning", "None - Baseline")
                mlflow.log_param("physics_mode", "none")
                
                # Log model to MLflow
                mlflow.sklearn.log_model(model, "model")

                # Save model to artifacts
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=model
                )
                logging.info(f"Model saved to {self.model_trainer_config.trained_model_file_path}")

                return global_r2

        except Exception as e:
            raise CustomException(e, sys)