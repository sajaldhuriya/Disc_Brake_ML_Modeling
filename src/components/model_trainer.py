import os
import sys
from dataclasses import dataclass
import mlflow
import mlflow.sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV
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

    def initiate_model_trainer(self, train_array, test_array, tuning_mode=True):
        try:
            logging.info("Splitting training and test data arrays")
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

            # Distinct run names so each test shows up clearly in the MLflow UI.
            run_name = "Tuned_ExtraTrees" if tuning_mode else "Baseline_ExtraTrees"

            with mlflow.start_run(run_name=run_name):
                if tuning_mode:
                    logging.info("Starting Hyperparameter Tuning with RandomizedSearchCV...")
                    base_model = ExtraTreesRegressor(random_state=42, n_jobs=-1)

                    param_distributions = {
                        'n_estimators': [50, 100, 200, 300],
                        'max_depth': [None, 10, 20, 30, 40],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                    }

                    search = RandomizedSearchCV(
                        estimator=base_model,
                        param_distributions=param_distributions,
                        n_iter=10,
                        cv=3,
                        scoring='r2',
                        random_state=42,
                        n_jobs=-1,
                    )
                    search.fit(X_train, y_train)

                    model = search.best_estimator_
                    best_params = search.best_params_

                    logging.info(f"Best Hyperparameters Found: {best_params}")
                    mlflow.log_params(best_params)
                    mlflow.log_param("tuning_strategy", "RandomizedSearchCV")
                else:
                    logging.info("Training baseline model without tuning...")
                    model = ExtraTreesRegressor(
                        n_estimators=100, random_state=42, n_jobs=-1
                    )
                    model.fit(X_train, y_train)
                    mlflow.log_param("tuning_strategy", "None - Baseline")

                logging.info("Evaluating model on test set...")
                y_pred = model.predict(X_test)
                global_r2 = r2_score(y_test, y_pred)

                logging.info(f"Global R2 Score: {global_r2}")
                mlflow.log_metric("global_r2_score", global_r2)
                mlflow.log_param("model", "ExtraTreesRegressor")

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