import os
import sys
import warnings
import logging

# 1. Suppress TensorFlow C++ backend warnings and oneDNN messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. Suppress Abseil (absl) logging warnings
logging.getLogger('absl').setLevel(logging.ERROR)

# 3. Suppress Keras UserWarnings (like the input_shape deprecation)
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
warnings.filterwarnings('ignore', category=UserWarning)

from dataclasses import dataclass
import numpy as np

import mlflow
import mlflow.sklearn
import mlflow.tensorflow

# Sklearn & ML Models
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path_pkl = os.path.join("artifacts", "model.pkl")
    trained_model_file_path_keras = os.path.join("artifacts", "model.keras")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, model_name, tuning_mode, log_to_mlflow, physics_mode, param_grid=None):
        try:
            logging.info(f"Extracting arrays. Model: {model_name} | Tuning: {tuning_mode}")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-4], train_array[:, -4:],
                test_array[:, :-4], test_array[:, -4:]
            )
            input_dim = X_train.shape[1]
            output_dim = y_train.shape[1]

            model = None
            best_params = {}

            # ==========================================
            # BLOCK 1: TENSORFLOW / KERAS NEURAL NETWORK
            # ==========================================
            if model_name == "NeuralNetwork":
                import tensorflow as tf
                from tensorflow import keras

                def build_keras_model(params):
                    # 1. Build layers dynamically so we can conditionally add Dropout & BatchNorm
                    layers = [keras.layers.Input(shape=(input_dim,))]
                    
                    # Hidden Layer 1
                    layers.append(keras.layers.Dense(params.get('neurons_layer_1', 64)))
                    if params.get('use_batch_norm', False):
                        layers.append(keras.layers.BatchNormalization())
                    layers.append(keras.layers.Activation('relu'))
                    if params.get('dropout_rate', 0.0) > 0:
                        layers.append(keras.layers.Dropout(params.get('dropout_rate', 0.0)))

                    # Hidden Layer 2
                    layers.append(keras.layers.Dense(params.get('neurons_layer_2', 32)))
                    if params.get('use_batch_norm', False):
                        layers.append(keras.layers.BatchNormalization())
                    layers.append(keras.layers.Activation('relu'))
                    if params.get('dropout_rate', 0.0) > 0:
                        layers.append(keras.layers.Dropout(params.get('dropout_rate', 0.0)))

                    # Output Layer
                    layers.append(keras.layers.Dense(output_dim, activation='linear'))
                    
                    nn = keras.Sequential(layers)
                    
                    # 2. Dynamic Optimizer Selection
                    opt_name = params.get('optimizer', 'adam').lower()
                    lr = params.get('learning_rate', 0.001)
                    
                    if opt_name == 'sgd':
                        optimizer = keras.optimizers.SGD(learning_rate=lr)
                    elif opt_name == 'rmsprop':
                        optimizer = keras.optimizers.RMSprop(learning_rate=lr)
                    else:
                        optimizer = keras.optimizers.Adam(learning_rate=lr)
                        
                    # 3. Dynamic Loss Selection
                    loss_fn = params.get('loss', 'mse')
                    
                    nn.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'])
                    return nn

                # --- HYPERPARAMETER TUNING LOOP ---
                if tuning_mode and param_grid:
                    logging.info("Starting Custom Grid Search for Neural Network...")
                    # Generate 5 random hyperparameter combinations
                    param_list = list(ParameterSampler(param_grid, n_iter=5, random_state=42))
                    best_r2 = -float('inf')
                    best_history = None

                    for p in param_list:
                        temp_model = build_keras_model(p)
                        
                        # Fit the model and save its history
                        history = temp_model.fit(
                            X_train, y_train, 
                            epochs=p.get('epochs', 50), 
                            batch_size=p.get('batch_size', 32), 
                            verbose=0
                        )
                        
                        y_pred_temp = temp_model.predict(X_test, verbose=0)
                        r2_temp = r2_score(y_test, y_pred_temp)
                        
                        if r2_temp > best_r2:
                            best_r2 = r2_temp
                            model = temp_model
                            best_params = p
                            best_history = history
                            
                    logging.info(f"Best NN parameters found: {best_params}")
                    
                else:
                    logging.info("Training baseline Neural Network without tuning...")
                    best_params = param_grid if param_grid else {
                        'epochs': 50, 'batch_size': 32, 'learning_rate': 0.001, 
                        'neurons_layer_1': 64, 'neurons_layer_2': 32,
                        'optimizer': 'adam', 'loss': 'mse'
                    }
                    model = build_keras_model(best_params)
                    history = model.fit(
                        X_train, y_train, 
                        epochs=best_params.get('epochs', 50), 
                        batch_size=best_params.get('batch_size', 32), 
                        verbose=0
                    )

                # Save Neural Network
                model.save(self.model_trainer_config.trained_model_file_path_keras)
                logging.info(f"Keras Model saved to {self.model_trainer_config.trained_model_file_path_keras}")

            # ==========================================
            # BLOCK 2: STANDARD MACHINE LEARNING MODELS
            # ==========================================
            else:
                models = {
                    "ExtraTrees": ExtraTreesRegressor(random_state=42, n_jobs=-1),
                    "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
                    "DecisionTree": DecisionTreeRegressor(random_state=42),
                    "LinearRegression": LinearRegression(),
                    "Ridge": Ridge(),
                    "Lasso": MultiOutputRegressor(Lasso()),
                    "KNeighbors": KNeighborsRegressor(n_jobs=-1),
                    "GradientBoosting": MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
                    "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),
                    "SVR": MultiOutputRegressor(SVR())
                }

                if model_name not in models:
                    raise Exception(f"Model '{model_name}' is not supported.")

                base_model = models[model_name]

                if tuning_mode and param_grid:
                    logging.info(f"Starting Hyperparameter Tuning for {model_name}...")
                    search = RandomizedSearchCV(
                        estimator=base_model, param_distributions=param_grid,
                        n_iter=5, cv=3, scoring='r2', random_state=42, n_jobs=-1
                    )
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    best_params = search.best_params_
                    logging.info(f"Best parameters found: {best_params}")
                else:
                    logging.info(f"Training baseline {model_name}...")
                    model = base_model
                    model.fit(X_train, y_train)
                    best_params = {"tuning": "None"}

                # Save Standard ML Model
                save_object(file_path=self.model_trainer_config.trained_model_file_path_pkl, obj=model)
                logging.info(f"ML Model saved to {self.model_trainer_config.trained_model_file_path_pkl}")

            # ==========================================
            # BLOCK 3: EVALUATION AND MLFLOW TRACKING
            # ==========================================
            logging.info("Evaluating model on test set...")
            # Predict handles both sklearn and tf/keras outputs seamlessly
            y_pred = model.predict(X_test)
            global_r2 = r2_score(y_test, y_pred)
            logging.info(f"Global R2 Score for {model_name}: {global_r2}")

            if log_to_mlflow:
                logging.info("Logging results to MLflow...")
                mlflow.set_tracking_uri("sqlite:///mlruns.db")
                mlflow.set_experiment("Disc_Brake_Thermal_Analysis")
                
                run_name = f"{model_name}_{'Tuned' if tuning_mode else 'Base'}_{physics_mode}"
                
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_param("model", model_name)
                    mlflow.log_param("physics_mode", physics_mode)
                    mlflow.log_params(best_params)
                    mlflow.log_metric("global_r2_score", global_r2)
                    
                    # Log appropriately based on framework
                    if model_name == "NeuralNetwork":
                        mlflow.tensorflow.log_model(model, "model")
                    else:
                        mlflow.sklearn.log_model(model, "model")
            else:
                logging.info("MLflow logging is DISABLED in params.yaml. Skipping tracking.")

            return global_r2

        except Exception as e:
            raise CustomException(e, sys)