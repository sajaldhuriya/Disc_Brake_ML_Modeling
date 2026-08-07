import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration holding the paths where the input and target scalers will be saved.
    """
    scaler_X_file_path: str = os.path.join('artifacts', 'scaler_X.pkl')
    scaler_y_file_path: str = os.path.join('artifacts', 'scaler_y.pkl')

class DataTransformation:
    """
    OOP Data Transformation class responsible for strictly selecting features, 
    scaling both inputs and outputs, and preventing data leakage.
    """
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data for transformation.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # 1. Strictly define our TARGET features
            target_features = [
                'Temperature Maximum [C]', 
                'Equivalent Stress Max [Pa]', 
                'Total Deformation Max [m]', 
                'Safety Factor Min -Dynamic'
            ]

            # 2. Strictly define our BASE INPUT features (prevents data leakage from other columns)
            base_input_features = [
                'Vehicle_Mass_kg', 'Velocity_kmh', 'Braking_Time_s', 
                'OuterDia [mm]', 'InnerDiameter [mm]', 'BrakeThickness [mm]', 
                'VentLength [mm]', 'Radiation Ambient Temp [C]', 'Convection Film Coeff [W/m^2.C]'
            ]
            
            # 3. Define possible engineered physics features that might exist
            possible_physics_features = [
                'Heat Flow Calculated [W]',
                'Disc Face Area [m^2]',
                'Engineered Heat Flux [W/m^2]',
                'Braking Torque [N.m]'
            ]
            
            # 4. Filter to only include physics features actually present in the dataset
            actual_physics_features = [f for f in possible_physics_features if f in train_df.columns]
            
            # 5. Final whitelist of input features
            final_input_features = base_input_features + actual_physics_features

            logging.info(f"Target columns identified: {len(target_features)}")
            logging.info(f"Input columns strictly selected: {len(final_input_features)}")

            # Separate Features (X) and Targets (y) using the strict whitelist
            input_feature_train_df = train_df[final_input_features]
            target_feature_train_df = train_df[target_features]

            input_feature_test_df = test_df[final_input_features]
            target_feature_test_df = test_df[target_features]

            logging.info("Initializing StandardScalers for X and y.")
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            # --- SCALE INPUTS (X) ---
            logging.info("Applying scaling to input features (X).")
            input_feature_train_arr = scaler_X.fit_transform(input_feature_train_df)
            input_feature_test_arr = scaler_X.transform(input_feature_test_df)

            # --- SCALE TARGETS (y) ---
            logging.info("Applying scaling to target features (y).")
            target_feature_train_arr = scaler_y.fit_transform(target_feature_train_df)
            target_feature_test_arr = scaler_y.transform(target_feature_test_df)

            # Re-combine into final NumPy arrays: [X_features, y_targets]
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            # Save BOTH scalers
            logging.info("Saving scaler_X and scaler_y objects.")
            save_object(file_path=self.transformation_config.scaler_X_file_path, obj=scaler_X)
            save_object(file_path=self.transformation_config.scaler_y_file_path, obj=scaler_y)

            return (
                train_arr,
                test_arr,
                self.transformation_config.scaler_X_file_path,
                self.transformation_config.scaler_y_file_path,
            )

        except Exception as e:
            logging.error("Exception occurred during Data Transformation pipeline!")
            raise CustomException(e, sys)


if __name__ == "__main__":
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')

    try:
        transformer = DataTransformation()
        train_array, test_array, scaler_X_path, scaler_y_path = transformer.initiate_data_transformation(
            train_data_path, test_data_path
        )
        print("Transformation complete with Strict Whitelisting!")
        print(f"Train array shape: {train_array.shape}")
        print(f"Test array shape: {test_array.shape}")
    except Exception as e:
        print(f"Error during transformation: {e}")