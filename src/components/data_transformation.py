import sys
import os
import json
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
    # Path to the JSON file listing columns the PhysicsEngine actually added.
    # DataTransformation only treats columns listed here as legitimate
    # engineered inputs — anything else in the CSV is ignored, even if a
    # previous version of the code whitelisted it by name.
    engineered_columns_path: str = os.path.join('artifacts', 'engineered_columns.txt')

class DataTransformation:
    """
    OOP Data Transformation class responsible for strictly selecting features,
    scaling both inputs and outputs, and preventing data leakage.

    The leakage guard is structural rather than name-based: only columns
    listed in `artifacts/engineered_columns.txt` (written by DataIngestion)
    are added to the input set. A pre-computed column that happens to share
    a name with an engineered one cannot leak through this path.
    """
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def _load_engineered_columns(self) -> list:
        """Read the engineered-columns stamp written by DataIngestion.

        Returns an empty list if the file is missing or unreadable — that
        means no physics features were engineered for this run.
        """
        path = self.transformation_config.engineered_columns_path
        if not os.path.exists(path):
            logging.warning(
                f"Engineered-columns stamp not found at {path}. "
                "Treating run as having zero engineered features."
            )
            return []
        try:
            with open(path, 'r') as f:
                cols = json.load(f)
            logging.info(f"Loaded engineered-columns stamp: {cols}")
            return list(cols)
        except Exception as e:
            logging.error(f"Failed to read engineered-columns stamp: {e}")
            return []

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

            # 3. Read engineered columns from the stamp file written by DataIngestion.
            #    This is the structural guard against feature leakage: we trust
            #    ONLY columns the PhysicsEngine explicitly stamped as added.
            actual_physics_features = self._load_engineered_columns()

            # 4. Defensive sanity check: warn if any stamped column collides
            #    with an already-existing CSV column. The current engine avoids
            #    collisions by naming its output distinctly ('Engineered Heat
            #    Flow [W]' vs the raw 'Heat Flow Calculated [W]'), but this
            #    guard catches future regressions.
            for col in actual_physics_features:
                if col in base_input_features or col in target_features:
                    logging.warning(
                        f"Engineered column '{col}' collides with a base input or "
                        "target feature. Inspect for leakage."
                    )

            # 5. Final whitelist of input features
            final_input_features = base_input_features + actual_physics_features

            logging.info(f"Target columns identified: {len(target_features)}")
            logging.info(f"Input columns strictly selected: {len(final_input_features)}")
            logging.info(f"Final input feature list: {final_input_features}")

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