import os
import sys
import json
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.physics_engine import PhysicsEngine


@dataclass
class DataIngestionConfig:
    """
    Configuration dataclass holding paths for output data artifacts.
    """
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    # Stamp file: lists every column the PhysicsEngine added during ingestion.
    # DataTransformation reads this so it can distinguish freshly-engineered
    # features from any pre-existing columns in the raw CSV.
    engineered_columns_path: str = os.path.join('artifacts', 'engineered_columns.txt')


class DataIngestion:
    """
    OOP Data Ingestion class that loads raw data, applies PhysicsEngine
    transformations, and saves split train/test artifacts.

    Also persists the list of columns the PhysicsEngine added to
    `engineered_columns_path` as JSON, so downstream consumers can trust
    those columns as legitimate engineered features rather than scanning
    the dataframe for column names (which is how feature leakage happens).
    """
    def __init__(self, physics_mode="full"):
        self.ingestion_config = DataIngestionConfig()
        # Composition: Pass the mode dynamically into the physics engine
        self.physics_engine = PhysicsEngine(mode=physics_mode)

    def initiate_data_ingestion(self):
        logging.info(f"Entered Data Ingestion with physics_mode='{self.physics_engine.mode}'")
        try:
            # 1. Read raw CSV dataset
            raw_csv_path = os.path.join('data', 'RawDataMergedFinal.csv')
            df = pd.read_csv(raw_csv_path)
            logging.info(f"Read dataset successfully with shape: {df.shape}")

            # 2. Create artifacts directory if missing
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # 3. Apply physics equations to engineer new domain features
            df_engineered = self.physics_engine.transform_physics_features(df)
            engineered_columns = self.physics_engine.get_engineered_columns()
            logging.info(f"PhysicsEngine stamped columns: {engineered_columns}")

            # 4. Save entire engineered dataset
            df_engineered.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # 5. Persist the engineered-columns stamp so DataTransformation
            #    can read it back. JSON makes it trivial to extend later.
            with open(self.ingestion_config.engineered_columns_path, 'w') as f:
                json.dump(engineered_columns, f)

            # 6. Perform Train/Test Split (80% Train, 20% Test)
            logging.info("Initiating Train-Test split...")
            train_set, test_set = train_test_split(df_engineered, test_size=0.2, random_state=42)

            # 7. Save Train and Test artifacts
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion and Physics Integration completed successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Exception occurred during Data Ingestion pipeline!")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test execution with NO physics (for Baseline ML testing)
    print("Testing Ingestion: NO PHYSICS")
    obj_base = DataIngestion(physics_mode="none")
    train_data, test_data = obj_base.initiate_data_ingestion()
    print(f"Data Ingestion (No Physics) completed.\nTrain path: {train_data}\nTest path: {test_data}\n")
    
    # Test execution with FULL physics (for PIML testing)
    print("Testing Ingestion: FULL PHYSICS")
    obj_piml = DataIngestion(physics_mode="full")
    train_data_piml, test_data_piml = obj_piml.initiate_data_ingestion()
    print(f"Data Ingestion (Full Physics) completed.\nTrain path: {train_data_piml}\nTest path: {test_data_piml}")