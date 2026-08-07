from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    # 'none' disables the physics engine — model trains on the 9 raw base
    # inputs only. This is the clean baseline (no feature leakage, no
    # physics-augmented inputs).
    print("1. Starting Baseline Data Ingestion (No Physics)...")
    ingestion = DataIngestion(physics_mode="none")
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()

    print("\n2. Starting Data Transformation...")
    transformer = DataTransformation()
    train_arr, test_arr, _, _ = transformer.initiate_data_transformation(train_data_path, test_data_path)

    print("\n3. Starting Baseline Model Training...")
    trainer = ModelTrainer()
    r2_score = trainer.initiate_model_trainer(train_arr, test_arr)

    print(f"\nSUCCESS! Baseline Model Trained. Global R2 Score: {r2_score:.4f}")