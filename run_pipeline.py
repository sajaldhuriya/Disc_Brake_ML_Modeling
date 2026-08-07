from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Flip this to True to run the hyperparameter-tuned ExtraTrees test.
# Each run is logged separately to MLflow under 'Disc_Brake_Thermal_Analysis'.
TUNING_MODE = True

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

    if TUNING_MODE:
        print("\n3. Starting Hyperparameter-Tuned ExtraTrees Training...")
    else:
        print("\n3. Starting Baseline ExtraTrees Training...")

    trainer = ModelTrainer()
    r2_score = trainer.initiate_model_trainer(train_arr, test_arr, tuning_mode=TUNING_MODE)

    label = "Tuned Model" if TUNING_MODE else "Baseline Model"
    print(f"\nSUCCESS! {label} Trained. Global R2 Score: {r2_score:.4f}")