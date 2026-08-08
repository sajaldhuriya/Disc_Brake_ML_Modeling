from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import read_yaml

if __name__ == "__main__":
    print("Loading params.yaml...")
    params = read_yaml("params.yaml")
    
    log_mlflow = params['pipeline_control']['log_to_mlflow']
    physics_mode = params['data_transformation']['physics_mode']
    
    model_name = params['model_training']['model_name']
    tuning_mode = params['model_training']['hyperparameter_tuning']
    
    # Extract the specific hyperparameter grid for the selected model
    param_grid = None
    if tuning_mode:
        param_grid = params['model_training']['hyperparameters'].get(model_name, {})

    print(f"\n--- PIPELINE CONFIGURATION ---")
    print(f"Physics Mode : {physics_mode}")
    print(f"Model        : {model_name}")
    print(f"Tuning       : {tuning_mode}")
    print(f"Log MLflow   : {log_mlflow}")
    print(f"------------------------------\n")

    print("1. Starting Data Ingestion...")
    ingestion = DataIngestion(physics_mode=physics_mode)
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()

    print("2. Starting Data Transformation...")
    transformation = DataTransformation()
    train_arr, test_arr, scaler_X_path, scaler_y_path = transformation.initiate_data_transformation(
        train_data_path, test_data_path
    )

    print("3. Starting Model Training...")
    trainer = ModelTrainer()
    r2 = trainer.initiate_model_trainer(
        train_arr, test_arr, 
        model_name=model_name, 
        tuning_mode=tuning_mode, 
        log_to_mlflow=log_mlflow,
        physics_mode=physics_mode,
        param_grid=param_grid
    )
    
    print(f"\nSUCCESS! Pipeline Finished. Global R2 Score: {r2}")