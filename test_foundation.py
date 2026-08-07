import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def main():
    print("--- 1. Loading Raw Data ---")
    df = pd.read_csv('data/RawDataMergedFinal.csv')

    input_features = [
        'Vehicle_Mass_kg', 'Velocity_kmh', 'Braking_Time_s', 
        'OuterDia [mm]', 'InnerDiameter [mm]', 'BrakeThickness [mm]', 
        'VentLength [mm]', 'Radiation Ambient Temp [C]', 'Convection Film Coeff [W/m^2.C]'
    ]
    
    target_features = [
        'Temperature Maximum [C]', 'Equivalent Stress Max [Pa]', 
        'Total Deformation Max [m]', 'Safety Factor Min -Dynamic'
    ]

    X = df[input_features]
    y = df[target_features]

    print("--- 2. Splitting and Scaling ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale Inputs (X)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # --- THE SECRET: Scale Targets (y) ---
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    print("--- 3. Training the Model ---")
    model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Train the model on the SCALED targets so it respects all 4 equally
    model.fit(X_train_scaled, y_train_scaled)
    print("Training Complete.\n")

    print("--- 4. Evaluation & Results ---")
    # Predict in the scaled domain
    y_pred_scaled = model.predict(X_test_scaled)
    
    # Inverse transform predictions back to real-world physics values (Celsius, Pascals, etc.)
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate R2 on the real-world values
    global_r2 = r2_score(y_test, y_pred_real)
    
    print("="*50)
    print(f"STANDALONE GLOBAL R2 SCORE: {global_r2:.4f}")
    print("="*50)

    y_pred_df = pd.DataFrame(y_pred_real, columns=target_features)
    y_test_df = y_test.reset_index(drop=True)

    print("\nIndividual Target Metrics:")
    for col in target_features:
        r2 = r2_score(y_test_df[col], y_pred_df[col])
        print(f"\n> {col}")
        print(f"  R2 Score : {r2:.4f}")

if __name__ == "__main__":
    main()