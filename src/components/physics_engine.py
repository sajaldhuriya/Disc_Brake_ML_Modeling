import numpy as np
import pandas as pd
import sys
from src.exception import CustomException
from src.logger import logging

class PhysicsEngine:
    """
    OOP Class responsible for engineering domain-specific physics features.
    Built with modularity to support different experimental baselines.

    The class also stamps every column it adds in `self.engineered_columns`
    so downstream code can distinguish freshly-engineered features from any
    pre-existing columns in the raw CSV. This is the structural guard against
    silent feature leakage (e.g. a pre-computed simulation output accidentally
    being treated as a model input).
    """
    # Columns added by each private physics routine. Kept as class constants
    # so both the methods and `transform_physics_features` share one source of truth.
    THERMAL_COLUMNS = [
        'Engineered Heat Flow [W]',
        'Disc Face Area [m^2]',
        'Engineered Heat Flux [W/m^2]',
    ]
    MECHANICAL_COLUMNS = [
        'Braking Torque [N.m]',
    ]

    def __init__(self, mode="full"):
        self.mode = mode
        # Stamp tracking — populated by transform_physics_features.
        self.engineered_columns: list[str] = []
        logging.info(f"Initialized PhysicsEngine with mode: {self.mode}")

    def get_engineered_columns(self) -> list[str]:
        """Accessor for the list of columns this engine instance added."""
        return list(self.engineered_columns)

    def _apply_thermal_equations(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Applies only Heat Flux and Area equations."""
        v_ms = X_df['Velocity_kmh'] * (1000.0 / 3600.0)
        ek = 0.5 * X_df['Vehicle_Mass_kg'] * (v_ms ** 2)
        p_total = ek / X_df['Braking_Time_s']
        # NOTE: column name is intentionally distinct from the raw CSV's
        # 'Heat Flow Calculated [W]' so we don't silently overwrite a
        # pre-computed simulation output.
        X_df['Engineered Heat Flow [W]'] = p_total * 0.30

        outer_m = X_df['OuterDia [mm]'] / 1000.0
        inner_m = X_df['InnerDiameter [mm]'] / 1000.0
        area_m2 = (np.pi / 4.0) * (outer_m**2 - inner_m**2)

        X_df['Disc Face Area [m^2]'] = area_m2
        X_df['Engineered Heat Flux [W/m^2]'] = X_df['Engineered Heat Flow [W]'] / area_m2
        return X_df

    def _apply_mechanical_equations(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Applies only Braking Torque equations."""
        v_ms = X_df['Velocity_kmh'] * (1000.0 / 3600.0)
        ek = 0.5 * X_df['Vehicle_Mass_kg'] * (v_ms ** 2)
        p_total = ek / X_df['Braking_Time_s']

        outer_m = X_df['OuterDia [mm]'] / 1000.0
        inner_m = X_df['InnerDiameter [mm]'] / 1000.0

        f_friction = p_total / (v_ms + 1e-6)
        r_eff = (outer_m + inner_m) / 4.0
        X_df['Braking Torque [N.m]'] = f_friction * r_eff
        return X_df

    def transform_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Routes the dataframe through the requested physics equations.

        Also populates `self.engineered_columns` so downstream consumers can
        tell which columns were added by this engine vs. which already
        existed in the source dataframe.
        """
        try:
            X_df = df.copy()
            # Reset the stamp on every transform so the list always reflects
            # the most recent invocation.
            self.engineered_columns = []

            if self.mode == "none":
                logging.info("Physics mode is 'none'. Skipping domain engineering.")
                return X_df

            if self.mode in ["full", "thermal_only", "heat_flow_only"]:
                logging.info("Applying thermal physics features...")
                X_df = self._apply_thermal_equations(X_df)
                self.engineered_columns.extend(self.THERMAL_COLUMNS)

            if self.mode in ["full", "torque_only"]:
                logging.info("Applying mechanical torque features...")
                X_df = self._apply_mechanical_equations(X_df)
                self.engineered_columns.extend(self.MECHANICAL_COLUMNS)

            logging.info(f"PhysicsEngine stamped engineered columns: {self.engineered_columns}")
            return X_df

        except Exception as e:
            logging.error("Exception occurred during Physics Routing!")
            raise CustomException(e, sys)