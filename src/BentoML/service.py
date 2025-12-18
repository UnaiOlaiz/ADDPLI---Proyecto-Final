import bentoml
import numpy as np
from pydantic import BaseModel, Field

XGB_TAG            = "ai4i2020_xgbclassifier:latest"
LOGR_TAG           = "ai4i2020_logistic_regression:latest"
SVM_TAG            = "ai4i2020_support_vector_machine:latest"
RF_TAG             = "ai4i2020_random_forest:latest"
HDBSCAN_MODEL_TAG  = "ai4i2020_hdbscan:latest"

XGB_SCALER_TAG       = "ai4i2020_scaler_xgbclassifier:latest"
LOGR_SCALER_TAG      = "ai4i2020_scaler_logistic_regression:latest"
SVM_SCALER_TAG       = "ai4i2020_scaler_svm:latest"
RF_SCALER_TAG        = "ai4i2020_scaler_random_forest:latest"
HDBSCAN_SCALER_TAG   = "ai4i2020_scaler_hdbscan:latest"


# Esquema para modelos de 7 columnas (Logistic, SVM, XGB)
class Input7Features2(BaseModel):
    # Definimos que esperamos una lista de listas de floats
    input_data: list[list[float]] = Field(
        default=[[298.9, 309.1, 2861, 4.6, 143, 1, 0]], # Datos de predicción de prueba estándar
        description="Matriz de entrada con 7 características."
    )

# Esquema para modelos de 7 columnas (Random Forest, HDBSCAN)
class Input7Features(BaseModel):
    input_data: list[list[float]] = Field(
        default=[[298.8, 308.9, 1455, 41.3, 208, 1, 0]],
        description="Matriz de entrada con 7 características."
    )

print("Cargando MODELOS del Model Store...")
xgb_model   = bentoml.sklearn.load_model(XGB_TAG)
logr_model  = bentoml.sklearn.load_model(LOGR_TAG)
svm_model   = bentoml.sklearn.load_model(SVM_TAG)
rf_model    = bentoml.sklearn.load_model(RF_TAG)
hdb_model   = bentoml.sklearn.load_model(HDBSCAN_MODEL_TAG)

print("Cargando SCALERS del Model Store...")
xgb_scaler   = bentoml.picklable_model.load_model(XGB_SCALER_TAG)
logr_scaler  = bentoml.picklable_model.load_model(LOGR_SCALER_TAG)
svm_scaler   = bentoml.picklable_model.load_model(SVM_SCALER_TAG)
rf_scaler    = bentoml.picklable_model.load_model(RF_SCALER_TAG)
hdb_scaler   = bentoml.picklable_model.load_model(HDBSCAN_SCALER_TAG)

print("Modelos y scalers cargados.")

@bentoml.service(name="AI4I2020__Failure__Prediction__Service")
class AI4I2020FailurePredictionService:
    """
    Servicio BentoML con Inputs definidos mediante Pydantic
    para mostrar ejemplos en la UI (Swagger).
    """

    # Convertir pydantic a numpy y validar
    def _prepare_data(self, pydantic_input, expected_cols: int) -> np.ndarray:
        # Extraemos la lista del objeto pydantic y convertimos a numpy
        data = np.array(pydantic_input.input_data)
        
        # Aseguramos 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        # Validamos columnas
        if data.shape[1] != expected_cols:
            raise ValueError(f"Se esperaban {expected_cols} columnas, recibidas {data.shape[1]}.")
            
        return data

    # Logistic Regression (Usa Input7Features2)
    @bentoml.api
    def predict_logreg(self, input_obj: Input7Features2) -> np.ndarray:
        data = self._prepare_data(input_obj, 7)
        scaled = logr_scaler.transform(data)
        return logr_model.predict_proba(scaled)

    # Random Forest (Usa Input7Features)
    @bentoml.api
    def predict_random_forest(self, input_obj: Input7Features) -> np.ndarray:
        data = self._prepare_data(input_obj, 7)
        scaled = rf_scaler.transform(data)
        return rf_model.predict(scaled)

    # SVM (Usa Input7Features2)
    @bentoml.api
    def predict_svm(self, input_obj: Input7Features2) -> np.ndarray:
        data = self._prepare_data(input_obj, 7)
        scaled = svm_scaler.transform(data)
        return svm_model.predict_proba(scaled)

    # XGBoost (Usa Input7Features2)
    @bentoml.api
    def predict_xgb(self, input_obj: Input7Features2) -> np.ndarray:
        data = self._prepare_data(input_obj, 7)
        scaled = xgb_scaler.transform(data)
        return xgb_model.predict_proba(scaled)

    # HDBSCAN (Usa Input7Features)
    @bentoml.api
    def cluster_hdbscan(self, input_obj: Input7Features) -> np.ndarray:
        data = self._prepare_data(input_obj, 7)
        scaled = hdb_scaler.transform(data)
        scaled = hdb_model.predict(scaled)
        return scaled
    
# Comando a ejecutar en terminal para servir el servicio BentoML
# bentoml serve service:AI4I2020FailurePredictionService --port 3000