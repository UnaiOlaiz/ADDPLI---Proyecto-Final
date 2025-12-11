import bentoml
import numpy as np
from bentoml.io import NumpyNdarray
from bentoml import Runnable

# Aquí va el modelo entrenado que queremos servir
XGB_TAG   = "ai4i2020_xgbclassifier:latest"
LOGR_TAG  = "ai4i2020_logistic_regression:latest"
SVM_TAG   = "ai4i2020_support_vector_machine:latest"
RF_TAG    = "ai4i2020_random_forest:latest"
HDBSCAN_TAG = "ai4i2020_hdbscan:latest"

# Cargar el modelo sklearn como un runner de BentoML
xgb_runner  = bentoml.sklearn.get(XGB_TAG).to_runner()
logr_runner = bentoml.sklearn.get(LOGR_TAG).to_runner()
svm_runner  = bentoml.sklearn.get(SVM_TAG).to_runner()
rf_runner   = bentoml.sklearn.get(RF_TAG).to_runner()
hdbscan_runner = bentoml.sklearn.get(HDBSCAN_TAG).to_runner()

# Tag del scaler dentro del Model Store
XGB_SCALER_TAG  = "ai4i2020_scaler_xgbclassifier:latest"
LOGR_SCALER_TAG = "ai4i2020_scaler_logistic_regression:latest"
SVM_SCALER_TAG  = "ai4i2020_scaler_svm:latest"
RF_SCALER_TAG   = "ai4i2020_scaler_random_forest:latest"
HDBSCAN_TAG     = "ai4i2020_scaler_hdbscan:latest"

class XGBScalerRunnable(Runnable):
    # obligatorio para BentoML runner/strategy
    SUPPORTED_RESOURCES = ("cpu",)
    # indicar capacidades de concurrencia que BentoML consulta
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        print(f"Cargando el scaler desde el Model Store {XGB_SCALER_TAG}...")
        self.scaler = bentoml.picklable_model.load_model(XGB_SCALER_TAG)
        print("¡Scaler cargado!")

    # @bentoml.runnable.method define una función que el runner puede llamar
    @Runnable.method(batchable=True, batch_dim=0)
    def transform(self, input_data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(input_data)
    
class LogrScalerRunnable(Runnable):
    # obligatorio para BentoML runner/strategy
    SUPPORTED_RESOURCES = ("cpu",)
    # indicar capacidades de concurrencia que BentoML consulta
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        print(f"Cargando el scaler desde el Model Store {LOGR_SCALER_TAG}...")
        self.scaler = bentoml.picklable_model.load_model(LOGR_SCALER_TAG)
        print("¡Scaler cargado!")

    # @bentoml.runnable.method define una función que el runner puede llamar
    @Runnable.method(batchable=True, batch_dim=0)
    def transform(self, input_data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(input_data)
    
class SVMScalerRunnable(Runnable):
    # obligatorio para BentoML runner/strategy
    SUPPORTED_RESOURCES = ("cpu",)
    # indicar capacidades de concurrencia que BentoML consulta
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        print(f"Cargando el scaler desde el Model Store {SVM_SCALER_TAG}...")
        self.scaler = bentoml.picklable_model.load_model(SVM_SCALER_TAG)
        print("¡Scaler cargado!")

    # @bentoml.runnable.method define una función que el runner puede llamar
    @Runnable.method(batchable=True, batch_dim=0)
    def transform(self, input_data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(input_data)
    
class RFScalerRunnable(Runnable):
    # obligatorio para BentoML runner/strategy
    SUPPORTED_RESOURCES = ("cpu",)
    # indicar capacidades de concurrencia que BentoML consulta
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        print(f"Cargando el scaler desde el Model Store {RF_SCALER_TAG}...")
        self.scaler = bentoml.picklable_model.load_model(RF_SCALER_TAG)
        print("¡Scaler cargado!")

    # @bentoml.runnable.method define una función que el runner puede llamar
    @Runnable.method(batchable=True, batch_dim=0)
    def transform(self, input_data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(input_data)
    
class HDBSCANScalerRunnable(Runnable):
    # obligatorio para BentoML runner/strategy
    SUPPORTED_RESOURCES = ("cpu",)
    # indicar capacidades de concurrencia que BentoML consulta
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        print(f"Cargando el scaler desde el Model Store {HDBSCAN_TAG}...")
        self.scaler = bentoml.picklable_model.load_model(HDBSCAN_TAG)
        print("¡Scaler cargado!")

    # @bentoml.runnable.method define una función que el runner puede llamar
    @Runnable.method(batchable=True, batch_dim=0)
    def transform(self, input_data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(input_data)
    
# 3. Crear el scaler_runner a partir de nuestra CLASE personalizada
# 3. Crear el scaler_runner a partir de nuestra CLASE personalizada
xgb_scaler_runner  = bentoml.Runner(XGBScalerRunnable, name="xgb_scaler_runner")
logr_scaler_runner = bentoml.Runner(LogrScalerRunnable, name="logr_scaler_runner")
svm_scaler_runner  = bentoml.Runner(SVMScalerRunnable, name="svm_scaler_runner")
rf_scaler_runner   = bentoml.Runner(RFScalerRunnable, name="rf_scaler_runner")
hdbscan_scaler_runner = bentoml.Runner(HDBSCANScalerRunnable, name="hdbscan_scaler_runner")
runners = [
    xgb_runner, logr_runner, svm_runner, rf_runner, hdbscan_runner,
    xgb_scaler_runner, logr_scaler_runner, svm_scaler_runner, rf_scaler_runner, hdbscan_scaler_runner
]

# Crear el servicio que incluirá el runner del modelo y el runner del scaler
svc = bentoml.Service(
    "ai4i2020__failure__prediction__service",
    runners=runners,
)

FEATURES_12 = 12
FEATURES_7  = 7

sample_12 = [[298.1, 308.6, 1551, 42.8, 0, 0, 0, 0, 0, 0, 0, 1]]
sample_7  = [[0.0] * FEATURES_7]

@svc.api(input=NumpyNdarray.from_sample(sample_12), output=NumpyNdarray())
async def predict_logreg(input_data: np.ndarray) -> np.ndarray:
    scaled = await logr_scaler_runner.transform.async_run(input_data)
    probs  = await logr_runner.predict.async_run(scaled)
    return probs

@svc.api(input=NumpyNdarray.from_sample(sample_7), output=NumpyNdarray())
async def predict_random_forest(input_data: np.ndarray) -> np.ndarray:
    scaled = await rf_scaler_runner.transform.async_run(input_data)
    preds  = await rf_runner.predict.async_run(scaled)   # multioutput (5 columnas)
    return preds

@svc.api(input=NumpyNdarray.from_sample(sample_12), output=NumpyNdarray())
async def predict_svm(input_data: np.ndarray) -> np.ndarray:
    scaled = await svm_scaler_runner.transform.async_run(input_data)
    probs  = await svm_runner.predict.async_run(scaled)
    return probs

@svc.api(input=NumpyNdarray.from_sample(sample_12), output=NumpyNdarray())
async def predict_xgb(input_data: np.ndarray) -> np.ndarray:
    scaled = await xgb_scaler_runner.transform.async_run(input_data)
    preds  = await xgb_runner.predict.async_run(scaled)
    return preds

@svc.api(input=NumpyNdarray.from_sample(sample_7), output=NumpyNdarray())
async def cluster_hdbscan(input_data: np.ndarray) -> np.ndarray:
    scaled = await hdbscan_scaler_runner.transform.async_run(input_data)
    labels = await hdbscan_runner.predict.async_run(scaled)  # o fit_predict según lo que guardaste
    return labels