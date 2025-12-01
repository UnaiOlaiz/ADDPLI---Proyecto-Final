# Librerías necesarias
import bentoml
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import accuracy_score

# Algoritmos que probaremos
from sklearn.linear_model import LogisticRegression # fallo/no fallo
from sklearn.ensemble import RandomForestClassifier # clasificación de errores
from sklearn import svm # Support vector machines para clasificación

# Para cada algoritmo haremos una función diferente; creando un modelo diferente para la model store, nombre diferente, ...
def logistic_regression():
    dataset = pd.read_csv("data/ai4i2020.csv")

    # Como X cogemos todas las columnas menos la variable que usaremos como 'y' y la variable de identificador
    X, y = dataset.drop(columns=["UDI", "Machine failure"]), dataset["Machine failure"] 

    # Pasamos a númericas las variables categóricas
    columnas_categoricas = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=columnas_categoricas, drop_first=True)

    # Dividimos el dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    
    # Inicializamos el modelo de regresión logística
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    
    # Realizamos las predicciones
    preds = lr.predict(X_test)

    # Por ahora solo tendremos como métrica de evaluación la accuracy
    accuracy = accuracy_score(y_test, preds)
    print(f"Accuracy obtenida: {accuracy:.4f}")

    # A partir de ahora, configuraremos el modelo para que sea compatible con BentoML
    bento_lr = bentoml.sklearn.save_model(
        "ai4i2020_logistic_regression",
        lr,
        metadata={
            "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "ai4i2020",
            "framework": "scikit-learn",
            "algoritmo": "regresión logística",
            "precision": accuracy,
            "carta_favorita_cr": "Reina Arquera MOMO SHOW"
        },
    )
    print(f"Modelo de Regresión Logística guardado en la BentoML store como: {bento_lr}")


def random_forest():
    dataset = pd.read_csv("data/ai4i2020.csv")

    # Como X cogemos todas las columnas menos la variable que usaremos como 'y' y la variable de identificador
    X, y = dataset.drop(columns=["UDI", "Machine failure", "TWF", "HDF", "PWF", "RNF", "OSF"]), dataset[["TWF", "HDF", "PWF", "RNF", "OSF"]]

    # Pasamos a númericas las variables categóricas
    columnas_categoricas = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=columnas_categoricas, drop_first=True)

    # Dividimos el dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    # Inicializamos el modelo de regresión logística
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # Realizamos las predicciones
    preds = rf.predict(X_test)

    # Por ahora solo tendremos como métrica de evaluación la accuracy
    accuracy = accuracy_score(y_test, preds)
    print(f"Accuracy obtenida: {accuracy:.4f}")

    # A partir de ahora, configuraremos el modelo para que sea compatible con BentoML
    bento_rf = bentoml.sklearn.save_model(
        "ai4i2020_random_forest",
        rf,
        metadata={
            "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "ai4i2020",
            "framework": "scikit-learn",
            "algoritmo": "clasificación con random forest",
            "precision": accuracy,
            "carta_favorita_cr": "Reina Arquera MOMO SHOW"
        },
    )
    print(f"Modelo de Clasificación con Random Forest guardado en la BentoML store como: {bento_rf}")

def support_vector_machines():
    dataset = pd.read_csv("data/ai4i2020.csv")

    # Como X cogemos todas las columnas menos la variable que usaremos como 'y' y la variable de identificador
    X, y = dataset.drop(columns=["UDI", "Machine failure"]), dataset["Machine failure"] 

    # Pasamos a númericas las variables categóricas
    columnas_categoricas = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=columnas_categoricas, drop_first=True)

    # Dividimos el dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    model_svm = svm.SVC()
    model_svm.fit(X_train, y_train)

    # Realizamos las predicciones
    preds = model_svm.predict(X_test)

    # Por ahora solo tendremos como métrica de evaluación la accuracy
    accuracy = accuracy_score(y_test, preds)
    print(f"Accuracy obtenida: {accuracy:.4f}")

    # A partir de ahora, configuraremos el modelo para que sea compatible con BentoML
    bento_svm = bentoml.sklearn.save_model(
        "ai4i2020_suppor_vector_machine",
        model_svm,
        metadata={
            "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "ai4i2020",
            "framework": "scikit-learn",
            "algoritmo": "clasificación con svms",
            "precision": accuracy,
            "carta_favorita_cr": "Reina Arquera MOMO SHOW"
        },
    )
    print(f"Modelo de Clasificación con Support Vector Machines (SVMs) guardado en la BentoML store como: {bento_svm}")




# Algoritmos a añadir: SVMs, XGboost, KNN, ...

# Función main(); cada algoritmo tendrá su función
if __name__ == "__main__":
    # logistic_regression()
    # random_forest()
    support_vector_machines()