# Proyecto Final Analítica de Datos para la Industria - Grupo 2

## Estructura del Proyecto
```
├── data
│   ├── ai4i2020_cleaned.csv
│   ├── ai4i2020.csv
│   └── datos_para_demo.csv
├── img
│   ├── xgb_confusion.png
│   └── xgb_rocauc.png
├── README.md
├── requirements.txt
└── src
    ├── BentoML
    │   ├── __pycache__
    │   │   ├── service.cpython-311.pyc
    │   │   ├── service.cpython-313.pyc
    │   │   └── service_v2.cpython-313.pyc
    │   ├── service_notebook.ipynb
    │   ├── service.py
    │   └── train.ipynb
    ├── Limpieza
    │   └── cleaning.ipynb
    └── Streamlit
        ├── ui.ipynb
        └── ui.py

```

## Instalación de dependencias
El fichero **requirements.txt** incluye todas las dependencias previas para trabajar con el proyecto, para instalarlas:
```bash
pip install -r requirements.txt
```

## Algoritmos de ML trabajados:
1. `Regresión Logística`

2. `Random Forest`

3. `Support Vector Machines (SVMs)`

4. `XGBClassifier`

5. `HDBSCAN`

Para listar los modelos creados, introduciremos el siguiente comando:
```bash
bentoml models list
```
Deberíamos de ver ahora parámetros como el nombre del modelo, el módulo empleado para la creación, tamaño y la fecha de creación.

Para desplegar BentoML, primero tendremos que navegar a la ruta correcta:
```bash
cd src/BentoML/
```
Y luego introducir el comando correcto:
```bash
bentoml serve service:AI4I2020FailurePredictionService --port 3000
```

## Pasos para ejecutar la aplicación de Streamlit (ui.py)
Dentro del directorio del proyecto, ejecutar el siguiente comando para abrir localmente la interfaz gráfica trabajada con **Streamlit**.
```bash
streamlit run src/Streamlit/ui.py
```

