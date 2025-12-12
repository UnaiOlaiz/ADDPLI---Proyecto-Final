# Proyecto Final Analítica de Datos para la Industria - Grupo 2

## Estructura del Proyecto
```
├── data (carpeta de datos)
│   └── ai4i2020.csv (nuestro dataset)
├── requirements.txt (fichero de dependencias)
└── src (source)
    ├── BentoML (rama de BentoML)
    │   ├── service.ipynb (script del servicio)
    │   └── train.ipynb (script de entrenamiento)
    └── Streamlit (rama de Streamlit)
        └── ui.ipynb (script de interfaz gráfica)
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

Para alistar los modelos creados, introduciremos el siguiente comando:
```bash
bentoml models list
```
Deberíamos de ver ahora parámetros como el nombre del modelo, el módulo empleado para la creación, tamaño y la fecha de creación.

## Pasos para ejecutar la aplicación de Streamlit (ui.py)
Dentro del directorio del proyecto, ejecutar el siguiente comando para abrir localmente la interfaz gráfica trabajada con **Streamlit**.
```bash
streamlit run src/Streamlit/ui.py
```

