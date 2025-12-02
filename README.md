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
1. Primero hemos hecho una **Regresión Logística** para predecir el output (binario) de un estado maquinario. En nuestro caso intentaremos predecir la variable de **Machine failure**; si el proceso resulta en error, devolverá un 1, si no un 0.

2. Después, hemos creado un clasificador con **Random Forest** para clasificar el tipo de error cometido, siendo el error uno de entre cinco tipos: 
    - TWF: Fallo por desgaste de herramienta.
    - HDF: Fallo por una mala disipación de calor.
    - PWF: Falla por potencia fuera de rango.
    - OSF: Fallo por un sobreesfuerzo mecánico.
    - RNF: Un fallo aleatorio.

3. También hemos empleado los **Support Vector Machines (SVMs)** como algoritmo más complejo para realizar la clasificación binaria de la misma variable usada con la **Regresión Logística**. Cumpliendo la misma funcionalidad que éste, las **SVMs** se caracterizan por ser algoritmos de clasificación más pesados y profundos, resultando en un entrenamiento más lento.

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

