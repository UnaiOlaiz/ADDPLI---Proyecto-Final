# Proyecto Final Analítica de Datos para la Industria - Grupo 2

## Estructura del Proyecto
```
├── data (carpeta de datos)
│   └── ai4i2020.csv (nuestro dataset)
├── requirements.txt (fichero de dependencias)
└── src (source)
    ├── BentoML (rama de BentoML)
    │   ├── service.py (script del servicio)
    │   └── train.py (script de entrenamiento)
    └── Streamlit (rama de Streamlit)
        └── ui.py (script de interfaz gráfica)
```

## Instalación de dependencias
El fichero **requirements.txt** incluye todas las dependencias previas para trabajar con el proyecto, para instalarlas:
```bash
pip install -r requirements.txt
```

## Algoritmos de ML trabajados:
1. Primero hemos hecho una regresión logística para predecir el output (binario) de un estado maquinario. En nuestro caso intentaremos predecir la variable de **Machine failure**


## Pasos para ejecutar la aplicación de Streamlit (ui.py)
Dentro del directorio del proyecto, ejecutar el siguiente comando para abrir localmente la interfaz gráfica trabajada con **Streamlit**.
```bash
streamlit run src/Streamlit/ui.py
```

