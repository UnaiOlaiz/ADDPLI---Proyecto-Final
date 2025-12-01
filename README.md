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
## Pasos para ejecutar el ui.py
Primero tener un enironment con las librerias necesarias. Instalar streamlit, pandas, matplotlib, seaborn y altair.
Como ejecutar: desde la carpeta general del proyecto, usar este comando: "streamlit run src/Streamlit/ui.py"
Al ejecutar el comando, clickar en la url que sale.
