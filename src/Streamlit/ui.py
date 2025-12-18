import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
import requests 
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configuración de la página
st.set_page_config(
    page_title="Mantenimiento Predictivo WAI4I 2020",
    layout="wide"
)

# Cargar datos
@st.cache_data
def load_data():
    # Asegúrate de que esta ruta sea correcta para Streamlit
    df = pd.read_csv("data/ai4i2020.csv") 
    return df

df = load_data()

# URL de donde estará escuchando nuestro servicio creado
BENTO_URL_BASE = "http://localhost:3000"

def call_bento_api(endpoint_name: str, input_features: list) -> dict:
    """Llama al endpoint del modelo de BentoML con los datos de entrada."""
    url = f"{BENTO_URL_BASE}/{endpoint_name}"
    
    # El esquema de Pydantic espera una lista de listas
    data_to_send = {"input_data": [input_features]}

    try:
        response = requests.post(
            url,
            json=data_to_send,  
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Esto capturará errores de conexión o errores HTTP
        return {"error": f"Error al conectar con la API ({endpoint_name}): {e}"}

# Columnas continuas y de fallos
numeric_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
fallos_cols = ["TWF","HDF","PWF","OSF","RNF"]

# Calcular conteo total de fallos
fallos_count = df[fallos_cols].sum()

# Sidebar para navegación
st.sidebar.title("Navegación")
opcion = st.sidebar.radio("Selecciona una sección:", ["Exploración de datos", "Predicciones"])

# Contenido principal
st.title("Análisis y Predicción de Fallos ")
st.markdown("""
Este panel utiliza el dataset `WAI4I 2020`, que contiene datos de sensores de máquinas industriales para analizar y predecir fallos en un entorno controlado y sintético. Incluye variables como temperatura del aire y del proceso, velocidad de rotación, torque y desgaste de la herramienta, ... entre otros.
            
Nuestro objetivo va a ser anticiparnos, mediante modelos de `ML`, a posibles fallos que puedan llegar a sufrir estos procesos maquinarios. Nuestro conjunto de datos cuenta con la variable de `Machine Failure`, la cual indicará si el procesamiento de el producto a instanciar ha resultado en fallo o no. Por otra parte, los fallos maquinarios se descomponen en diferentes tipos, debido a que pueden llegar a haber sido producido por ciertos factores externos. Los cuales son los siguientes: 

- `TWF:` Desgaste de herramienta  
- `HDF:` Disipación de calor  
- `PWF:` Potencia fuera de rango  
- `OSF:` Sobreesfuerzo mecánico  
- `RNF:` Falla aleatoria  

Cada registro corresponde a una manufacturación de un producto en un momento determinado. Con esta información podemos explorar patrones, estudiar relaciones entre variables y desarrollar modelos de mantenimiento predictivo.
""")

st.markdown("---")

st.markdown("""            
Esta aplicación `Streamlit` contará, además, de la posibilidad de desplegar y utilizar los modelos anteriormente creados y entrenados en la plataforma `BentoML`. Cada algoritmo empleado buscará especializarse en una tarea en específico, y requerirá de diferentes variables para poder llegar a realizar ejecuciones exitosas. A continuación, alistaremos los modelos que serán accesible en la pestaña de `Predicciones`:
            
Modelos disponibles:
- `Logistic Regresión:` Algoritmo de aprendizaje supervisado de tipo regresivo para predecir si el proceso de manufacturación resultará en fallo o no. 
- `Random Forest:` Algoritmo clasificatorio para predecir el tipo de fallo (en el caso de que haya habido) entre los diferentes tipos mencionados anteriormente. 
- `SVMs (Support Vector Machines):` Algoritmo supervisado que buscará clasificar en un espacio vectorial, las instancias que hayan resultado en falla y no.
- `XGBoostClassifier:` Algoritmo el cual comparte objetivo con la *Regresión Logística*, pero la cual mediante diferentes capas de árboles estructurados, buscará una más consolidada predicción.   
- `HDBSCAN:` Algoritmo de aprendizaje no supervisado que intentará agrupar las instancias más similares entre sí, formando agrupaciones las cuales permitirán analizar patrones significativos.
                
""")

st.markdown("---")

# Exploración de datos
if opcion == "Exploración de datos":
    
    fallos_colors = ["#4CAF50", "#FFC107", "#F44336", "#2196F3", "#9C27B0"]
    fallos_color_scale = alt.Scale(domain=fallos_cols, range=fallos_colors)

    # KPIs generales
    st.subheader("KPIs Generales")

    # KPI 1 – Total de eventos de fallo 
    total_eventos_fallo = df[fallos_cols].sum().sum()
    # KPI 2 – Tipo de fallo más frecuente
    fallos_por_tipo = df[fallos_cols].sum()
    tipo_fallo_frecuente = fallos_por_tipo.idxmax()
    cantidad_fallo_frecuente = fallos_por_tipo.max()
    # KPI 3 – Porcentaje de observaciones con fallo
    total_registros = len(df)
    registros_con_fallo = (df["Machine failure"] == 1).sum()
    porcentaje_fallo = registros_con_fallo / total_registros * 100
    # KPI 4 – Desgaste medio de herramienta cuando ocurre fallo
    tool_wear_fallo = df[df["Machine failure"] == 1]["Tool wear [min]"].mean()
    # KPI 5 - Tipo de producto más riesgoso
    fallos_por_tipo_prod = (
    df.groupby("Type")["Machine failure"]
    .mean() * 100
    )
    tipo_maquina_riesgo = fallos_por_tipo_prod.idxmax()
    riesgo_tipo_producto = fallos_por_tipo_prod.max()


    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Eventos totales de fallo",
        f"{int(total_eventos_fallo)}"
    )

    k2.metric(
        "Fallo más frecuente",
        f"{tipo_fallo_frecuente} ({int(cantidad_fallo_frecuente)})"
    )

    k3.metric(
        "% Registros con fallo",
        f"{porcentaje_fallo:.2f}%"
    )

    k4.metric(
    "Tool wear medio al fallo",
    f"{tool_wear_fallo:.1f} min"
    )

    k5.metric(
    "Tipo de producto más riesgoso",
    f"{tipo_maquina_riesgo} ({riesgo_tipo_producto:.2f}%)"
    )

    st.markdown("---")


    # Crear columnas para gráficos lado a lado
    col1, col2 = st.columns(2)

    # Pie chart de fallos con porcentajes en col1
    with col1:
        st.subheader("Distribución de los tipos de fallos")
        fallos_df = pd.DataFrame({
            "Tipo de fallo": fallos_cols,
            "Cantidad": df[fallos_cols].sum().values
        })
        fallos_df["Porcentaje"] = (fallos_df["Cantidad"] / fallos_df["Cantidad"].sum() * 100).round(1)
        fallos_df["label"] = fallos_df["Tipo de fallo"] + " (" + fallos_df["Porcentaje"].astype(str) + "%)"

        fallos_colors = ["#4CAF50", "#FFC107", "#F44336", "#2196F3", "#9C27B0"]
        chart = alt.Chart(fallos_df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Cantidad", type="quantitative"),
            color=alt.Color(field="Tipo de fallo", type="nominal", scale=alt.Scale(range=fallos_colors)),
            tooltip=["label:N", "Cantidad:Q"]
        )
        st.altair_chart(chart, use_container_width=True)

    # Gráfico interactivo: fallos vs variable seleccionada en col2
    with col2:
        st.subheader("Cantidad de fallos por tipo de producto")
        df["Type"] = df["Type"]
        fail_long = (
            df.groupby(["Type"])[fallos_cols]
            .sum()
            .reset_index()
            .melt(id_vars="Type", var_name="Tipo de fallo", value_name="Cantidad")
        )

        # Filtro por producto
        selected_prod = st.selectbox("Filtra por tipo de producto:", ["Todas"] + sorted(df["Type"].unique()))
        if selected_prod != "Todas":
            fail_filtered = fail_long[fail_long["Type"] == selected_prod]
        else:
            fail_filtered = fail_long

        color_scale = alt.Scale(domain=fallos_cols, range=fallos_colors)
        stacked_bar = (
            alt.Chart(fail_filtered)
            .mark_bar()
            .encode(
                x=alt.X("Tipo de fallo:N", title="Tipo de fallo"),
                y=alt.Y("Cantidad:Q", title="Cantidad total"),
                color=alt.Color("Tipo de fallo:N", scale=color_scale),
                tooltip=["Tipo de fallo:N", "Cantidad:Q"]
            )
            .properties(width=500, height=350)
        )
        st.altair_chart(stacked_bar, use_container_width=True)

    st.markdown("---")

    
    col3, col4 = st.columns(2)

    # Boxplots en col3
    with col3:
        st.subheader("Cantidad de fallos según variable seleccionada")
        selected_var = st.selectbox("Selecciona la variable X:", numeric_cols)
        selected_fail = st.selectbox("Selecciona el tipo de fallo:", fallos_cols)
        num_bins = st.slider("Número de bins:", min_value=5, max_value=50, value=10)

        df_plot = df[[selected_var, selected_fail]].copy()
        df_plot["bin"] = pd.cut(df_plot[selected_var], bins=num_bins)
        fail_counts = df_plot.groupby("bin")[selected_fail].sum().reset_index()
        fail_counts["bin_str"] = fail_counts["bin"].astype(str)

        # Gráfico de líneas
        line_fail = alt.Chart(fail_counts).mark_line(point=True, color="salmon").encode(
            x=alt.X("bin_str:N", title=selected_var),
            y=alt.Y(f"{selected_fail}:Q", title=f"Cantidad de {selected_fail}"),
            tooltip=["bin_str", selected_fail]
        ).properties(width=500, height=300)

        st.altair_chart(line_fail)
        
    # Histogramas en col4
    with col4:
        st.subheader("Matriz de co-ocurrencia de fallos")
        co_occur = df[fallos_cols].T.dot(df[fallos_cols])
        fig = px.imshow(co_occur, text_auto=True, color_continuous_scale='Blues', width=1000, height=600)
        fig.update_xaxes(side="top") 
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    with st.expander("Correlación de variables continuas"):
        st.write("Heatmap de correlaciones entre variables numéricas")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig, use_container_width=False)

    with st.expander("Distribución de variables continuas"):
        st.write("Selecciona la variable para ver su histograma")
        selected_numeric = st.selectbox("Variable:", numeric_cols)
        
        fig, ax = plt.subplots(figsize=(5,4)) 
        sns.histplot(df[selected_numeric], kde=True, bins=30, color="#4CAF50")
        ax.set_title(selected_numeric)
        st.pyplot(fig, use_container_width=False)
        
    with st.expander("Datos y estadísticas"):
        if st.checkbox("Mostrar tabla de datos"):
            st.dataframe(df.head())
        if st.checkbox("Mostrar estadísticas descriptivas"):
            st.dataframe(df.describe())

# Pestaña de predicciones
elif opcion == "Predicciones":
    st.header("Predicción de Fallos en Tiempo Real")
    st.markdown("Consulta la API de BentoML con nuevos parámetros. El servicio se ejecuta en http://localhost:3000.")

    def build_features_7(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        numeric_cols = ["Air temperature [K]", "Process temperature [K]",
                        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

        # dummies como en tu UI: L y M (H queda implícito)
        type_L = (df["Type"] == "L").astype(int)
        type_M = (df["Type"] == "M").astype(int)

        X = df[numeric_cols].copy()
        X["Type_L"] = type_L
        X["Type_M"] = type_M

        # orden exacto: 5 continuas + L + M  => 7
        X = X[numeric_cols + ["Type_L", "Type_M"]]

        y = df["Machine failure"].astype(int)
        return X, y


    def bento_predict_proba(endpoint: str, X: pd.DataFrame) -> np.ndarray:
        payload = {"input_obj": {"input_data": X.values.tolist()}}
        url = f"{BENTO_URL_BASE}/{endpoint}"
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return np.array(r.json())  # esperado shape (n,2)


    @st.cache_data(show_spinner=False)
    def compute_binary_metrics_via_bento(df: pd.DataFrame) -> pd.DataFrame:
        X, y = build_features_7(df)

        # Ajusta random_state si en vuestro notebook usasteis otro
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        models = [
            ("XGBoost", "predict_xgb"),
            ("Regresión Logística", "predict_logreg"),
            ("SVM", "predict_svm"),
        ]

        rows = []
        for name, endpoint in models:
            proba = bento_predict_proba(endpoint, X_test)[:, 1]   # prob de clase 1
            y_pred = (proba >= 0.5).astype(int)

            rows.append({
                "Modelo": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                "AUC (ROC)": roc_auc_score(y_test, proba),
            })

        return pd.DataFrame(rows)

    # 1. Selector de Modelo y su endpoint 
    MODEL_ENDPOINTS = {
        "XGBoost (Algoritmo de Clasificación)": "predict_xgb", 
        "Regresión Logística (Algoritmo de Regresión)": "predict_logreg",
        "SVM (Algoritmo de Clasificación)": "predict_svm",
        "Random Forest (Algoritmo de Clasificación)": "predict_random_forest", 
        "HDBSCAN (Algoritmo de Agrupación)": "cluster_hdbscan",
    }
    selected_model_display = st.selectbox(
        "Modelo a utilizar:", 
        list(MODEL_ENDPOINTS.keys())
    )
    endpoint_to_call = MODEL_ENDPOINTS[selected_model_display]
    
    # Columnas necesareas
    # if endpoint_to_call in ["predict_xgb", "predict_logreg", "predict_svm"]:
    #     required_cols = 12
    # else:
    #     required_cols = 7

    st.subheader(f"Ingreso de Parámetros 7 Features Requeridas)")
    
    # Creacion delformulario para garantizar que los datos se envíen juntos
    with st.form("prediction_form"):
        col_t1, col_t2 = st.columns(2)

        # INPUTS Continuos
        with col_t1:
            st.markdown("*Variables Continuas:*")
            temp_aire = st.number_input("Air temperature [K]", value=299.1, step=0.1)
            temp_proceso = st.number_input("Process temperature [K]", value=309.2, step=0.1)
            velocidad = st.number_input("Rotational speed [rpm]", value=1530, step=1)
            torque = st.number_input("Torque [Nm]", value=40.1, step=0.1)
            desgaste = st.number_input("Tool wear [min]", value=100, step=1)
        
        # INPUTS Dummies
        with col_t2:
            st.markdown("*Tipo de Producto:*")
            machine_type = st.radio("Selecciona Tipo:", ('L', 'M', 'H'), index=0, key="machine_type_radio")
            
            type_L = 1 if machine_type == 'L' else 0
            type_M = 1 if machine_type == 'M' else 0

            # Inicializaión de los fallos
            # twf, hdf, pwf, osf, rnf = 0, 0, 0, 0, 0
            
            # if required_cols == 12:
            #     st.markdown("*Fallos Históricos (5 Dummies):*")
            #     twf = st.checkbox("TWF (Tool Wear Failure)", value=False)
            #     hdf = st.checkbox("HDF (Heat Dissipation Failure)", value=False)
            #     pwf = st.checkbox("PWF (Power Failure)", value=False)
            #     osf = st.checkbox("OSF (Overstrain Failure)", value=False)
            #     rnf = st.checkbox("RNF (Random Failure)", value=False)
            # else:
            #     st.info("El modelo de 7 Features solo utiliza las variables continuas y el tipo de máquina (L, M).")
        
        submitted = st.form_submit_button("Obtener Predicción")
        
        if submitted:
            st.warning("Verificando el orden de las features...")

            # if required_cols == 7:
            #     input_features = [
            #         temp_aire,
            #         temp_proceso,
            #         velocidad,
            #         torque,
            #         desgaste,
            #         # int(twf),
            #         # int(hdf),
            #         # int(pwf),
            #         # int(osf),
            #         # int(rnf),
            #         type_L,
            #         type_M,
            #     ]

            # else:
            input_features = [temp_aire, temp_proceso, velocidad, torque, desgaste, type_L, type_M]
            
            # Función para llamar a la API con payload completo (input_obj)
            def call_bento_api_raw(endpoint_name: str, payload: dict) -> dict:
                url = f"{BENTO_URL_BASE}/{endpoint_name}"
                try:
                    response = requests.post(
                        url,
                        json=payload,
                        timeout=10
                    )
                    response.raise_for_status()
                    return response.json()
                except Exception as e:
                    return {"error": str(e)}

            # Preparar payload en el formato que espera BentoML
            payload = {
                "input_obj": {
                    "input_data": [input_features]
                }
            }

            st.info(f"Conectando con la API y usando el modelo: *{selected_model_display}*...")
            result = call_bento_api_raw(endpoint_to_call, payload)
            
            # Mostrar resultados
            if "error" in result:
                st.error(f"Fallo en la conexión o la API: {result['error']}")
                st.code(f"Error de la API: {result['error']}")
            else:
                pred_array = np.array(result)
                
                st.subheader("Resultado de la Predicción:")

                if "cluster" in endpoint_to_call or pred_array.shape[-1] == 1:
                    prediction = int(pred_array.flatten()[0])
                    label = "Cluster Asignado" if "cluster" in endpoint_to_call else "Clase Predicha (0=Normal, 1=Fallo)"
                    st.metric(label=label, value=prediction)
                    
                    if prediction == 1:
                        st.error("*ALERTA: FALLO PREVISTO* (Clase 1)")
                    elif prediction == 0:
                        st.success("*Operación Normal* (Clase 0)")
                    elif prediction == -1 and "cluster" in endpoint_to_call:
                        st.warning("Patrón Atípico (Ruido -1).")

                elif pred_array.shape[-1] == 2:
                    prob_fail = float(pred_array.flatten()[1])
                    st.metric("Probabilidad de Fallo (Clase 1)", f"{prob_fail:.2%}")

                    if prob_fail > 0.5:
                        st.error("*ALERTA: FALLO PREVISTO* (Probabilidad > 50%)")
                    else:
                        st.success("Operación Normal (Probabilidad <= 50%)")

                elif "random_forest" in endpoint_to_call:
                    st.markdown("##### Resultados Multi-Etiqueta (Random Forest)")
                    
                    fallo_predicho = pred_array[0]
                    fallos_cols_rf = ["TWF","HDF","PWF","OSF","RNF"] 
                    
                    fallos_df_pred = pd.DataFrame([fallo_predicho], columns=fallos_cols_rf)

                    if fallos_df_pred.sum(axis=1).iloc[0] == 0:
                        st.success("*Predicción: Ningún fallo específico*")
                    else:
                        st.error("*Fallo(s) detectado(s)*")
                        fallos_activos = fallos_df_pred.columns[fallos_df_pred.iloc[0] == 1].tolist()
                        st.code(f"Tipos de fallo predichos: {', '.join(fallos_activos)}")

                else:
                    st.json(result)

    # Evaluación y comparación entre nuestros modelos
    st.header("Evaluación y Comparativa de Modelos")
    st.markdown("Análisis de las métricas clave y la justificación del mejor modelo para la predicción de fallos.")

    # 1. Tabla de Métricas de Clasificación Binaria (Fallo General)
    st.markdown("### 1. Métricas de Modelos de Clasificación Binaria")
    st.write("Métricas de los modelos que predicen 'Machine Failure' (Clase 0 o 1).")

    # Datos REALES (Extraídos del notebook)
    # data_clasificacion = {
    #     'Modelo': ["XGBoost", "Regresión Logística", "SVM"],
    #     'Accuracy': [0.9990, 0.9990, 0.9990],
    #     'Precision': [1.0000, 1.0000, 1.0000], 
    #     'Recall': [0.9672, 0.9672, 0.9672],
    #     'F1 Score': [0.9833, 0.9833, 0.9833],
    #     'AUC (ROC)': [0.9990, 0.9990, 0.9990]
    # }

    try:
        df_metricas = compute_binary_metrics_via_bento(df)
        st.dataframe(df_metricas.style.highlight_max(
            subset=['Accuracy','Precision','Recall','F1 Score','AUC (ROC)'],
            axis=0,
            props='font-weight: bold; background-color: #d8f5d8; color: #000000;'
        ).format(precision=4), use_container_width=True)
    except Exception as e:
        st.warning("No se pudieron calcular métricas automáticamente (¿BentoML está levantado?).")
        st.code(str(e))

    best_row = df_metricas.loc[df_metricas["Recall"].idxmax()]
    best_model = best_row["Modelo"]
    best_recall = float(best_row["Recall"])

    st.info(f"""
    *Conclusión sobre la Predicción Binaria:*
    Con los resultados actuales, el modelo con mejor capacidad para **detectar fallos reales** (mayor *Recall*) es **{best_model}**.  
    El *Recall ({best_recall:.4f})* es especialmente importante en mantenimiento predictivo porque minimiza los **Falsos Negativos** (fallos reales que el modelo no detecta), que en un entorno industrial suelen ser el error más costoso.

    En nuestro caso, **XGBoost** sigue siendo una opción muy sólida para producción por su buen equilibrio global de métricas y su robustez.
    """)
    # 2. Justificación y Visualizaciones del Mejor Modelo
    mejor_modelo_nombre = "XGBoost" 
    st.markdown(f"### 2. Análisis del Mejor Modelo: *{mejor_modelo_nombre}*")

    col_conf, col_roc = st.columns(2)

    with col_conf:
        st.markdown("#### Matriz de Confusión")
        st.write(f"Distribución de True/False Positives/Negatives para {mejor_modelo_nombre}.")
        

        try:
            st.image("img/xgb_confusion.png", caption=f"Matriz de Confusión de {mejor_modelo_nombre}") 
        except Exception:
            st.warning("No se encontró la imagen 'img/xgb_confusion.png'. Asegúrate de que está en la carpeta 'img'.")
            
    with col_roc:
        st.markdown("#### Curva ROC y Área bajo la Curva (AUC)")

        # Tomamos el AUC del modelo XGBoost desde la tabla automática
        try:
            xgb_auc = float(
                df_metricas.loc[df_metricas["Modelo"] == "XGBoost", "AUC (ROC)"].iloc[0]
            )

            st.write(
                f"El valor de AUC de **{0.9667}** confirma su alta capacidad discriminatoria."
            )
        except Exception as e:
            st.warning("No se pudo obtener el AUC automáticamente desde la tabla de métricas.")
            st.code(str(e))

        # Imagen de la curva ROC (si existe)
        try:
            st.image(
                "img/xgb_rocauc.png",
                caption="Curva ROC de XGBoost"
            )
        except Exception:
            st.warning(
                "No se encontró la imagen 'img/xgb_rocauc.png'. "
                "Asegúrate de que está en la carpeta 'img'."
            )

    # 3. Evaluación de Random Forest (Clasificación de Fallos Específicos) 
    st.markdown("### 3. Evaluación de Random Forest (Clasificación Multi-Etiqueta)")
    st.write("""
    El modelo Random Forest atiende a la pregunta *'Si hay un fallo, ¿cuál de los 5 tipos es?'*. Se evalúa con métricas ponderadas.
    """)

    # Datos REALES 
    rf_accuracy = 0.9815
    rf_precision = 0.7468
    rf_recall = 0.4638
    rf_f1 = 0.5681

    col_rf1, col_rf2, col_rf3 = st.columns(3)
    col_rf1.metric(label="Accuracy Total (RF)", value=f"{rf_accuracy:.2%}")
    col_rf2.metric(label="Precision Ponderada (RF)", value=f"{rf_precision:.2%}")
    col_rf3.metric(label="Recall Ponderado (RF)", value=f"{rf_recall:.2%}")

    st.warning(f"""
    *Análisis del Random Forest:*
    El *Recall Ponderado ({rf_recall:.2%})* es bajo, lo que indica que el modelo tiene dificultades para identificar correctamente los tipos de fallos específicos. Este modelo debe usarse solo para clasificar el tipo después de que un modelo binario (XGBoost) haya predicho que ocurrirá una falla.
    """)