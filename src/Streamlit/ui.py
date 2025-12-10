import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# Configuración de la página
st.set_page_config(
    page_title="Mantenimiento Predictivo WAI4I 2020",
    layout="wide"
)

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("data/ai4i2020.csv")  
    return df

df = load_data()

# Columnas continuas y de fallos
numeric_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
fallos_cols = ["TWF","HDF","PWF","OSF","RNF"]

# Calcular conteo total de fallos
fallos_count = df[fallos_cols].sum()

# Sidebar para navegación
st.sidebar.title("Navegación")
opcion = st.sidebar.radio("Selecciona una sección:", ["Exploración de datos", "Predicciones"])
fault_info = pd.DataFrame({
    'Fault': ['Desgaste de herramienta ', 'Disipación de calor', 'Potencia fuera de rango', 'Sobreesfuerzo mecánico', 'Falla Aleatoria'],
    'Abbrev.': ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
})

# Mostrar la tabla en la barra lateral
st.sidebar.subheader("Significado de Fallos")
st.sidebar.table(fault_info)

# Contenido principal
st.title("Análisis y Predicción de Fallos ")
st.markdown("""
Este panel utiliza el dataset **WAI4I 2020**, que contiene datos de sensores de máquinas industriales para analizar y predecir fallos.  
Incluye variables como temperatura del aire y del proceso, velocidad de rotación, torque y desgaste de la herramienta, así como registros de distintos tipos de fallos:  

- **TWF:** Desgaste de herramienta  
- **HDF:** Disipación de calor  
- **PWF:** Potencia fuera de rango  
- **OSF:** Sobreesfuerzo mecánico  
- **RNF:** Falla aleatoria  

Cada registro corresponde a una máquina en un momento determinado. Con esta información podemos explorar patrones, estudiar relaciones entre variables y desarrollar modelos de mantenimiento predictivo.
""")

# Exploración de datos
if opcion == "Exploración de datos":
    
    

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

        chart = alt.Chart(fallos_df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Cantidad", type="quantitative"),
            color=alt.Color(field="Tipo de fallo", type="nominal"),
            tooltip=["label:N", "Cantidad:Q"]
        )
        st.altair_chart(chart, use_container_width=True)

       

    # Gráfico interactivo: fallos vs variable seleccionada en col2
    # Gráfico interactivo mejorado: fallos vs variable seleccionada en col2
    with col2:
        st.subheader("Cantidad de fallos por tipo de máquina y tipo de fallo")

        # Extraer la categoría del producto (L, M, H)
        df["MachineType"] = df["Product ID"].str[0]

        # Agrupa por tipo de fallo sumando por tipo de máquinaa
        fail_long = (
            df.groupby(["MachineType"])[fallos_cols]
            .sum()
            .reset_index()
            .melt(
                id_vars="MachineType",
                var_name="Tipo de fallo",
                value_name="Cantidad"
            )
        )

        # Filtro interactivo por máquina
        selected_machine = st.selectbox("Filtra por tipo de máquina:", sorted(df["MachineType"].unique()))
        fail_filtered = fail_long[fail_long["MachineType"] == selected_machine]

        # Colores consistentes para tipos de fallo
        color_scale = alt.Scale(domain=fallos_cols, range=["#4CAF50", "#FFC107", "#F44336", "#2196F3", "#9C27B0"])

        # Gráfico de barras horizontales interactivo
        bar_chart = (
            alt.Chart(fail_filtered)
            .mark_bar()
            .encode(
                y=alt.Y("Tipo de fallo:N", sort="-x", title="Tipo de fallo"),
                x=alt.X("Cantidad:Q", title="Cantidad total"),
                color=alt.Color("Tipo de fallo:N", title="Tipo de fallo", scale=color_scale),
                tooltip=[
                    alt.Tooltip("Tipo de fallo:N"),
                    alt.Tooltip("Cantidad:Q", format=",")
                ]
            )
            .properties(width=500, height=350)
        )

        st.altair_chart(bar_chart, use_container_width=True)

        
    

    # Aqui se pueden hacer mas columnas
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
        # Matriz de co-ocurrencia de fallos
        st.subheader("Matriz de co-ocurrencia de fallos")
        co_occur = df[fallos_cols].T.dot(df[fallos_cols])
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(co_occur, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    
        # Resumen estadístico general
    st.subheader("Resumen general de fallos")

    # Total de fallos
    total_fallos = df[fallos_cols].sum().sum()

    # Tipo de fallo más frecuente
    tipo_mas_frecuente = df[fallos_cols].sum().idxmax()
    cantidad_mas_frecuente = df[fallos_cols].sum().max()

    # Número de máquinas con al menos un fallo
    num_maquinas_fallo = df[df[fallos_cols].sum(axis=1) > 0]["Product ID"].nunique()

    # Mostrar KPIs en columnas
    kpi1, kpi2, kpi3 = st.columns(3)

    kpi1.metric("Total de fallos", f"{total_fallos}")
    kpi2.metric("Tipo de fallo más frecuente", f"{tipo_mas_frecuente} ({cantidad_mas_frecuente})")
    kpi3.metric("Máquinas con al menos un fallo", f"{num_maquinas_fallo}")
    
        


    # Mostrar tabla de datos
    if st.checkbox("Mostrar tabla de datos"):
        st.subheader("Primeras filas del dataset")
        st.dataframe(df.head())

    # Mostrar estadísticas descriptivas
    if st.checkbox("Mostrar estadísticas descriptivas"):
        st.subheader("Estadísticas descriptivas de variables numéricas")
        st.dataframe(df.describe())
        




   
    

    
    


  


    
# ------------------------------------
# SECCIÓN 2: Predicciones
# ------------------------------------
elif opcion == "Predicciones":
    st.header("📈 Predicciones")
    st.write("Aquí se implementará el modelo y su interfaz para hacer predicciones con nuevos datos.")
