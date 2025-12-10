import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px

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
    
    fallos_colors = ["#4CAF50", "#FFC107", "#F44336", "#2196F3", "#9C27B0"]
    fallos_color_scale = alt.Scale(domain=fallos_cols, range=fallos_colors)

    # KPIs generales
    st.subheader("KPIs Generales")
    total_fallos = df[fallos_cols].sum().sum()
    tipo_mas_frecuente = df[fallos_cols].sum().idxmax()
    cantidad_mas_frecuente = df[fallos_cols].sum().max()
    num_maquinas_fallo = df[df[fallos_cols].sum(axis=1) > 0]["Product ID"].nunique()
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total de fallos", f"{total_fallos}")
    kpi2.metric("Tipo de fallo más frecuente", f"{tipo_mas_frecuente} ({cantidad_mas_frecuente})")
    kpi3.metric("Máquinas con al menos un fallo", f"{num_maquinas_fallo}")

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
    # Gráfico interactivo mejorado: fallos vs variable seleccionada en col2
    with col2:
        st.subheader("Cantidad de fallos por tipo de máquina")
        df["MachineType"] = df["Product ID"].str[0]
        fail_long = (
            df.groupby(["MachineType"])[fallos_cols]
            .sum()
            .reset_index()
            .melt(id_vars="MachineType", var_name="Tipo de fallo", value_name="Cantidad")
        )

        # Filtro por máquina
        selected_machine = st.selectbox("Filtra por tipo de máquina:", ["Todas"] + sorted(df["MachineType"].unique()))
        if selected_machine != "Todas":
            fail_filtered = fail_long[fail_long["MachineType"] == selected_machine]
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
        st.subheader("Matriz de co-ocurrencia de fallos")
        co_occur = df[fallos_cols].T.dot(df[fallos_cols])
        fig = px.imshow(co_occur, text_auto=True, color_continuous_scale='Blues', width=1000, height=600)
        fig.update_xaxes(side="top")  
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    with st.expander("Correlación de variables continuas"):
        st.write("Heatmap de correlaciones entre variables numéricas")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    with st.expander("Distribución de variables continuas"):
        st.write("Selecciona la variable para ver su histograma")
        selected_numeric = st.selectbox("Variable:", numeric_cols)
        
        fig, ax = plt.subplots(figsize=(5,4))  
        sns.histplot(df[selected_numeric], kde=True, bins=30, color="#4CAF50")
        ax.set_title(selected_numeric)
        st.pyplot(fig)
        
    with st.expander("Datos y estadísticas"):
        if st.checkbox("Mostrar tabla de datos"):
            st.dataframe(df.head())
        if st.checkbox("Mostrar estadísticas descriptivas"):
            st.dataframe(df.describe())
        




   
    

    
    


  


    
# ------------------------------------
# SECCIÓN 2: Predicciones
# ------------------------------------
elif opcion == "Predicciones":
    st.header("📈 Predicciones")
    st.write("Aquí se implementará el modelo y su interfaz para hacer predicciones con nuevos datos.")
