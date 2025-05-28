import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import plotly.express as px
from utils import get_itslive, get_processed_data, get_future_dates
import json

st.set_page_config(layout="wide")
primary_color = st.get_option("theme.primaryColor")

st.title("ITS_LIVE! Predicción de series de tiempo de Velocidad de Glaciares")
st.write("")
st.write("Consulta los datos originales de [ITS_LIVE](https://its-live.jpl.nasa.gov).")
st.write("Haz clic en el mapa para seleccionar un punto. O usa los campos manuales para mover el marcador.")

# Estado para el punto seleccionado
if "coords" not in st.session_state:
    st.session_state.coords = None
if "manual_lat" not in st.session_state:
    st.session_state.manual_lat = 70.0
if "manual_lon" not in st.session_state:
    st.session_state.manual_lon = -45.0

# Mapa base: centra y haz zoom si hay coordenadas seleccionadas
if st.session_state.coords:
    map_center = st.session_state.coords
    zoom = 8 
else:
    map_center = [70, -45]
    zoom = 3

m = folium.Map(location=map_center, zoom_start=zoom)

# Agregar marcador si hay punto seleccionado
if st.session_state.coords:
    lat, lon = st.session_state.coords
    folium.Marker(
        location=[lat, lon],
        popup=f"({lat:.4f}, {lon:.4f})",
        icon=folium.Icon(color="red")
    ).add_to(m)

# Capa de velocidad ITS_LIVE
vel_layer = folium.raster_layers.TileLayer(
    tiles="https://glacierflow.nyc3.digitaloceanspaces.com/webmaps/vel_map/{z}/{x}/{y}.png",
    attr='ITS_LIVE velocity mosaic (https://its-live.jpl.nasa.gov/)',
    name="ITS_LIVE Velocity Mosaic",
    overlay=True,
    control=True,
    opacity=0.7
)
vel_layer.add_to(m)

# Otras capas opcionales (ejemplo: coastlines)
coastlines = folium.raster_layers.TileLayer(
    tiles="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/Coastlines_15m/default/GoogleMapsCompatible_Level13/{z}/{y}/{x}.png",
    attr="NASA GIBS Imagery",
    name="Coastlines",
    overlay=True,
    control=True,
    opacity=0.7
)
coastlines.add_to(m)

# Control de capas
folium.LayerControl().add_to(m)

# Mostrar mapa en Streamlit
map_data = st_folium(m, width=1000, height=500)

# Si se selecciona un punto en el mapa, actualiza los inputs y el marcador
if map_data and map_data["last_clicked"]:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    # Actualiza los valores ANTES de crear los widgets
    st.session_state.coords = (lat, lon)
    st.session_state.manual_lat = lat
    st.session_state.manual_lon = lon
    st.rerun()

def actualizar_coords():
    st.session_state.coords = (st.session_state.manual_lat, st.session_state.manual_lon)
    st.rerun()

with open("glaciares.json") as f:
    glaciares = json.load(f)

def seleccionar_glaciar():
    coords = glaciares[st.session_state.glaciar_sel]
    if coords:
        st.session_state.coords = coords
        st.session_state.manual_lat = coords[0]
        st.session_state.manual_lon = coords[1]
        st.rerun()

# Selector de glaciar predefinido
st.selectbox(
    "O selecciona un glaciar predefinido:",
    options=["Selecciona un glaciar"] + list(glaciares.keys()),
    key="glaciar_sel",
    on_change=seleccionar_glaciar,
)

# Inputs numéricos para latitud y longitud sincronizados y reactivos
col1, col2 = st.columns(2)
with col1:
    st.number_input(
        "Latitud",
        min_value=-90.0,
        max_value=90.0,
        format="%.6f",
        key="manual_lat",
        on_change=actualizar_coords
    )
with col2:
    st.number_input(
        "Longitud",
        min_value=-180.0,
        max_value=180.0,
        format="%.6f",
        key="manual_lon",
        on_change=actualizar_coords
    )

# Selector de intervalo de días
st.write("Selecciona el intervalo de separación de días entre imágenes satelitales:")
min_dt, max_dt = st.slider(
    "Intervalo (días)",
    min_value=0,
    max_value=365,
    value=(1, 120),
    step=1
)

# Graficar datos
if st.session_state.coords:
    if st.button("Graficar serie de tiempo"):
        with st.spinner("Descargando y graficando datos..."):
            df = get_itslive([st.session_state.coords])
            glacier = get_processed_data(df, min_dt=min_dt, max_dt=max_dt)
            st.session_state.glacier = glacier  # Guarda el DataFrame en el estado de sesión
    # Mostrar la gráfica y la tabla si ya hay datos en el estado de sesión
    if "glacier" in st.session_state and not st.session_state.glacier.empty:
        glacier = st.session_state.glacier
        fig = px.scatter(
            glacier,
            x=glacier.index,
            y="v",
            labels={"v": "Velocidad (m/año)", "mid_date": "Fecha"},
            title=f"Serie de tiempo de velocidad ITS_LIVE ({min_dt}-{max_dt} días)",
            trendline="lowess",
            color_discrete_sequence=[primary_color]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(glacier)

        # Selector de modelo
        modelo_sel = st.selectbox(
            "Selecciona el modelo de predicción:",
            options=["XGBoost"],
            key="modelo_sel"
        )

        # Botón para entrenar el modelo
        if st.button("Entrenar modelo de predicción"):

            split_idx = int(len(glacier) * 0.66)
            X = glacier[['year', 'month', 'dayofyear']]
            y = glacier['v']
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            plot_df = pd.DataFrame({
                'v [m/yr]': pd.concat([y_train, y_test]),
                'split': ['train'] * len(X_train) + ['test'] * len(X_test)
            })

            fig = px.scatter(
                plot_df,
                x=plot_df.index,
                y='v [m/yr]',
                color='split',
                title='Train/Test Split: v [m/yr] over Time',
                opacity=0.7,
                trendline='lowess',
            )
            st.plotly_chart(fig, use_container_width=True)

            if modelo_sel == "XGBoost":
                with st.spinner("Entrenando modelo XGBoost..."):
                        from model import get_xgboost_model
                        model, scores = get_xgboost_model(X_train, y_train)
                        st.session_state.model = model
                        st.success(f"Modelo {modelo_sel} entrenado exitosamente.")
                        st.write(f"Cross-validated neg_mean_squared_log_error: {-scores.mean():.4f} ± {scores.std():.4f}")

            # Una vez entrenado el modelo, puedes hacer predicciones
            if st.session_state.model:
                with st.spinner("Realizando predicciones..."):

                    y_pred = st.session_state.model.predict(X_test)
                    future_dates = get_future_dates(X_test.index[-1], until='2030-12-31')
                    future_predictions = model.predict(future_dates[['year', 'month', 'dayofyear']])

                    # Graficar puntos reales y predicciones
                    fig = px.scatter(
                        plot_df,
                        x=plot_df.index,
                        y='v [m/yr]',
                        color='split',
                        title='Train/Test Split & Predictions: v [m/yr] over Time',
                        opacity=0.4,
                        trendline='lowess',
                    )

                    # Agregar las predicciones como línea
                    fig.add_scatter(
                        x=y_test.index,
                        y=y_pred,
                        mode='lines',
                        name='Predicción (test)',
                        line=dict(color="#4e59f6", width=2),
                    )

                    # Agregar las predicciones futuras
                    fig.add_scatter(
                        x=future_dates.index,
                        y=future_predictions,
                        mode='lines',
                        name='Predicción (futuro)',
                        line=dict(color="#e006bf", width=2),
                    )

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se ha entrenado ningún modelo. Por favor, entrena un modelo primero.")

        elif "df" in st.session_state and st.session_state.df.empty:
            st.error("No se encontraron datos para las coordenadas seleccionadas.")