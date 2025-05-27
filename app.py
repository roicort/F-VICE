import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import plotly.express as px
from utils import get_itslive, get_processed_data
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
            df = get_processed_data(df, min_dt=min_dt, max_dt=max_dt)
            st.session_state.df = df  # Guarda el DataFrame en el estado de sesión
    # Mostrar la gráfica y la tabla si ya hay datos en el estado de sesión
    if "df" in st.session_state and not st.session_state.df.empty:
        df = st.session_state.df
        fig = px.scatter(
            df,
            x="mid_date",
            y="v",
            labels={"v": "Velocidad (m/año)", "mid_date": "Fecha"},
            title=f"Serie de tiempo de velocidad ITS_LIVE ({min_dt}-{max_dt} días)",
            trendline="lowess",
            color_discrete_sequence=[primary_color]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df)
        # Botón para entrenar el modelo
        if st.button("Entrenar modelo de predicción"):
            split_idx = int(len(df) * 0.50)
            X = df[['mid_date','year', 'month', 'dayofyear']]
            y = df['v']
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            plot_df = pd.DataFrame({
                'date': pd.concat([X_train['mid_date'], X_test['mid_date']]),
                'v [m/yr]': pd.concat([y_train, y_test]),
                'split': ['train'] * len(X_train) + ['test'] * len(X_test)
            })

            fig = px.scatter(
                plot_df,
                x='date',
                y='v [m/yr]',
                color='split',
                title='Train/Test Split: v [m/yr] over Time',
                opacity=0.7,
                trendline='lowess',
            )
            st.plotly_chart(fig, use_container_width=True)
    elif "df" in st.session_state and st.session_state.df.empty:
        st.error("No se encontraron datos para las coordenadas seleccionadas.")