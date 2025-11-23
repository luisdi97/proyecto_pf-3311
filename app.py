import streamlit as st
import plotly.express as px
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import numpy as np
import networkx as nx
from shapely.geometry import MultiLineString, LineString, Point
from scipy.spatial import cKDTree
import pyproj
import os


# ----- Fuentes de datos -----

# El archivo original se obtuvo mediante el servicio WFS:
#  https://geos.snitcr.go.cr/be/IGN_200/wfs y la capa
#  IGN_200:redvial_200k

archivo_redvial_200k = os.path.join('datos', 'redvial_200k.gpkg')

# El archivo original se obtuvo mediante el servicio WFS:
#  https://geos.snitcr.go.cr/be/IGN_200/wfs y la capa
#  IGN_200:edificaciones_y_construcciones_200k

archivo_edificaciones_y_construcciones_200k = (
    os.path.join('datos', 'edificaciones_y_construcciones_200k.gpkg')
)

# El archivo original se obtuvo mediante el servicio WFS:
#  https://geos.snitcr.go.cr/be/IGN_5_CO/wfs y la capa
#  IGN_5_CO:limiteprovincial_5k

archivo_limiteprovincial_5k = os.path.join('datos', 'limiteprovincial_5k.gpkg')

# ----- Mapeo ubicaciones iniciales para provincias -----
dict_ubicaciones_iniciales = {
    'Cartago': [9.863744662224683, -83.9134299481798],
    'San José': [9.932865819126107, -84.0770484032477],
    'Limón': [9.99101025451099, -83.03367232154892],
    'Alajuela': [10.016366523904257, -84.21427237300195],
    'Guanacaste': [10.63584872966224, -85.43251884536156],
    'Heredia': [9.998448606539409, -84.11756704331432],
    'Puntarenas': [9.976819848977657, -84.83894343366579],
    'Todas': [9.932865819126107, -84.0770484032477],
}


# ----- Funciones para recuperar y analizar los datos -----
def red_vial_red_vial_nodos():
    limiteprovincial_gdf = (
        gpd.read_file(archivo_limiteprovincial_5k)
    )

    # Asegurarse que el CRS sea CRTM05 (EPSG 5367)
    limiteprovincial_gdf = (
        limiteprovincial_gdf.to_crs(epsg=5367)
    )

    # Reducción de columnas
    limiteprovincial_gdf = (
        limiteprovincial_gdf[['PROVINCIA', 'geometry']]
    )

    red_vial_gdf = gpd.read_file(archivo_redvial_200k)

    # Asegurarse que el CRS sea CRTM05 (EPSG 5367)
    red_vial_gdf = red_vial_gdf.to_crs(epsg=5367)

    # Reducción de columnas
    red_vial_gdf = red_vial_gdf[['categoria', 'geometry']]

    # Unión espacial
    red_vial_gdf = (
        red_vial_gdf.sjoin(
            limiteprovincial_gdf,
            predicate='intersects'
        )
    )

    # ----- Análisis para obtener el grafo -----

    # Separar las líneas en listas
    red_vial_gdf['split_lines'] = (
        red_vial_gdf['geometry'].apply(split_LineString)
    )

    # Crear filas separadas por cada elementos de la lista
    red_vial_gdf = red_vial_gdf.explode('split_lines', ignore_index=True)

    # Agregar pesos iguales a la distancia de cada segmento
    red_vial_gdf['weight'] = (
        gpd.GeoSeries(red_vial_gdf['split_lines']).length
    )

    # Obtener los puntos de inicio y fin de cada segmento y ponerles un nombre
    #  de acuerdo a sus coordenadas

    red_vial_gdf['source_Point'] = (
        red_vial_gdf['split_lines']
        .apply(lambda x: Point(x.coords[0]))
    )

    red_vial_gdf['source_Point_X'] = (
        red_vial_gdf['source_Point']
        .apply(lambda x: x.x)
    )

    red_vial_gdf['source_Point_Y'] = (
        red_vial_gdf['source_Point']
        .apply(lambda x: x.y)
    )

    red_vial_gdf['source'] = (
        red_vial_gdf['source_Point_X'].astype(str)
        .str.replace('.', '_')
        + '-'
        + red_vial_gdf['source_Point_Y'].astype(str)
        .str.replace('.', '_')
    )

    red_vial_gdf['target_Point'] = (
        red_vial_gdf['split_lines']
        .apply(lambda x: Point(x.coords[1]))
    )

    red_vial_gdf['target_Point_X'] = (
        red_vial_gdf['target_Point']
        .apply(lambda x: x.x)
    )

    red_vial_gdf['target_Point_Y'] = (
        red_vial_gdf['target_Point']
        .apply(lambda x: x.y)
    )

    red_vial_gdf['target'] = (
        red_vial_gdf['target_Point_X'].astype(str)
        .str.replace('.', '_')
        + '-'
        + red_vial_gdf['target_Point_Y'].astype(str)
        .str.replace('.', '_')
    )

    # Obtener los nodos de inicio y fin de los segmentos en un GeoDataFrame
    #  aparte

    red_vial_gdf_nodes = pd.concat([
        red_vial_gdf
        .loc[:, ['source_Point_X', 'source_Point_Y', 'source_Point', 'source']]
        .rename(
            columns={
                'source_Point_X': 'node_Point_X',
                'source_Point_Y': 'node_Point_Y',
                'source_Point': 'geometry',
                'source': 'node',
            }
        ),
        red_vial_gdf
        .loc[:, ['target_Point_X', 'target_Point_Y', 'target_Point', 'target']]
        .rename(
            columns={
                'target_Point_X': 'node_Point_X',
                'target_Point_Y': 'node_Point_Y',
                'target_Point': 'geometry',
                'target': 'node',
            }
        ),
    ])

    red_vial_gdf_nodes = (
        gpd.GeoDataFrame(
            red_vial_gdf_nodes,
            geometry='geometry',
            crs=red_vial_gdf.crs,
        )
    )

    red_vial_gdf_nodes = red_vial_gdf_nodes.drop_duplicates(subset=['node'])

    red_vial_gdf_nodes = (
        red_vial_gdf_nodes
        .rename(
            columns={'node_Point_X': 'CoordX', 'node_Point_Y': 'CoordY'}
        )
    )

    return red_vial_gdf, red_vial_gdf_nodes


# Función para cargar los datos y almacenarlos en caché
# para mejorar el rendimiento
@st.cache_resource
def cargar_datos_redvial_200k():
    red_vial_gdf, red_vial_gdf_nodes = (
        red_vial_red_vial_nodos()
    )

    # Crear un grafo con los segmentos de línea

    G = nx.from_pandas_edgelist(
        red_vial_gdf,
        edge_attr=["weight"],
    )

    return red_vial_gdf_nodes, G


@st.cache_data
def cargar_datos_edificaciones_y_construcciones_200k():
    _, red_vial_gdf_nodes = red_vial_red_vial_nodos()

    limiteprovincial_gdf = (
        gpd.read_file(archivo_limiteprovincial_5k)
    )

    # Asegurarse que el CRS sea CRTM05 (EPSG 5367)
    limiteprovincial_gdf = (
        limiteprovincial_gdf.to_crs(epsg=5367)
    )

    # Reducción de columnas
    limiteprovincial_gdf = (
        limiteprovincial_gdf[['PROVINCIA', 'geometry']]
    )

    edificaciones_y_construcciones_gdf = (
        gpd.read_file(archivo_edificaciones_y_construcciones_200k)
    )

    # Asegurarse que el CRS sea CRTM05 (EPSG 5367)
    edificaciones_y_construcciones_gdf = (
        edificaciones_y_construcciones_gdf.to_crs(epsg=5367)
    )

    # Reducción de columnas
    edificaciones_y_construcciones_gdf = (
        edificaciones_y_construcciones_gdf[['categoria', 'nombre', 'geometry']]
    )

    # Unión espacial
    edificaciones_y_construcciones_gdf = (
        edificaciones_y_construcciones_gdf.sjoin(
            limiteprovincial_gdf,
            predicate='intersects'
        )
    )

    # Determinar las edificaciones y construcciones más cercanas a cada nodo
    #  del grafo, pero más cercanos que 1000 m

    edificaciones_y_construcciones_gdf = (
        gdf_closest(edificaciones_y_construcciones_gdf, red_vial_gdf_nodes)
    )

    edificaciones_y_construcciones_gdf = (
        edificaciones_y_construcciones_gdf
        .loc[edificaciones_y_construcciones_gdf['distance'] < 1000].copy()
    )

    return edificaciones_y_construcciones_gdf


@st.cache_data
def cargar_datos_limiteprovincial_5k():
    limiteprovincial_gdf = (
        gpd.read_file(archivo_limiteprovincial_5k)
    )

    # Asegurarse que el CRS sea CRTM05 (EPSG 5367)
    limiteprovincial_gdf = (
        limiteprovincial_gdf.to_crs(epsg=5367)
    )

    # Reducción de columnas
    limiteprovincial_gdf = (
        limiteprovincial_gdf[['PROVINCIA', 'geometry']]
    )

    return limiteprovincial_gdf


# ----- Funciones para grafos -----
def create_list_LineStrings(geom):
    """Separar la línea en segmentos
    """
    return list(map(LineString, zip(geom.coords[:-1], geom.coords[1:])))


def split_LineString(curve):
    """Wrapper alrededor de create_list_LineStrings para tomar en cuenta
     MultiLineString u otros tipos
    """
    if type(curve) is LineString:
        return create_list_LineStrings(curve)
    elif type(curve) is MultiLineString:
        lista_return = []
        for geom in curve.geoms:
            lista_return += create_list_LineStrings(geom)
        return lista_return
    else:
        raise Exception('curve is ' + str(type(curve)))


def gdf_closest(gdf1, gdf2):
    """Encontrar los puntos más cercanos de un GeoDataFrame a otro
     GeoDataFrame mediante cKDTree
    """
    serie1 = gdf1['geometry']
    serie2 = gdf2['geometry']

    arr_1 = np.array(list(serie1.apply(lambda x: (x.x, x.y))))
    arr_2 = np.array(list(serie2.apply(lambda x: (x.x, x.y))))

    cKDTree_2 = cKDTree(arr_2)

    distance, index = cKDTree_2.query(arr_1, k=1)

    gdf2_closest = (
        gdf2.iloc[index]
        .reset_index(drop=True)
        .rename(columns={'geometry': 'geometry_closest'})
    )

    gdf1 = gdf1.reset_index(drop=True)

    gdf1_final = gdf1.join(gdf2_closest)

    gdf1_final['distance'] = distance

    return gdf1_final


# ----- Conversión de coordenadas -----
source_crs = 'EPSG:4326'
target_crs = 'EPSG:5367'

latlon_to_CR = pyproj.Transformer.from_crs(source_crs, target_crs)


# Título de la aplicación
st.title(
    'Datos de la red vial, edificaciones y provincias del SNIT para análisis'
    ' con grafos'
)

# ----- Carga de datos -----

# Mostrar un mensaje mientras se cargan los datos de redvial_200k
estado_carga_redvial_200k = st.text('Cargando datos de redvial_200k...')
# Cargar los datos
red_vial_gdf_nodes, G = (
    cargar_datos_redvial_200k()
)
# Actualizar el mensaje una vez que los datos han sido cargados
estado_carga_redvial_200k.text('Los datos de redvial_200k fueron cargados.')

# Cargar datos geoespaciales de edificaciones_y_construcciones_200k
estado_carga_edificaciones_y_construcciones_200k = (
    st.text('Cargando datos de edificaciones_y_construcciones_200k...')
)
edificaciones_y_construcciones_gdf = (
    cargar_datos_edificaciones_y_construcciones_200k()
)
(
    estado_carga_edificaciones_y_construcciones_200k
    .text(
        'Los datos de edificaciones_y_construcciones_200k fueron cargados.'
    )
)

# Mostrar un mensaje mientras se cargan los datos de limiteprovincial_5k
estado_carga_limiteprovincial_5k = (
    st.text('Cargando datos de limiteprovincial_5k...')
)
# Cargar los datos
limiteprovincial_gdf = cargar_datos_limiteprovincial_5k()
# Actualizar el mensaje una vez que los datos han sido cargados
(
    estado_carga_limiteprovincial_5k
    .text(
        'Los datos de limiteprovincial_5k fueron cargados.'
    )
)

# ----- Lista de selección en la barra lateral -----

# Obtener la lista de provincias únicas
lista_provincias = (
    edificaciones_y_construcciones_gdf['PROVINCIA'].unique().tolist()
)

lista_provincias.sort()

# Añadir la opción "Todos" al inicio de la lista
opciones_provincias = ['Todas'] + lista_provincias

# Crear el selectbox en la barra lateral
provincia_seleccionada = st.sidebar.selectbox(
    'Selecciona una provincia',
    opciones_provincias
)

# ----- Filtrar datos según la selección de provincia -----

if provincia_seleccionada != 'Todas':
    # Filtrar los datos para la provincia seleccionada
    edif_filtrados = (
        edificaciones_y_construcciones_gdf[
            edificaciones_y_construcciones_gdf['PROVINCIA'] ==
            provincia_seleccionada
        ]
    )
else:
    # No aplicar filtro
    edif_filtrados = edificaciones_y_construcciones_gdf.copy()

# ----- Selección de las categorías -----

lista_categorias = (
    edif_filtrados['categoria'].dropna().unique().tolist()
)

lista_categorias.sort()

categorias_multiselect = (
    st.sidebar.multiselect("Seleccione las categorías", lista_categorias)
)

# ----- Filtrar datos según la selección de categorías -----

if len(categorias_multiselect) != 0:
    # Filtrar los datos para las categorías seleccionadas
    edif_filtrados = (
        edif_filtrados[
            edif_filtrados['categoria'].isin(categorias_multiselect)
        ]
    )

# ----- Tabla de edificaciones y construcciones por provincia -----

# Mostrar la tabla
st.subheader('Edificaciones y construcciones por provincia')
st.dataframe(edif_filtrados, hide_index=True)

# ----- Gráfico de pastel de categorías de edificaciones y construcciones por
# provincia -----

# Cálculo del conteo por categoría
edif_filtrados_conteo = (
    edif_filtrados
    .groupby('categoria')['geometry']
    .count()
    .sort_values(ascending=True)
    .reset_index()
)

# Creación del gráfico de pastel
fig = px.pie(
    edif_filtrados_conteo,
    names='categoria',
    values='geometry',
    title='Distribución de categorías de edificios o construcciones',
    labels={'categoria': 'Categoría', 'geometry': 'Cantidad'}
)

# Atributos globales de la figura
fig.update_layout(
    legend_title_text='Categoría'
)

# Atributos de las propiedades visuales
fig.update_traces(textposition='inside', textinfo='percent')

# Mostrar el gráfico
st.subheader(
    'Distribución de categorías de edificaciones y construcciones por'
    ' provincia'
)
st.plotly_chart(fig)

# ----- Mapa con folium -----

# Crear el mapa interactivo con las edificaciones y construcciones al mapa
mapa = edif_filtrados.explore(
    name='Edificaciones y Construcciones',
    marker_type='circle',
    marker_kwds={'radius': 20, 'color': 'red'},
    tooltip=['categoria', 'nombre'],
    popup=True
)

# Agregar el control de capas al mapa
folium.LayerControl().add_to(mapa)

# Mostrar el mapa
st.subheader('Mapa de edificaciones y construcciones por provincia')
st_folium(mapa)

# ----- Ruta más corta con el grafo -----

# Obtener la lista de categorías de edificaciones y construcciones
lista_categorias = (
    edif_filtrados['categoria'].dropna().unique().tolist()
)

lista_categorias.sort()

# Seleccionar categoría de edificaciones y construcciones
categoria_seleccionada = st.selectbox(
    'Seleccione una categoría',
    lista_categorias
)

st.title("Haga clic en el mapa para seleccionar una ubicación")

# Inicializar estado
if "marker_location" not in st.session_state:
    st.session_state.marker_location = (
        dict_ubicaciones_iniciales[provincia_seleccionada]
    )

# Crear mapa centrado en la última ubicación
m = folium.Map(location=st.session_state.marker_location, zoom_start=20)

# Agregar marcador en la posición actual
folium.Marker(
    location=st.session_state.marker_location,
    popup="Ubicación actual",
).add_to(m)

# Mostrar mapa y capturar clics
map_data = st_folium(m, height=500, width=700)

# Si el usuario hace clic, actualizar posición
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.session_state.marker_location = [lat, lon]

# Mostrar coordenadas
lat, lon = st.session_state.marker_location
CoordY, CoordX = latlon_to_CR.transform(lat, lon)
st.write(f"Posición actual del marcador: X={CoordX:.2f}, Y={CoordY:.2f}")

# Inicializar estado
if "ruta_calculada" not in st.session_state:
    st.session_state.ruta_calculada = None

if st.button("Determinar ruta:"):
    list_nodes = (
        edif_filtrados
        .loc[
            (
                edif_filtrados['categoria'] ==
                categoria_seleccionada
            ),
            'node'
        ]
        .tolist()
    )

    # Se busca el nodo más cercano a ese punto y se usa como punto de inicio
    gdf_closest_source = (
        gdf_closest(
            gpd.GeoDataFrame(geometry=[Point(CoordX, CoordY)], crs=target_crs),
            red_vial_gdf_nodes,
        )
    )

    node_nearest = gdf_closest_source.loc[gdf_closest_source.index[0], 'node']

    source = node_nearest

    # Se determinan todas las rutas de cada punto a la edificación de la
    #  edificación y construcción más cercana
    lengths, paths = nx.multi_source_dijkstra(G, sources=list_nodes)

    st.session_state.ruta_calculada = {
        "distancia": lengths[source],
        "nodos": paths[source],
    }

# ---- Mostrar ruta si existe ----
if st.session_state.ruta_calculada:
    st.write(
        "La distancia de la ruta más corta es: "
        f"{st.session_state.ruta_calculada['distancia']:.2f} metros"
    )

    # Obtener nodos de la ruta
    nodos_ruta = st.session_state.ruta_calculada['nodos']
    df_ruta = (
        pd.DataFrame
        .from_dict(
            {
                'node': nodos_ruta,
                'orden': [len(nodos_ruta)-i-1 for i in range(len(nodos_ruta))],
            }
        )
    )

    # Filtrar ruta
    red_vial_gdf_nodes_ruta = (
        red_vial_gdf_nodes
        .merge(df_ruta, on='node')
    )

    # ----- Mapa con folium -----
    mapa_ruta = red_vial_gdf_nodes_ruta.explore(
        name='Ruta',
        tooltip=['orden'],
        popup=True
    )

    edificaciones_y_construcciones_gdf.explore(
        m=mapa_ruta,
        name='Edificaciones y Construcciones',
        marker_kwds={'color': 'red'},
        tooltip=['categoria', 'nombre'],
        popup=True
    )

    # Agregar un control de capas al mapa
    folium.LayerControl().add_to(mapa_ruta)

    # Mostrar el mapa
    st.subheader('Mapa de la ruta más corta')
    st_folium(mapa_ruta, height=500, width=700)
