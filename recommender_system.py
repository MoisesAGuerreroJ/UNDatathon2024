import numpy as np
import pandas as pd
import geopandas as gpd

from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

df_atractivos = pd.read_csv('data/atractivos_turisticos.csv', 
                           header=0,
                           sep=',',
                           encoding='latin-1')
df_atractivos = gpd.GeoDataFrame(df_atractivos, 
                                 geometry=gpd.points_from_xy(df_atractivos['longitud'], df_atractivos['latitud']))
df_atractivos.head()

# Extract latitudes and longitudes
locations = df_atractivos[['longitud', 'latitud']].values

# Generate Voronoi diagram
vor = Voronoi(locations)

# Limit number of clusters
LIMIT_NUM_CLUSTERS = 7  # Set desired number of clusters

# Randomly select a limited number of Voronoi vertices as initial centers
num_vertices = min(LIMIT_NUM_CLUSTERS, len(vor.vertices))
selected_centers = vor.vertices[np.random.choice(len(vor.vertices), num_vertices, replace=False)]

def apply_clustering(df, centers):
    # cluster_centers = {f"Cluster_{i}": centers[i,:] for i in range(len(centers))}
    df['cluster'] = df.apply(lambda x: [np.linalg.norm(np.array([x.longitud, x.latitud])-center) for center in centers], axis=1)
    df['cluster'] = df['cluster'].apply(lambda x:np.argmin(x))
    return df

df_atractivos = apply_clustering(df_atractivos, selected_centers)
print(df_atractivos.head())

# FEATURES

df_ind_pob = pd.read_csv('data/indice_pobreza_multidimen.csv', encoding='latin-1')
df_desemp = pd.read_csv('data/tasa_de_desempleo.csv', encoding='latin-1')
dfx1 = df_atractivos.merge(df_ind_pob, left_on='nombre_comuna', right_on='nombre', how='inner')[['nombre_sitio','nombre_comuna','latitud','longitud','geometry','cluster','i_2023']].rename(columns={'i_2023':'indice_pobreza'})
dfx2 = dfx1.merge(df_desemp, left_on='nombre_comuna', right_on='nombre', how='inner')[['nombre_sitio','nombre_comuna','latitud','longitud','geometry','cluster','indice_pobreza','i_2022']].rename(columns={'i_2022':'tasa_desempleo'})

# RECOMMENDER SYSTEM

def tourist_attraction_recommendation_algorithm(df, user_location, historical_visits):
    # Paso 1: Calcular puntaje de recomendación para cada POI en función de `indice_pobreza` y `tasa_desempleo`
    weight_poverty = 0.3  # Mayor peso para `indice_pobreza`
    weight_unemployment = 0.7  # Menor peso para `tasa_desempleo`
    df['recommendation_score'] = (weight_poverty * df['indice_pobreza']) + (weight_unemployment * df['tasa_desempleo'])

    # Paso 2: Identificar el clúster del punto de inicio del usuario
    # Asumimos que `user_location` contiene `nombre` del POI inicial para identificar su clúster
    user_cluster = df.loc[df['nombre_sitio'] == user_location['nombre_sitio'], 'cluster'].values[0]
    
    # Filtrar puntos de interés que estén en el mismo clúster que el usuario
    #df = df[df['cluster'] == user_cluster]
    df = df[df['cluster'] == user_cluster].copy()

    # Paso 3: Calcular proximidad al punto de inicio del usuario (user_location)
    df['distance_to_user'] = cdist([(user_location['latitud'], user_location['longitud'])],
                                   df[['latitud', 'longitud']], metric='euclidean').flatten()
    
    # Paso 4: Calcular popularidad basada en visitas históricas
    visit_counts = historical_visits['poi'].value_counts().rename("popularity_score")
    df = df.merge(visit_counts, how='left', left_on='nombre_sitio', right_index=True)
    df['popularity_score'] = df['popularity_score'].fillna(0)  # Rellenar con 0 donde no hay visitas

    # Paso 5: Calcular puntaje final ponderado basado en pobreza, desempleo, distancia y popularidad
    weight_recommendation_score = 0.6
    weight_distance = 0.3
    weight_popularity = 0.1
    
    df['final_score'] = (
        weight_recommendation_score * df['recommendation_score'] -
        weight_distance * df['distance_to_user'] +  # Penalizamos por distancia para priorizar lugares cercanos
        weight_popularity * df['popularity_score']
    )

    # Paso 6: Ordenar puntos de interés por el puntaje final
    sorted_pois = df[df['nombre_sitio'] != user_location['nombre_sitio']].sort_values(by='final_score', ascending=False)

    # Retornar la lista de puntos de interés recomendados
    recommended_pois = sorted_pois[['nombre_sitio', 'distance_to_user', 'popularity_score', 'final_score', 'cluster']]
    return recommended_pois

historical_visits = pd.read_csv('data/fake_historical_user_data.csv', header=0, sep=',')

# Ubicación inicial del usuario en 'Museo de Antioquia'

print(f"Selecciona un lugar de origen: ")
for _, i in df_atractivos[['OBJECTID','nombre_sitio']].iterrows():
    print(f" [{i['OBJECTID']}] - {i['nombre_sitio']}")

objectId = int(input("Ubicación origen: "))

user_location = {
    'nombre_sitio': df_atractivos[df_atractivos.OBJECTID == objectId]['nombre_sitio'].iloc[0],
    'latitud': df_atractivos[df_atractivos.OBJECTID == objectId]['latitud'].iloc[0],
    'longitud': df_atractivos[df_atractivos.OBJECTID == objectId]['longitud'].iloc[0]
}

print(user_location)

# Generar recomendaciones
recommended_pois = tourist_attraction_recommendation_algorithm(dfx2, user_location, historical_visits)

# Mostrar los puntos de interés recomendados
print("Puntos de interés recomendados basados en ubicación, popularidad, clúster y características socioeconómicas:")
print(recommended_pois)