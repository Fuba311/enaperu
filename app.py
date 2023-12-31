import json
import pandas as pd
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
from dash import dash_table
from collections import defaultdict
import numpy as np
from flask_caching import Cache
import geopandas as gpd
import os

# Función para convertir de forma segura a float
def safe_convert_to_float(value):
    try:
        # Convierte a float y trata los NaN como ceros
        return float(value) if not pd.isna(value) else 0.0
    except ValueError:
        return 0.0

# Use the raw URL of the .dta file
raw_url = 'https://github.com/Fuba311/enaperu/raw/main/02_Cap200ab.parquet'
raw_url2 = 'https://github.com/Fuba311/enaperu/raw/main/16_Cap900.parquet'
raw_url3 = 'https://github.com/Fuba311/enaperu/raw/main/15_Cap800.parquet'
raw_url4 = 'https://github.com/Fuba311/enaperu/raw/main/14_Cap700.parquet'
raw_url5 = 'https://github.com/Fuba311/enaperu/raw/main/25_Cap1200d.parquet'
raw_url6 = 'https://github.com/Fuba311/enaperu/raw/main/04_Cap200b_1.parquet'
raw_url7 = 'https://github.com/Fuba311/enaperu/raw/main/DEPARTAMENTOS_inei_geogpsperu_suyopomalia.shp'
raw_url8 = 'https://github.com/Fuba311/enaperu/raw/main/17_Cap1000.parquet'

# Read the .dta file directly into a pandas dataframe
df = pd.read_parquet(raw_url)
df_cap900 = pd.read_parquet(raw_url2)
df_cap800 = pd.read_parquet(raw_url3)

# Group by CONGLOMERADO, NSELUA, and UA, and sum the P217_SUP_ha
grouped_df = df.groupby(['CONGLOMERADO', 'NSELUA', 'UA'])['P217_SUP_ha'].sum().reset_index()
grouped_df.rename(columns={'P217_SUP_ha': 'Total_Ha'}, inplace=True)

# Merge the summed data back into the original DataFrame
df = pd.merge(df, grouped_df, on=['CONGLOMERADO', 'NSELUA', 'UA'], how='left')

# Select only the relevant columns for the merge from df
relevant_columns_df = df[['CONGLOMERADO', 'NSELUA', 'UA', 'Total_Ha']].drop_duplicates()

# Merge these columns into df_cap900
df_cap900 = pd.merge(df_cap900, relevant_columns_df, on=['CONGLOMERADO', 'NSELUA', 'UA'], how='left')


# Merge these columns into df_cap800
df_cap800 = pd.merge(df_cap800, relevant_columns_df, on=['CONGLOMERADO', 'NSELUA', 'UA'], how='left')

# Step 1: Create a new DataFrame with dummy columns for each 'cultivo'
cultivo_dummies = pd.get_dummies(df['P204_NOM'])
df_with_dummies = pd.concat([df[['CONGLOMERADO', 'NSELUA', 'UA']], cultivo_dummies], axis=1)

# Step 2: Sum the dummy columns for each unique combination of 'CONGLOMERADO', 'NSELUA', and 'UA'
df_grouped = df_with_dummies.groupby(['CONGLOMERADO', 'NSELUA', 'UA']).sum().reset_index()

# Now 'df_grouped' has one row per unique combination with 1s and 0s indicating the presence of each 'cultivo'

# Step 3: Merge the new DataFrame with 'df_cap800'
df_cap800 = pd.merge(df_cap800, df_grouped, on=['CONGLOMERADO', 'NSELUA', 'UA'], how='left')

# Load additional DataFrame
df_cap700 = pd.read_parquet(raw_url4)

# Perform the merge
relevant_columns_df = df[['CONGLOMERADO', 'NSELUA', 'UA', 'Total_Ha']].drop_duplicates()
df_cap700 = pd.merge(df_cap700, relevant_columns_df, on=['CONGLOMERADO', 'NSELUA', 'UA'], how='left')
gdf = gpd.read_file(raw_url7)
gdf = gdf.to_crs(epsg=4326)  # Ensure the GeoDataFrame is in WGS 84 coordinate system
geojson = json.loads(gdf.to_json())

# Load new data
df_cap1200d = pd.read_parquet(raw_url5)
relevant_columns_df = df[['CONGLOMERADO', 'NSELUA', 'UA', 'Total_Ha', 'FACTOR']].drop_duplicates()
# Outer merge with df_cap1200d
merged_df2 = pd.merge(df_cap1200d, relevant_columns_df, on=['CONGLOMERADO', 'NSELUA', 'UA', 'FACTOR'], how='outer')
#merged_df2.to_stata('C:\\UC\\RIMISP\\Encuestas Perú\\2019\\2022\\1760 - Características unidad agropecuaria en últimos 12 meses - maquinaria y equipo\\test.dta')

# Load additional data
df_prob = pd.read_parquet(raw_url6)

# Rename 'P224B_NOM' to 'P204_NOM' in df_prob for merging
df_prob.rename(columns={'P224B_NOM': 'P204_NOM'}, inplace=True)

# Merge df with df_prob
df3 = pd.merge(df, df_prob, on=['CONGLOMERADO', 'NSELUA', 'UA', 'P204_NOM', 'FACTOR'], how='outer')

# Retrieve specified columns and fill NaNs with 0
loss_columns = ['P224E_1', 'P224E_2', 'P224E_3', 'P224E_4', 'P224E_5', 'P224E_6', 'P224E_7']
df3[loss_columns] = df3[loss_columns].fillna('Pase')

# Load additional DataFrame
df_cap1000 = pd.read_parquet(raw_url8)

# Perform the merge
relevant_columns_df = df[['CONGLOMERADO', 'NSELUA', 'UA', 'Total_Ha']].drop_duplicates()
df_cap1000 = pd.merge(df_cap1000, relevant_columns_df, on=['CONGLOMERADO', 'NSELUA', 'UA'], how='left')


# Hectare Ranges
hectare_ranges = [
    (0, 0.5),
    (0.5, 1),
    (1, 2.5),
    (2.5, 5),
    (5, 10),
    (10, 20),
    (20, 50),
    (50, 100),
    (100, float('inf'))
]


def calcular_kilos(row):
    P219_EQUIV_KG = safe_convert_to_float(row['P219_EQUIV_KG']) * row['FACTOR']
    total = safe_convert_to_float(row['P219_CANT_1']) * P219_EQUIV_KG
    venta = (safe_convert_to_float(row['P220_1_CANT_1']) + safe_convert_to_float(
        row['P220_1_CANT_2']) / 1000) * P219_EQUIV_KG
    consumo = (safe_convert_to_float(row['P220_2_ENT']) + safe_convert_to_float(
        row['P220_2_DEC']) / 1000) * P219_EQUIV_KG
    semilla_autoconsumo = (safe_convert_to_float(row['P220_3A_ENT']) + safe_convert_to_float(
        row['P220_3A_DEC']) / 1000) * P219_EQUIV_KG
    semilla_venta = (safe_convert_to_float(row['P220_3B_ENT']) + safe_convert_to_float(
        row['P220_3B_DEC']) / 1000) * P219_EQUIV_KG
    trueque = (safe_convert_to_float(row['P220_4_ENT']) + safe_convert_to_float(
        row['P220_4_DEC']) / 1000) * P219_EQUIV_KG
    alimento_animales = (safe_convert_to_float(row['P220_5_ENT']) + safe_convert_to_float(
        row['P220_5_DEC']) / 1000) * P219_EQUIV_KG
    derivados = (safe_convert_to_float(row['P220_6_ENT']) + safe_convert_to_float(
        row['P220_6_DEC']) / 1000) * P219_EQUIV_KG
    pago_especie = (safe_convert_to_float(row['P220_7_ENT']) + safe_convert_to_float(
        row['P220_7_DEC']) / 1000) * P219_EQUIV_KG
    donaciones = (safe_convert_to_float(row['P220_8_ENT']) + safe_convert_to_float(
        row['P220_8_DEC']) / 1000) * P219_EQUIV_KG
    robos = (safe_convert_to_float(row['P220_9_ENT']) + safe_convert_to_float(row['P220_9_DEC']) / 1000) * P219_EQUIV_KG
    otros = (safe_convert_to_float(row['P220_10_ENT']) + safe_convert_to_float(
        row['P220_10_DEC']) / 1000) * P219_EQUIV_KG
    otros2 = sum([total, venta, consumo, semilla_autoconsumo, semilla_venta, trueque, alimento_animales,
                  derivados, pago_especie, donaciones, robos]) - (venta + consumo)
    return pd.Series([total, venta, consumo, semilla_autoconsumo, semilla_venta, trueque, alimento_animales,
                      derivados, pago_especie, donaciones, robos, otros, otros2])


df[['total_kg', 'venta_kg', 'consumo_kg', 'semilla_autoconsumo_kg', 'semilla_venta_kg',
    'trueque_kg', 'alimento_animales_kg', 'derivados_kg', 'pago_especie_kg', 'donaciones_kg',
    'robos_kg', 'otros_kg', 'otros_kg1']] = df.apply(calcular_kilos, axis=1)

df['hectareas_total'] = df['P217_SUP_ha'] * df['FACTOR']

def calcular_kilos2(row):
    P219_EQUIV_KG = safe_convert_to_float(row['P219_EQUIV_KG']) * row['FACTOR']
    total = safe_convert_to_float(row['P219_CANT_1']) * P219_EQUIV_KG
    venta = (safe_convert_to_float(row['P220_1_CANT_1']) + safe_convert_to_float(
        row['P220_1_CANT_2']) / 1000) * P219_EQUIV_KG

    return pd.Series([total, venta])

# Applying the function to the DataFrame
df[['total_kg2', 'venta_kg2']] = df.apply(calcular_kilos2, axis=1)


def calcular_equivalencia_kg(row, col_cant, col_equiv_kg, col_factor):
    cantidad = safe_convert_to_float(row[col_cant])
    equiv_kg = safe_convert_to_float(row[col_equiv_kg])
    factor = safe_convert_to_float(row[col_factor])
    return (cantidad * equiv_kg * factor) / 1000


# Agregar la nueva columna al DataFrame
df['equivalencia_kg'] = df.apply(lambda row: calcular_equivalencia_kg(row, 'P220_1_CANT_1', 'P219_EQUIV_KG', 'FACTOR'),
                                 axis=1)

# List of columns to be converted to categorical type
categorical_columns = [
    'P222_1', 'P222_2', 'P222_3', 'P222_4', 'P222_5', 'P222_6', 'P222_7',
    'P221_1', 'P221_2', 'P223_1', 'P223_2', 'P223_3', 'P223_4', 'P223_5', 'P223_6'
]

# Convert each specified column to categorical
for col in categorical_columns:
    df[col] = df[col].astype('category')


def calcular_ventas_por_entidad(df):
    categorias = {
        'P222_1': 'Acopiador',
        'P222_2': 'Comerciante mayorista',
        'P222_3': 'Comerciante minorista',
        'P222_4': 'Asociación / cooperativa',
        'P222_5': 'Empresa / agroindustria',
        'P222_6': 'Consumidor Final',
        'P222_7': 'Otro'
    }

    ventas = {}
    for col, categoria in categorias.items():
        # Filtrar las filas donde la columna contiene el texto de la categoría
        filtered_df = df[df[col] == categoria]
        # Sumar los valores de 'equivalencia_kg' para las filas filtradas
        total_kg = filtered_df['equivalencia_kg'].sum()
        # Agregar el total a nuestro diccionario de ventas
        ventas[categoria] = total_kg
        # Imprimir para depuración

    return ventas

def calcular_destino_produccion(df, col_filter=None):
    categorias_destinos = {
        'P223_1': 'Mercado Local',
        'P223_2': 'Mercado Regional',
        'P223_3': 'Mercado Exterior',
        'P223_4': 'Agroindustria',
        'P223_5': 'Mercados de Lima',
        'P223_6': 'No Sabe'
    }

    # Initialize totals with zeros for all categories including the new ones
    totales_destinos = {categoria: 0 for categoria in categorias_destinos.values()}
    totales_destinos['Mercado Local y Regional'] = 0
    totales_destinos['Local, Regional y Lima'] = 0
    totales_destinos['Mercado Regional y Agroindustria'] = 0

    for col, categoria in categorias_destinos.items():
        filter_condition = (df[col] != 'nan') & (df[col] != '0.0')

        # Check if both 'Mercado Local' and 'Mercado Regional' are present and exclude such rows
        if categoria in ['Mercado Local', 'Mercado Regional']:
            filter_condition &= ~((df['P223_1'] != 'nan') & (df['P223_1'] != '0.0') &
                                  (df['P223_2'] != 'nan') & (df['P223_2'] != '0.0'))

        # Check if both 'Mercado Local' and 'Mercado Regional' are present and exclude such rows
        if categoria in ['Mercados de Lima']:
            filter_condition &= ~((df['P223_1'] != 'nan') & (df['P223_1'] != '0.0') &
                                  (df['P223_2'] != 'nan') & (df['P223_2'] != '0.0') &
                                  (df['P223_5'] != 'nan') & (df['P223_5'] != '0.0'))

        # Check if both 'Mercado Regional' and 'Agroindustria' are present and exclude such rows
        if categoria in ['Mercado Regional', 'Agroindustria']:
            filter_condition &= ~((df['P223_2'] != 'nan') & (df['P223_2'] != '0.0') &
                                  (df['P223_4'] != 'nan') & (df['P223_4'] != '0.0'))

        if col_filter:
            filter_condition &= (df['P204_NOM'] == col_filter)

        totales_destinos[categoria] = df[filter_condition]['equivalencia_kg'].sum()

    # Calculate 'Todo el país'
    todo_el_pais_filter = (
            (df['P223_1'] != 'nan') & (df['P223_1'] != '0.0') &
            (df['P223_2'] != 'nan') & (df['P223_2'] != '0.0') &
            (df['P223_5'] != 'nan') & (df['P223_5'] != '0.0')
    )

    if col_filter:
        todo_el_pais_filter &= (df['P204_NOM'] == col_filter)

    totales_destinos['Local, Regional y Lima'] = df[todo_el_pais_filter]['equivalencia_kg'].sum()

    # Calculate 'Mercado Local y Regional'
    mercado_local_y_regional_filter = (
            (df['P223_1'] != 'nan') & (df['P223_1'] != '0.0') &
            (df['P223_2'] != 'nan') & (df['P223_2'] != '0.0')
    )
    if col_filter:
        mercado_local_y_regional_filter &= (df['P204_NOM'] == col_filter)

    # Exclude rows that meet 'Todo el país' conditions
    mercado_local_y_regional_filter &= ~todo_el_pais_filter

    totales_destinos['Mercado Local y Regional'] = df[mercado_local_y_regional_filter]['equivalencia_kg'].sum()

    # Calculate 'Mercado Regional y Agroindustria'
    regional_agroindustria_filter = (
            (df['P223_2'] != 'nan') & (df['P223_2'] != '0.0') &
            (df['P223_4'] != 'nan') & (df['P223_4'] != '0.0')
    )

    if col_filter:
        regional_agroindustria_filter &= (df['P204_NOM'] == col_filter)

    totales_destinos['Mercado Regional y Agroindustria'] = df[regional_agroindustria_filter]['equivalencia_kg'].sum()

    # Return the totals as a list in the specific order
    ordered_totals = [totales_destinos[category] for category in
                      list(categorias_destinos.values()) + ['Mercado Local y Regional', 'Local, Regional y Lima', 'Mercado Regional y Agroindustria']]

    return ordered_totals

def calcular_ventas_chacra(df):
    venta_dentro_chacra_ton = df[df['P221_1'] == 'Dentro de la chacra']['equivalencia_kg'].sum()
    venta_fuera_chacra_ton = df[df['P221_2'] == 'Fuera de la chacra']['equivalencia_kg'].sum()
    return venta_dentro_chacra_ton, venta_fuera_chacra_ton



# Inicializar la aplicación Dash con un tema de Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Visualizador Interactivo ENA 2022 Perú",
                        style={
                            'textAlign': 'center',
                            'fontFamily': '"Century Gothic", Arial, sans-serif',  # Gothic-style font
                            'color': '#007bff',
                            'fontWeight': 'bold'
                        }),
                width=12, className="mb-4 mt-4")),

    dbc.Row(dbc.Col(html.P("Por Andrés Fuica (andresfuba@uc.cl)",
                        style={
                            'textAlign': 'center',
                            'fontFamily': '"Century Gothic", Arial, sans-serif',
                            'color': '#6c757d',  # Example color (Bootstrap secondary color)
                            'fontSize': 'small'  # Smaller font size
                        }),
                width=12, className="mb-4")),

    dbc.Row(dbc.Col(html.P("Para todos los gráficos, puede filtrar por Región, Departamento, Provincia, Distrito y, solo para algunos, por Cultivo específico. Todos los gráficos junto con los datos utilizados para generarlos son descargables.",
                      style={
                          'textAlign': 'center',
                          'fontFamily': '"Century Gothic", Arial, sans-serif',
                          'color': '#6c757d',  # Example color (Bootstrap secondary color)
                          'fontSize': 'small'  # Smaller font size
                      }),
                width=12, className="mb-4")),

    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': region, 'value': region} for region in df['REGION'].unique()],
                placeholder="Selecciona una Región"
            ), width=6, className="mb-2 mt-2"  # Adds a bottom margin (mb-2)
        )
    ]),

    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='departamento-dropdown',
                options=[{'label': dep, 'value': dep} for dep in df['NOMBREDD'].unique()],
                placeholder="Selecciona un Departamento"
            ), width=6, className="mb-2 mt-1"  # Agrega un margen inferior (mb-2)
        ),
        dbc.Col(
            dcc.Dropdown(
                id='provincia-dropdown',
                placeholder="Selecciona una Provincia"
            ), width=6, className="mb-2 mt-1"  # Agrega un margen inferior (mb-2)
        )
    ]),

    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='distrito-dropdown',
                placeholder="Selecciona un Distrito"
            ), width=6, className="mb-2 mt-2"  # Agrega un margen inferior (mb-2)
        ),
        dbc.Col(
            dcc.Dropdown(
                id='cultivo-dropdown',
                options=[{'label': cult, 'value': cult} for cult in df['P204_NOM'].unique()],
                placeholder="Selecciona un Cultivo"
            ), width=6, className="mb-2 mt-2"  # Agrega un margen inferior (mb-2)
        )
    ]),

    # Add this inside your dbc.Container in the layout
    dbc.Row([
        dbc.Col(html.Label("Ver por rangos de hectáreas:"), width=3, className="mt-5"),
        dbc.Col(dcc.Checklist(
            options=[{'label': ' ', 'value': 'toggle'}],
            value=[],
            id='chart-toggle'
        ), width=1, className="mt-5")
    ]),

    dbc.Row([dbc.Col(dcc.Graph(id='cantidad-chart'), width=12, className="mb-4 mt-3")]),
    # Agrega un margen inferior (mb-4)
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='graph-selection-dropdown',
                options=[
                    {'label': 'Producción - Uso y Destinos de Venta', 'value': 'prod_uso_venta'},
                    {'label': 'Pérdidas de Cultivos y Jornaleros', 'value': 'cultivo_losses'},
                    {'label': 'Servicios Financieros', 'value': 'servicios_financieros'},
                    {'label': 'Asociatividad', 'value': 'asociatividad'},
                    {'label': 'Uso de insumos', 'value': 'uso_insumos'},
                    {'label': 'Capacitación y Asistencia Técnica', 'value': 'capacitacion_asistencia'},
                    {'label': 'Mapa Coroplético', 'value': 'mapa_coropletico'},
                    {'label': 'Maquinaria y Equipos Agrícolas', 'value': 'maquinaria_equipos'}

                ],
                placeholder="Selecciona un conjunto de gráficos",
                value='prod_uso_venta'
            )
        ], width=12, className="mb-2")
    ]),

    dbc.Collapse(
        [
            dbc.Row([dbc.Col(dcc.Graph(id='cultivo-pie-chart'), width=12, className="mb-4")]),
            dbc.Row([dbc.Col(dcc.Graph(id='selling-vs-non-selling-chart'), width=12, className="mb-4")]),
            dbc.Row([dbc.Col(dcc.Graph(id='ventas-chacra-pie-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='ventas-entidad-bar-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='destino-produccion-bar-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='sales-proportion-table'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='venta-contribution-table'), width=12)], className="mb-1"),
            dbc.Row([dbc.Col(dcc.Graph(id='proportions-category-table'), width=12)], className="mb-4"),
        ],
        id="collapse"
    ),


    # Add the new graph in the corresponding dbc.Collapse section
    dbc.Collapse(
        [
            dbc.Row([dbc.Col(dcc.Graph(id='acceso-credito-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='obtencion-credito-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='entidad-credito-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='razones-rechazo-credito-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='seguro-agricola-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='proveedor-seguro-chart'), width=12)], className="mb-4")
        ],
        id="acceso-credito-collapse"
    ),

    dbc.Collapse(
        [
            dbc.Row([dbc.Col(dcc.Graph(id='asociatividad-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='razones-no-asociatividad-chart'), width=12)], className="mb-4")
        ],
        id="asociatividad-collapse"
    ),

    dbc.Collapse(
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='insumos-selection-dropdown',
                    options=[
                        {'label': 'Abono', 'value': 'P301A_13'},
                        {'label': 'Fertilizante', 'value': 'P301A_14'},
                        {'label': 'Plaguicida', 'value': 'P301A_15'}
                    ],
                    multi=True,
                    placeholder="Selecciona insumos"
                ),
                dcc.Graph(id='insumos-usage-chart')
            ], width=12)
        ]),
        id='insumos-collapse'
    ),

    dbc.Collapse(
        [
            dbc.Row([dbc.Col(dcc.Graph(id='capacitacion-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='asistencia-chart'), width=12)], className="mb-4")
        ],
        id="capacitacion-collapse"
    ),

    dbc.Collapse(
        [
            dbc.Row([dbc.Col(dcc.Graph(id='maquinaria-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='unique-maquinaria-equipos-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='ua-maquinaria-equipos-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='maquinaria-obtencion-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='maquinaria2'), width=12)], className="mb-4"), 
            dbc.Row([dbc.Col(dcc.Graph(id='maquinaria1'), width=12)], className="mb-4"), 
            dbc.Row([dbc.Col(dcc.Graph(id='maquinaria3'), width=12)], className="mb-4"),
            
        ],
        id="maquinaria-collapse"
    ),

    dbc.Collapse(
    dbc.Row([
        dbc.Col(dcc.Graph(id='problemas-cultivo-chart'), width=12, className="mb-4 mt-3"),
        dbc.Col(dcc.Graph(id='crops-affected-reasons-chart'), width=12, className="mb-4 mt-3"),
        dbc.Row([dbc.Col(dcc.Graph(id='jornaleros-chart'), width=12)], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(id='jornaleros-desagregado-chart'), width=12)], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(id='employed-people-chart'), width=12)], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(id='employed-peoples-chart'), width=12)], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(id='worked-days-chart'), width=12)], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(id='dias-contribution-table'), width=12)], className="mb-4"),
    ]),
    id='cultivo-losses-collapse'
    ),

    dbc.Collapse(
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='data-type-dropdown',
                    options=[
                        {'label': 'Hectáreas Cultivadas', 'value': 'hectares'},
                        {'label': 'Producción Total', 'value': 'production'},
                        {'label': 'Ventas Totales', 'value': 'sales'},
                        {'label': 'UAs Asociadas', 'value': 'cooperatives'}  # Add this line
                    ],
                    placeholder="Select Data Type",
                    value='hectares'
                ),
                dcc.Graph(id='choropleth-map')
            ], width=12)
        ]),
        id='mapa-coropletico-collapse'
    ),



], fluid=True)

# Retrieve the connection string from the environment variable
redis_connection_string = os.environ.get('AZURE_REDIS_CONNECTIONSTRING')

# Configure cache
server.config.update({
    'CACHE_TYPE': 'RedisCache',
    'CACHE_REDIS_URL': redis_connection_string  # Use the connection string directly
})

# Initialize cache
cache = Cache(server)


# Callbacks para actualizar menús y gráficos
@app.callback(
    Output('provincia-dropdown', 'options'),
    Input('departamento-dropdown', 'value')
)
def set_provincia_options(selected_departamento):
    return [{'label': i, 'value': i} for i in df[df['NOMBREDD'] == selected_departamento]['NOMBREPV'].unique()]


@app.callback(
    Output('distrito-dropdown', 'options'),
    Input('provincia-dropdown', 'value')
)
def set_distrito_options(selected_provincia):
    return [{'label': i, 'value': i} for i in df[df['NOMBREPV'] == selected_provincia]['NOMBREDI'].unique()]


@app.callback(
    Output('cultivo-pie-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('cultivo-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_chart(selected_departamento, selected_provincia, selected_distrito, selected_cultivo, toggle_state, region):
    # Filtrado condicional basado en las selecciones del usuario
    filtered_df = df.copy()
    if selected_departamento is not None:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == selected_departamento]
    if selected_provincia is not None:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == selected_provincia]
    if selected_distrito is not None:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == selected_distrito]
    if selected_cultivo is not None:
        filtered_df = filtered_df[filtered_df['P204_NOM'] == selected_cultivo]
    if region:  # Filter based on selected 'REGION'
        filtered_df = filtered_df[filtered_df['REGION'] == region]


    # Group by unique production units
    unique_productions = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA'])['Total_Ha'].first().reset_index()

    if 'toggle' in toggle_state:
        hectare_range_data = []
        for lower, upper in hectare_ranges:
            # Select units within the current hectare range
            range_productions = unique_productions[
                (unique_productions['Total_Ha'] >= lower) & (unique_productions['Total_Ha'] < upper)]
            # Merge back to the main DataFrame to get ventas data for selected units
            range_df = pd.merge(range_productions, filtered_df, on=['CONGLOMERADO', 'NSELUA', 'UA'], how='left')
            # Summarize data for the hectare range
            venta_ton = range_df['venta_kg'].sum() / 1000
            consumo_ton = range_df['consumo_kg'].sum() / 1000
            animal_ton = range_df['alimento_animales_kg'].sum() / 1000
            otros_ton = range_df[['semilla_autoconsumo_kg', 'semilla_venta_kg', 'trueque_kg',
                                  'derivados_kg', 'pago_especie_kg', 'donaciones_kg', 'robos_kg', 'otros_kg']].sum(
                axis=1).sum() / 1000

            hectare_range_data.append(
                {'Hectare Range': f'{lower}-{upper}', 'Ventas': venta_ton, 'Consumo': consumo_ton, 'Otros': otros_ton, 'Alimento Animal': animal_ton})

        # Convert to DataFrame
        plot_df = pd.DataFrame(hectare_range_data)
        fig = px.bar(
            plot_df,
            x='Hectare Range',
            y=['Ventas', 'Consumo', 'Otros', 'Alimento Animal'],
            title='Producción por Rango de Hectáreas (Toneladas)',
            labels={'value': 'Toneladas', 'variable': 'Categoría'},
            text_auto=True
        )
        fig.update_layout(barmode='stack')
        return fig

    else:
        # Calcula las toneladas para el gráfico de pie
        total_produccion_ton = filtered_df['total_kg'].sum() / 1000
        categorias = ['Venta', 'Consumo', 'Semilla Autoconsumo', 'Semilla Venta', 'Trueque',
                      'Alimento Animales', 'Derivados', 'Pago en Especie', 'Donaciones', 'Robos', 'Otros']
        valores_ton = [
            filtered_df['venta_kg'].sum() / 1000,
            filtered_df['consumo_kg'].sum() / 1000,
            filtered_df['semilla_autoconsumo_kg'].sum() / 1000,
            filtered_df['semilla_venta_kg'].sum() / 1000,
            filtered_df['trueque_kg'].sum() / 1000,
            filtered_df['alimento_animales_kg'].sum() / 1000,
            filtered_df['derivados_kg'].sum() / 1000,
            filtered_df['pago_especie_kg'].sum() / 1000,
            filtered_df['donaciones_kg'].sum() / 1000,
            filtered_df['robos_kg'].sum() / 1000,
            filtered_df['otros_kg'].sum() / 1000
        ]

        # Preparar datos para el gráfico
        data = {'Categoría': categorias, 'Toneladas': valores_ton}
        df_grafico = pd.DataFrame(data)
        df_grafico['Porcentaje'] = df_grafico['Toneladas'] / total_produccion_ton * 100

        # Crear gráfico
        fig = px.pie(df_grafico, values='Toneladas', names='Categoría', title="Destino de la Producción en Toneladas",
                     hover_data=['Porcentaje'], labels={'Porcentaje': '% del Total'})
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="lightblue",
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="darkblue"
            )
        )

        return fig


# Callbacks para gráficos nuevos
@app.callback(
    Output('ventas-chacra-pie-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('cultivo-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_ventas_chacra_chart(selected_departamento, selected_provincia, selected_distrito, selected_cultivo,
                               toggle_state, region):
    filtered_df = df.copy()
    if selected_departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == selected_departamento]
    if selected_provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == selected_provincia]
    if selected_distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == selected_distrito]
    if selected_cultivo:
        filtered_df = filtered_df[filtered_df['P204_NOM'] == selected_cultivo]
    if region:  # Filter based on selected 'REGION'
        filtered_df = filtered_df[filtered_df['REGION'] == region]


    # Group by unique production units
    unique_productions = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA'])['Total_Ha'].first().reset_index()

    if 'toggle' in toggle_state:
        hectare_range_data = []
        for lower, upper in hectare_ranges:
            # Select units within the current hectare range
            range_productions = unique_productions[
                (unique_productions['Total_Ha'] >= lower) & (unique_productions['Total_Ha'] < upper)]
            # Merge back to the main DataFrame to get ventas data for selected units
            range_df = pd.merge(range_productions, filtered_df, on=['CONGLOMERADO', 'NSELUA', 'UA'], how='left')
            venta_dentro_chacra_ton, venta_fuera_chacra_ton = calcular_ventas_chacra(range_df)
            hectare_range_data.append({
                'Hectare Range': f'{lower}-{upper}',
                'Dentro de la Chacra': venta_dentro_chacra_ton,
                'Fuera de la Chacra': venta_fuera_chacra_ton
            })

        plot_df = pd.DataFrame(hectare_range_data)
        fig = px.bar(
            plot_df,
            x='Hectare Range',
            y=['Dentro de la Chacra', 'Fuera de la Chacra'],
            title="Ventas Dentro y Fuera de la Chacra por Rango de Hectáreas",
            text_auto=True
        )
        fig.update_layout(barmode='stack')
        return fig
    else:
        # Regular pie chart logic (existing code)
        venta_dentro_chacra_ton, venta_fuera_chacra_ton = calcular_ventas_chacra(filtered_df)
        data = {
            'Lugar de Venta': ['Dentro de la Chacra', 'Fuera de la Chacra'],
            'Toneladas': [venta_dentro_chacra_ton, venta_fuera_chacra_ton]
        }
        df_ventas_chacra = pd.DataFrame(data)
        fig = px.pie(df_ventas_chacra, values='Toneladas', names='Lugar de Venta',
                     title="Ventas Dentro y Fuera de la Chacra (ton)")

    return fig


@app.callback(
    Output('ventas-entidad-bar-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('cultivo-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_ventas_entidad_chart(selected_departamento, selected_provincia, selected_distrito, selected_cultivo,
                                toggle_state, region):
    # Copy and filter the DataFrame based on dropdown selections
    filtered_df = df.copy()
    if selected_departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == selected_departamento]
    if selected_provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == selected_provincia]
    if selected_distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == selected_distrito]

    # Filter by cultivo if selected
    if selected_cultivo:
        filtered_df = filtered_df[filtered_df['P204_NOM'] == selected_cultivo]
    if region:  # Filter based on selected 'REGION'
        filtered_df = filtered_df[filtered_df['REGION'] == region]


    # Group by unique production units and sum Total_Ha
    unique_productions = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA'])['Total_Ha'].first().reset_index()

    categorias = ['Acopiador', 'Comerciante mayorista', 'Comerciante minorista', 'Asociación / cooperativa',
                  'Empresa / agroindustria', 'Consumidor Final', 'Otro']

    if 'toggle' in toggle_state:
        hectare_range_data = []
        for lower, upper in hectare_ranges:
            # Select units within the current hectare range
            range_productions = unique_productions[
                (unique_productions['Total_Ha'] >= lower) & (unique_productions['Total_Ha'] < upper)]
            # Merge back to the main DataFrame to get ventas data for selected units
            range_df = pd.merge(range_productions, filtered_df, on=['CONGLOMERADO', 'NSELUA', 'UA'], how='left')
            ventas_por_rango = calcular_ventas_por_entidad(range_df)
            ventas_por_rango['Hectare Range'] = f'{lower}-{upper}'
            hectare_range_data.append(ventas_por_rango)

        # Convert to DataFrame
        plot_df = pd.DataFrame(hectare_range_data)
        # Create a stacked bar chart
        fig = px.bar(
            plot_df,
            x='Hectare Range',
            y=categorias,
            title="Ventas por Entidad y Rango de Hectáreas (ton)",
            text_auto=True
        )
        fig.update_layout(barmode='stack')
    else:
        # For the regular bar chart, use the unique production units
        unique_df = pd.merge(unique_productions, filtered_df, on=['CONGLOMERADO', 'NSELUA', 'UA'], how='left')
        ventas = calcular_ventas_por_entidad(unique_df)
        cantidades = [ventas[categoria] for categoria in categorias]
        df_ventas_entidad = pd.DataFrame({'Entidad': categorias, 'Cantidad': cantidades})
        fig = px.bar(df_ventas_entidad, x='Entidad', y='Cantidad', title="Ventas por Entidad (ton)")

    return fig


@app.callback(
    Output('destino-produccion-bar-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('cultivo-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_destino_produccion_chart(selected_departamento, selected_provincia, selected_distrito, selected_cultivo, toggle_state, region):
    filtered_df = df.copy()
    if selected_departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == selected_departamento]
    if selected_provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == selected_provincia]
    if selected_distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == selected_distrito]
    if selected_cultivo:
        filtered_df = filtered_df[filtered_df['P204_NOM'] == selected_cultivo]
    if region:  # Filter based on selected 'REGION'
        filtered_df = filtered_df[filtered_df['REGION'] == region]


    # Group by unique production units
    unique_productions = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA'])['Total_Ha'].first().reset_index()

    categorias = ['Mercado Local', 'Mercado Regional', 'Mercado Exterior', 'Agroindustria', 'Mercados de Lima',
                  'No Sabe', 'Mercado Local y Regional', 'Local, Regional y Lima', 'Mercado Regional y Agroindustria']

    if 'toggle' in toggle_state:
        hectare_range_data = []
        for lower, upper in hectare_ranges:
            # Select units within the current hectare range
            range_productions = unique_productions[
                (unique_productions['Total_Ha'] >= lower) & (unique_productions['Total_Ha'] < upper)]
            # Merge back to the main DataFrame to get destino de produccion data for selected units
            range_df = pd.merge(range_productions, filtered_df, on=['CONGLOMERADO', 'NSELUA', 'UA'], how='left')
            totales_destinos = calcular_destino_produccion(range_df, selected_cultivo)
            rango_data = {category: totales_destinos[i] for i, category in enumerate(categorias)}
            rango_data['Hectare Range'] = f'{lower}-{upper}'
            hectare_range_data.append(rango_data)

        plot_df = pd.DataFrame(hectare_range_data)
        fig = px.bar(
            plot_df,
            x='Hectare Range',
            y=categorias,
            title="Destino de la Producción por Rango de Hectáreas (ton)",
            text_auto=True
        )
        fig.update_layout(barmode='stack')
        return fig
    else:
        totales_destinos = calcular_destino_produccion(filtered_df, selected_cultivo)
        df_destino_produccion = pd.DataFrame({'Destino': categorias, 'Cantidad': totales_destinos})
        fig = px.bar(df_destino_produccion, x='Destino', y='Cantidad', title="Destino de la Producción (ton)")

    return fig



@app.callback(
    [
        Output("collapse", "is_open"),
        Output("acceso-credito-collapse", "is_open"),
        Output("asociatividad-collapse", "is_open"),
        Output("insumos-collapse", "is_open"),
        Output("capacitacion-collapse", "is_open"),
        Output("mapa-coropletico-collapse", "is_open"),
        Output("maquinaria-collapse", "is_open"),
        Output("cultivo-losses-collapse", "is_open")
    ],
    [Input("graph-selection-dropdown", "value")]
)
@cache.memoize(timeout=60 * 60)
def update_graph_visibility(selected_graph_set):
    return [
        selected_graph_set == 'prod_uso_venta',  # Collapse for Producción - Uso y Destinos de Venta
        selected_graph_set == 'servicios_financieros',  # Collapse for Servicios Financieros
        selected_graph_set == 'asociatividad',          # Collapse for Asociatividad
        selected_graph_set == 'uso_insumos',            # Collapse for Uso de insumos
        selected_graph_set == 'capacitacion_asistencia',# Collapse for Capacitación y Asistencia Técnica
        selected_graph_set == 'mapa_coropletico',        # Collapse for Mapa Coroplético
        selected_graph_set == 'maquinaria_equipos',
        selected_graph_set == 'cultivo_losses'      
    ]

@app.callback(
    Output('acceso-credito-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_acceso_credito_chart(departamento, provincia, distrito, toggle_state, region):
    # Filter df_cap900 based on selected area
    filtered_df = df_cap900.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:  # Filter based on selected 'REGION'
        filtered_df = filtered_df[filtered_df['REGION'] == region]


    if 'toggle' in toggle_state:
        # Logic for generating the stacked bar chart based on Total_Ha
        nivel_agregacion = 'NOMBREDI' if distrito else 'NOMBREPV' if provincia else 'NOMBREDD'
        hectare_range_data = []
        for lower, upper in hectare_ranges:
            range_df = filtered_df[(filtered_df['Total_Ha'] >= lower) & (filtered_df['Total_Ha'] < upper)]
            no_count = range_df[range_df['P901'] == 'No']['FACTOR'].sum()
            si_count = range_df[range_df['P901'] == 'Sí']['FACTOR'].sum()
            total = no_count + si_count
            prop_no = no_count / total * 100 if total > 0 else 0
            prop_si = si_count / total * 100 if total > 0 else 0
            hectare_range_data.append({
                'Hectare Range': f'{lower}-{upper}',
                'No': prop_no,
                'Sí': prop_si
            })

        plot_df = pd.DataFrame(hectare_range_data)
        fig = px.bar(
            plot_df,
            x='Hectare Range',
            y=['No', 'Sí'],
            title="¿Solicitó un crédito? por Rango de Hectáreas Cultivadas",
            barmode='stack',
            text_auto=True
        )
        fig.update_traces(texttemplate='%{y:.2f}%', textposition='inside')
    else:
        # Logic for the default state without the toggle
        no_count = filtered_df[filtered_df['P901'] == 'No']['FACTOR'].sum()
        si_count = filtered_df[filtered_df['P901'] == 'Sí']['FACTOR'].sum()
        total = no_count + si_count
        prop_no = no_count / total * 100 if total > 0 else 0
        prop_si = si_count / total * 100 if total > 0 else 0
        values = [prop_no, prop_si]
        labels = ['No', 'Sí']
        fig = px.pie(names=labels, values=values, title="¿Solicitó un crédito?")

    return fig


@app.callback(
    Output('obtencion-credito-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_obtencion_credito_chart(departamento, provincia, distrito, toggle_state, region):
    # Filter df_cap900 based on selected area
    filtered_df = df_cap900.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:  # Filter based on selected 'REGION'
        filtered_df = filtered_df[filtered_df['REGION'] == region]


    if 'toggle' in toggle_state:
        # Logic for generating the stacked bar chart based on Total_Ha
        nivel_agregacion = 'NOMBREDI' if distrito else 'NOMBREPV' if provincia else 'NOMBREDD'
        hectare_range_data = []
        for lower, upper in hectare_ranges:
            range_df = filtered_df[(filtered_df['Total_Ha'] >= lower) & (filtered_df['Total_Ha'] < upper)]
            no_count = range_df[range_df['P902'] == 'No']['FACTOR'].sum()
            si_count = range_df[range_df['P902'] == 'Sí']['FACTOR'].sum()
            total = no_count + si_count
            prop_no = (no_count / total) * 100 if total > 0 else 0
            prop_si = (si_count / total) * 100 if total > 0 else 0
            hectare_range_data.append({
                'Hectare Range': f'{lower}-{upper}',
                'No': prop_no,
                'Sí': prop_si
            })

        plot_df = pd.DataFrame(hectare_range_data)
        fig = px.bar(
            plot_df,
            x='Hectare Range',
            y=['No', 'Sí'],
            title="¿Obtuvo el crédito? por Rango de Hectáreas Cultivadas",
            barmode='stack',
            text_auto=True
        )
        fig.update_traces(texttemplate='%{y:.2f}%', textposition='inside')
    else:
        # Logic for the default state without the toggle
        no_count = filtered_df[filtered_df['P902'] == 'No']['FACTOR'].sum()
        si_count = filtered_df[filtered_df['P902'] == 'Sí']['FACTOR'].sum()
        total = no_count + si_count
        prop_no = (no_count / total) * 100 if total > 0 else 0
        prop_si = (si_count / total) * 100 if total > 0 else 0
        values = [prop_no, prop_si]
        labels = ['No', 'Sí']
        fig = px.pie(names=labels, values=values, title="¿Obtuvo el crédito?")

    return fig

@app.callback(
    Output('seguro-agricola-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_seguro_agricola_chart(departamento, provincia, distrito, toggle_state, region):
    # Filter df_cap900 based on selected area
    filtered_df = df_cap900.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:  # Filter based on selected 'REGION'
        filtered_df = filtered_df[filtered_df['REGION'] == region]

    if 'toggle' in toggle_state:
        # Logic for generating the stacked bar chart based on Total_Ha
        nivel_agregacion = 'NOMBREDI' if distrito else 'NOMBREPV' if provincia else 'NOMBREDD'
        hectare_range_data = []
        for lower, upper in hectare_ranges:
            range_df = filtered_df[(filtered_df['Total_Ha'] >= lower) & (filtered_df['Total_Ha'] < upper)]
            no_count = range_df[range_df['P905'] == 'No']['FACTOR'].sum()
            si_count = range_df[range_df['P905'] == 'Sí']['FACTOR'].sum()
            total = no_count + si_count
            prop_no = no_count / total * 100 if total > 0 else 0
            prop_si = si_count / total * 100 if total > 0 else 0
            hectare_range_data.append({
                'Hectare Range': f'{lower}-{upper}',
                'No': prop_no,
                'Sí': prop_si
            })

        plot_df = pd.DataFrame(hectare_range_data)
        fig = px.bar(
            plot_df,
            x='Hectare Range',
            y=['No', 'Sí'],
            title="¿Accedió a un seguro agrícola? por Rango de Hectáreas Cultivadas",
            barmode='stack',
            text_auto=True
        )
        fig.update_traces(texttemplate='%{y:.2f}%', textposition='inside')
    else:
        # Logic for the default state without the toggle
        no_count = filtered_df[filtered_df['P905'] == 'No']['FACTOR'].sum()
        si_count = filtered_df[filtered_df['P905'] == 'Sí']['FACTOR'].sum()
        total = no_count + si_count
        prop_no = no_count / total * 100 if total > 0 else 0
        prop_si = si_count / total * 100 if total > 0 else 0
        values = [prop_no, prop_si]
        labels = ['No', 'Sí']
        fig = px.pie(names=labels, values=values, title="¿Accedió a un seguro agrícola?")
        fig.update_traces(textinfo='percent')

    return fig

@app.callback(
    Output('proveedor-seguro-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_proveedor_seguro_chart(departamento, provincia, distrito, toggle_state, region):
    # Filter df_cap900 based on selected area
    filtered_df = df_cap900.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]

    if 'toggle' in toggle_state:
        # Logic for generating the stacked bar chart based on Total_Ha
        nivel_agregacion = 'NOMBREDI' if distrito else 'NOMBREPV' if provincia else 'NOMBREDD'
        hectare_range_data = []
        for lower, upper in hectare_ranges:
            range_df = filtered_df[(filtered_df['Total_Ha'] >= lower) & (filtered_df['Total_Ha'] < upper)]
            counts = range_df.groupby('P906')['FACTOR'].sum()
            hectare_range_data.append({
                'Hectare Range': f'{lower}-{upper}',
                'Ministerio de Desarrollo Agrario y Riego': counts.get('Ministerio de Desarrollo Agrario y Riego', 0),
                'Empresa aseguradora': counts.get('Empresa aseguradora', 0),
                'Banca privada': counts.get('Banca privada', 0),
                'AGROBANCO': counts.get('AGROBANCO', 0)
            })

        plot_df = pd.DataFrame(hectare_range_data)
        fig = px.bar(
            plot_df,
            x='Hectare Range',
            y=['Ministerio de Desarrollo Agrario y Riego', 'Empresa aseguradora', 'Banca privada', 'AGROBANCO'],
            title="¿Quién le proporcionó el seguro? por Rango de Hectáreas Cultivadas",
            barmode='stack',
            text_auto=True
        )
    else:
        # Logic for the default state without the toggle
        counts = filtered_df.groupby('P906')['FACTOR'].sum()
        values = [counts.get('Ministerio de Desarrollo Agrario y Riego', 0), counts.get('Empresa aseguradora', 0), counts.get('Banca privada', 0), counts.get('AGROBANCO', 0)]
        labels = ['Ministerio de Desarrollo Agrario y Riego', 'Empresa aseguradora', 'Banca privada', 'AGROBANCO']
        fig = px.pie(names=labels, values=values, title="¿Quién le proporcionó el seguro?")

    return fig

@app.callback(
    Output('entidad-credito-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_entidad_credito_chart(departamento, provincia, distrito, toggle_state, region):
    filtered_df = df_cap900.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]

    credit_providers = [
        'AGROBANCO', 'Caja Municipal', 'Caja Rural', 'Banca privada', 'Financiera/EDPYME',
        'Organismo No Gubernamental (ONG)', 'Cooperativa', 'Establecimiento comercial',
        'Prestamista/Habilitador', 'Programas del Estado', 'Otro'
    ]

    if 'toggle' in toggle_state:
        hectare_range_data = []
        for lower, upper in hectare_ranges:
            range_df = filtered_df[(filtered_df['Total_Ha'] >= lower) & (filtered_df['Total_Ha'] < upper)]
            counts = {provider: range_df[range_df[f'P903_{i + 1}'] == provider]['FACTOR'].sum() for i, provider in
                      enumerate(credit_providers)}
            counts['Hectare Range'] = f'{lower}-{upper}'
            hectare_range_data.append(counts)

        plot_df = pd.DataFrame(hectare_range_data)
        fig = px.bar(
            plot_df,
            x='Hectare Range',
            y=credit_providers,
            title="¿Quién le proporcionó el crédito? por Rango de Hectáreas Cultivadas",
            barmode='stack',
            text_auto=True
        )
    else:
        overall_counts = {provider: filtered_df[filtered_df[f'P903_{i + 1}'] == provider]['FACTOR'].sum() for
                          i, provider in enumerate(credit_providers)}
        fig = px.pie(names=list(overall_counts.keys()), values=list(overall_counts.values()),
                     title="¿Quién le proporcionó el crédito?")

    return fig


@app.callback(
    Output('razones-rechazo-credito-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_razones_rechazo_credito_chart(departamento, provincia, distrito, toggle_state, region):
    filtered_df = df_cap900.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]

    specified_reasons = [
        'Por falta de garantía',
        'Incumplimiento de pagos anteriores',
        'Por no tener título de propiedad de la tierra (parcela)',
        'Actividad agropecuaria con alto riesgo'
    ]

    # Function to categorize reasons
    def categorize_reason(row):
        if row in specified_reasons:
            return row
        elif pd.notna(row):
            return 'Otras'
        else:
            return 'No especificado'

    # Apply categorization
    filtered_df['Categorized_Reason'] = filtered_df['P904A'].apply(categorize_reason)

    if 'toggle' in toggle_state:
        hectare_range_data = []
        for lower, upper in hectare_ranges:
            range_df = filtered_df[(filtered_df['Total_Ha'] >= lower) & (filtered_df['Total_Ha'] < upper)]
            counts = range_df.groupby('Categorized_Reason')['FACTOR'].sum().to_dict()
            counts['Hectare Range'] = f'{lower}-{upper}'
            hectare_range_data.append(counts)

        plot_df = pd.DataFrame(hectare_range_data).fillna(0)
        fig = px.bar(
            plot_df,
            x='Hectare Range',
            y=specified_reasons + ['Otras'],
            title="Razones para Rechazo de Crédito por Rango de Hectáreas Cultivadas (Con Factor)",
            barmode='stack',
            text_auto=True
        )
    else:
        counts = filtered_df.groupby('Categorized_Reason')['FACTOR'].sum().to_dict()
        fig = px.pie(names=list(counts.keys()), values=list(counts.values()),
                     title="Razones para Rechazo de Crédito (Con Factor)")

    return fig


@app.callback(
    Output('asociatividad-chart', 'figure'),
    [
        Input('departamento-dropdown', 'value'),
        Input('provincia-dropdown', 'value'),
        Input('distrito-dropdown', 'value'),
        Input('chart-toggle', 'value'),
        Input('cultivo-dropdown', 'value'),
        Input('region-dropdown', 'value')
    ]
)
@cache.memoize(timeout=60 * 60)
def update_asociatividad_chart(departamento, provincia, distrito, toggle_state, selected_cultivo, region):
    # Start with the merged DataFrame that includes the 'cultivo' dummy columns
    filtered_df = df_cap800.copy()

    # Apply filters based on selected departamento, provincia, and distrito
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]

    # Filter by the selected 'cultivo' if one is selected
    if selected_cultivo:
        filtered_df = filtered_df[filtered_df[selected_cultivo] == 1]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]

    if 'toggle' in toggle_state:
        hectare_range_data = []
        for lower, upper in hectare_ranges:
            range_df = filtered_df[(filtered_df['Total_Ha'] >= lower) & (filtered_df['Total_Ha'] < upper)]
            si_count = range_df[range_df['P801'] == 'Sí']['FACTOR'].sum()
            no_count = range_df[range_df['P801'] == 'No']['FACTOR'].sum()
            hectare_range_data.append({
                'Hectare Range': f'{lower}-{upper}',
                'Sí': si_count,
                'No': no_count
            })

        plot_df = pd.DataFrame(hectare_range_data)
        fig = px.bar(
            plot_df,
            x='Hectare Range',
            y=['Sí', 'No'],
            title="Asociatividad por Rango de Hectáreas Cultivadas",
            barmode='stack',
            text_auto=True
        )
    else:
        si_count = filtered_df[filtered_df['P801'] == 'Sí']['FACTOR'].sum()
        no_count = filtered_df[filtered_df['P801'] == 'No']['FACTOR'].sum()
        fig = px.pie(names=['Sí', 'No'], values=[si_count, no_count], title="Asociatividad")

    return fig

@app.callback(
    Output('razones-no-asociatividad-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_razones_no_asociatividad_chart(departamento, provincia, distrito, toggle_state, region):
    filtered_df = df_cap800.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]  

    # Define the reasons for no asociatividad
    no_asociatividad_reasons = {
        'P809A_1': 'Por desconfianza',
        'P809A_2': 'No hay disponibilidad de cooperativas en la zona',
        'P809A_3': 'Por desconocimiento',
        'P809A_4': 'No considera útil asociarse',
        'P809A_5': 'Pérdida de ingresos por pago de impuestos',
        'P809A_6': 'Otro'
    }

    # Function to categorize reasons
    def categorize_reason(row):
        for code, reason in no_asociatividad_reasons.items():
            if row[code] == reason:  # If the reason is marked as true
                return reason
        return 'No especificado'  # Use this if no reason is marked

    # Apply categorization
    filtered_df['Categorized_Reason'] = filtered_df.apply(categorize_reason, axis=1)

    if 'toggle' in toggle_state:
        hectare_range_data = []
        for lower, upper in hectare_ranges:
            range_df = filtered_df[(filtered_df['Total_Ha'] >= lower) & (filtered_df['Total_Ha'] < upper)]
            counts = range_df.groupby('Categorized_Reason')['FACTOR'].sum().to_dict()
            counts['Hectare Range'] = f'{lower}-{upper}'
            hectare_range_data.append(counts)

        # After grouping and summing, make sure to fill missing reasons with zeros
        for hectare_info in hectare_range_data:
            for reason in no_asociatividad_reasons.values():
                if reason not in hectare_info:
                    hectare_info[reason] = 0  # Set missing reasons to zero

        # Then create your DataFrame
        plot_df = pd.DataFrame(hectare_range_data)

        # Now plot_df has consistent data and can be used in px.bar
        fig = px.bar(
            plot_df,
            x='Hectare Range',
            y=list(no_asociatividad_reasons.values()) + ['No especificado'],
            title="Razones de No Asociatividad por Rango de Hectáreas Cultivadas",
            barmode='stack',
            text_auto=True,
            labels={'variable': 'Razón', 'value': 'FACTOR'}
        )
    else:
        counts = filtered_df.groupby('Categorized_Reason')['FACTOR'].sum().to_dict()
        fig = px.pie(names=list(counts.keys()), values=list(counts.values()),
                     title="Razones de No Asociatividad")

    return fig



@app.callback(
    Output('selling-vs-non-selling-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_selling_chart(selected_departamento, selected_provincia, selected_distrito, toggle_state, region):
    # Filter DataFrame based on selected criteria
    filtered_df = df.copy()
    if selected_departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == selected_departamento]
    if selected_provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == selected_provincia]
    if selected_distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == selected_distrito]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]  

    # Replace NaNs with zeros for 'venta_kg' and 'total_kg'
    filtered_df['venta_kg'].fillna(0, inplace=True)
    filtered_df['total_kg'].fillna(0, inplace=True)

    # Group by unique production units
    group = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA'])
    venta_kg_sum = group['venta_kg'].sum()
    total_kg_sum = group['total_kg'].sum()
    total_ha_first = group['Total_Ha'].first()
    factor_first = group['FACTOR'].first()

    if 'toggle' in toggle_state:
        hectare_range_data = []

        for lower, upper in hectare_ranges:
            upper_bound = 'inf' if upper == float('inf') else upper

            # Create a mask for the current hectare range
            mask = (total_ha_first >= lower) & (total_ha_first < upper)
            filtered_venta_kg_sum = venta_kg_sum[mask]
            filtered_total_kg_sum = total_kg_sum[mask].replace(0, np.nan)  # Replace 0 with NaN to avoid division by zero
            filtered_factor_first = factor_first[mask]

            # Calculate the weighted average for the hectare range
            weighted_proportions = (filtered_venta_kg_sum / filtered_total_kg_sum).fillna(0) * filtered_factor_first
            total_factor = filtered_factor_first.sum()

            weighted_avg_percentage = (weighted_proportions.sum() / total_factor) * 100 if total_factor > 0 else 0

            hectare_range_data.append({
                'Rango de Hectáreas': f'{lower}-{upper_bound}',
                'Promedio Ponderado de Proporción de Ventas': weighted_avg_percentage
            })

        plot_df = pd.DataFrame(hectare_range_data)
        fig = px.bar(plot_df, x='Rango de Hectáreas', y='Promedio Ponderado de Proporción de Ventas',
                     title='Promedio Ponderado del (%) de Ventas por UA')
    else:
        # Non-toggle state
        # Handle division by zero for the non-toggle state
        proportion_selling = (venta_kg_sum / total_kg_sum.replace(0, np.nan)).fillna(0) * factor_first
        proportion_selling_avg = (proportion_selling.sum() / factor_first.sum()) * 100 if factor_first.sum() > 0 else 0

        # Prepare data for the bar chart
        data = {
            'Category': ['Proportion Selling'],
            'Value': [proportion_selling_avg]
        }
        df_chart = pd.DataFrame(data)
        fig = px.bar(df_chart, x='Value', y='Category', orientation='h',
                     title='Proporción de Ventas en Relación a la Producción Total (%)')

        # Customize the layout for better presentation
        fig.update_layout(
            xaxis_title='Porcentaje',
            yaxis_title='',
            showlegend=False
        )

    return fig

@app.callback(
    Output('insumos-usage-chart', 'figure'),
    [
        Input('departamento-dropdown', 'value'),
        Input('provincia-dropdown', 'value'),
        Input('distrito-dropdown', 'value'),
        Input('chart-toggle', 'value'),
        Input('insumos-selection-dropdown', 'value'),
        Input('region-dropdown', 'value')
    ]
)
@cache.memoize(timeout=60 * 60)
def update_insumos_usage_chart(departamento, provincia, distrito, toggle_state, selected_insumos, region):
    # Load and merge data
    df_cap300ab = pd.read_stata('C:\\UC\\RIMISP\\Encuestas Perú\\2019\\2022\\1744 - BPA\\08_Cap300ab.dta')
    relevant_columns_df = df[['CONGLOMERADO', 'NSELUA', 'UA', 'Total_Ha']].drop_duplicates()
    merged_df = pd.merge(df_cap300ab, relevant_columns_df, on=['CONGLOMERADO', 'NSELUA', 'UA'], how='left')

    # Apply filters
    if departamento:
        merged_df = merged_df[merged_df['NOMBREDD'] == departamento]
    if provincia:
        merged_df = merged_df[merged_df['NOMBREPV'] == provincia]
    if distrito:
        merged_df = merged_df[merged_df['NOMBREDI'] == distrito]
    if region:
        merged_df = merged_df[merged_df['REGION'] == region]

    # Logic to handle the selected insumos
    if selected_insumos:
        # Initialize counts for each hectare range
        range_counts = {range_val: {'Sí': 0, 'No': 0} for range_val in hectare_ranges}

        for lower, upper in hectare_ranges:
            for index, row in merged_df.iterrows():
                if row['Total_Ha'] >= lower and row['Total_Ha'] < upper:
                    if all(row[insumo] == 'Sí' for insumo in selected_insumos):
                        range_counts[(lower, upper)]['Sí'] += row['FACTOR']
                    else:
                        range_counts[(lower, upper)]['No'] += row['FACTOR']

        # Prepare data for the chart
        chart_data = []
        for range_val, counts in range_counts.items():
            chart_data.append({'Hectare Range': f'{range_val[0]}-{range_val[1]}', 'Insumos': 'Sí', 'Count': counts['Sí']})
            chart_data.append({'Hectare Range': f'{range_val[0]}-{range_val[1]}', 'Insumos': 'No', 'Count': counts['No']})
        chart_df = pd.DataFrame(chart_data)

        # Create the figure based on the toggle state
        if 'toggle' in toggle_state:
            # Stacked bar chart
            fig = px.bar(chart_df, x='Hectare Range', y='Count', color='Insumos', title="Uso de Insumos por Rango de Hectáreas", barmode='stack')
        else:
            # Pie chart (overall usage without considering hectare ranges)
            total_si = sum(counts['Sí'] for _, counts in range_counts.items())
            total_no = sum(counts['No'] for _, counts in range_counts.items())
            fig = px.pie(names=['Sí', 'No'], values=[total_si, total_no], title="Uso de Insumos General")
    else:
        # If no insumo is selected, display an empty figure
        fig = go.Figure()

    return fig

@app.callback(
    Output('capacitacion-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_capacitacion_chart(departamento, provincia, distrito, toggle_state, region):
    # Filter df_cap700 based on selected area
    filtered_df = df_cap700.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]

    if 'toggle' in toggle_state:
        hectare_range_data = []
        for lower, upper in hectare_ranges:
            range_df = filtered_df[(filtered_df['Total_Ha'] >= lower) & (filtered_df['Total_Ha'] < upper)]
            si_factor_sum = range_df[range_df['P701'] == 'Sí']['FACTOR'].sum()
            no_factor_sum = range_df[range_df['P701'] == 'No']['FACTOR'].sum()
            total_factor_sum = si_factor_sum + no_factor_sum
            prop_si = si_factor_sum / total_factor_sum * 100 if total_factor_sum > 0 else 0
            prop_no = no_factor_sum / total_factor_sum * 100 if total_factor_sum > 0 else 0
            hectare_range_data.append({
                'Hectare Range': f'{lower}-{upper}',
                'Sí': prop_si,
                'No': prop_no,
                'Total Sí': si_factor_sum,
                'Total No': no_factor_sum
            })

        plot_df = pd.DataFrame(hectare_range_data)
        fig = px.bar(
            plot_df,
            x='Hectare Range',
            y=['Sí', 'No'],
            title="Personas que Recibieron Capacitación en los últimos 12 meses por Rango de Hectáreas Cultivadas",
            text_auto=True
        )

        fig.update_layout(barmode='stack')
        fig.update_traces(texttemplate='%{y:.2f}%', textposition='inside')
    else:
        si_count = filtered_df[filtered_df['P701'] == 'Sí']['FACTOR'].sum()
        no_count = filtered_df[filtered_df['P701'] == 'No']['FACTOR'].sum()
        total_count = si_count + no_count
        prop_si = si_count / total_count * 100 if total_count > 0 else 0
        prop_no = no_count / total_count * 100 if total_count > 0 else 0
        fig = px.pie(names=['Sí', 'No'], values=[prop_si, prop_no], title="¿Recibió capacitación en los últimos 12 meses?")

    return fig

@app.callback(
    Output('asistencia-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_asistencia_chart(departamento, provincia, distrito, toggle_state, region):
    # Filter df_cap700 based on selected area
    filtered_df = df_cap700.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]

    if 'toggle' in toggle_state:
        hectare_range_data = []
        for lower, upper in hectare_ranges:
            range_df = filtered_df[(filtered_df['Total_Ha'] >= lower) & (filtered_df['Total_Ha'] < upper)]
            si_factor_sum = range_df[range_df['P704'] == 'Sí']['FACTOR'].sum()
            no_factor_sum = range_df[range_df['P704'] == 'No']['FACTOR'].sum()
            total_factor_sum = si_factor_sum + no_factor_sum
            prop_si = si_factor_sum / total_factor_sum * 100 if total_factor_sum > 0 else 0
            prop_no = no_factor_sum / total_factor_sum * 100 if total_factor_sum > 0 else 0
            hectare_range_data.append({
                'Hectare Range': f'{lower}-{upper}',
                'Sí': prop_si,
                'No': prop_no,
                'Total Sí': si_factor_sum,
                'Total No': no_factor_sum
            })

        plot_df = pd.DataFrame(hectare_range_data)
        fig = px.bar(
            plot_df,
            x='Hectare Range',
            y=['Sí', 'No'],
            title="Personas que Recibieron Asistencia Técnica en los últimos 12 meses por Rango de Hectáreas Cultivadas",
            text_auto=True
        )

        fig.update_layout(barmode='stack')
        fig.update_traces(texttemplate='%{y:.2f}%', textposition='inside')
    else:
        si_count = filtered_df[filtered_df['P704'] == 'Sí']['FACTOR'].sum()
        no_count = filtered_df[filtered_df['P704'] == 'No']['FACTOR'].sum()
        total_count = si_count + no_count
        prop_si = si_count / total_count * 100 if total_count > 0 else 0
        prop_no = no_count / total_count * 100 if total_count > 0 else 0
        fig = px.pie(names=['Sí', 'No'], values=[prop_si, prop_no], title="¿Recibió capacitación en los últimos 12 meses?")

    return fig

@app.callback(
    Output('cantidad-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_cantidad_chart(departamento, provincia, distrito, toggle_state, region):
    # Filter df based on selected area
    filtered_df = df.copy()  # Replace 'df' with your actual DataFrame variable
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]

    # Group by unique combinations and take the first occurrence of Total_Ha and FACTOR
    unique_combinations = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA']).agg({
        'Total_Ha': 'first',
        'FACTOR': 'first',
        'venta_kg': 'sum',
        'total_kg': 'sum'
    }).reset_index()

    # Calculate if there's any sale for each combination
    unique_combinations['ventas_exists'] = unique_combinations['venta_kg'] > 0

    # Calculate the sales proportion for each unique production
    unique_combinations['sales_proportion'] = unique_combinations.apply(
        lambda x: x['venta_kg'] / x['total_kg'] if x['total_kg'] > 0 else 0, axis=1
    )

    # Define the proportion categories with the new 'No vende' category
    proportion_conditions = [
        (unique_combinations['sales_proportion'] == 0),
        (unique_combinations['sales_proportion'] > 0) & (unique_combinations['sales_proportion'] < 0.3),
        (unique_combinations['sales_proportion'] >= 0.3) & (unique_combinations['sales_proportion'] < 0.5),
        (unique_combinations['sales_proportion'] >= 0.5) & (unique_combinations['sales_proportion'] < 0.75),
        (unique_combinations['sales_proportion'] >= 0.75)
    ]
    proportion_categories = ['No vende', '< 30%', '30-50%', '50-75%', '75% o más']
    unique_combinations['proportion_category'] = np.select(proportion_conditions, proportion_categories, default='No Sale')

    # Aggregate by proportion category and hectare range
    proportion_range_data = []
    for lower, upper in hectare_ranges:
        upper_bound = 'inf' if upper == float('inf') else upper
        mask = (unique_combinations['Total_Ha'] >= lower) & (unique_combinations['Total_Ha'] < upper)
        for category in proportion_categories:
            category_mask = (unique_combinations['proportion_category'] == category)
            proportion_factor = (unique_combinations[mask & category_mask]['FACTOR']).sum()
            proportion_range_data.append({
                'Rango de Hectáreas': f'{lower}-{upper_bound}',
                'Category': category,
                'Value': proportion_factor
            })

    # Convert to DataFrame
    proportion_df = pd.DataFrame(proportion_range_data)

    # Check if the toggle is active and modify the chart accordingly
    if toggle_state:
        # Area chart for the toggle state, disaggregated by sales proportions
        fig = px.area(proportion_df, x='Rango de Hectáreas', y='Value', color='Category',
                      title='Área de Ventas por Rango de Hectáreas y Proporción de Ventas')
    else:
        # Prepare the original hectare range dataframe for the default area chart
        hectare_range_data = []
        for lower, upper in hectare_ranges:
            upper_bound = 'inf' if upper == float('inf') else upper
            mask = (unique_combinations['Total_Ha'] >= lower) & (unique_combinations['Total_Ha'] < upper)
            selling_factor = (unique_combinations[mask & unique_combinations['ventas_exists']]['FACTOR']).sum()
            not_selling_factor = (unique_combinations[mask & ~unique_combinations['ventas_exists']]['FACTOR']).sum()
            hectare_range_data.append({
                'Rango de Hectáreas': f'{lower}-{upper_bound}',
                'Selling': selling_factor,
                'Not Selling': not_selling_factor
            })

        plot_df = pd.DataFrame(hectare_range_data)

        # Create the default area chart
        fig = px.area(plot_df, x='Rango de Hectáreas', y=['Selling', 'Not Selling'],
                      title='Área de Ventas y No Ventas por Rango de Hectáreas')

    # Return the figure
    return fig

@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('cultivo-dropdown', 'value'),
     Input('data-type-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_map(cultivo, data_type):
    # Filter the dataframe based on the selected cultivo
    if cultivo:
        filtered_df = df[df['P204_NOM'] == cultivo]
    else:
        filtered_df = df.copy()

    # Apply the function to calculate kilos
    filtered_df[['total_kg2', 'venta_kg2']] = filtered_df.apply(calcular_kilos2, axis=1)

    # Create a unique DataFrame that only contains unique 'CONGLOMERADO', 'NSELUA', and 'UA' combinations
    df_unique = filtered_df[['CONGLOMERADO', 'NSELUA', 'UA', 'NOMBREDD']].drop_duplicates()

    # Merge df_unique with df_cap800, only retrieving 'P801'
    merged_df = pd.merge(df_unique, df_cap800[['CONGLOMERADO', 'NSELUA', 'UA', 'P801', 'FACTOR']], 
                         on=['CONGLOMERADO', 'NSELUA', 'UA'], how='left')

    # Aggregate data
    total_hectares = filtered_df.groupby('NOMBREDD')['P217_SUP_ha'].sum()
    total_production = filtered_df.groupby('NOMBREDD')['total_kg2'].sum() / 1000
    total_sales = filtered_df.groupby('NOMBREDD')['venta_kg2'].sum() / 1000
    cooperatives_factor = merged_df[merged_df['P801'] == 'Sí'].groupby('NOMBREDD')['FACTOR'].sum()

    # Merge with the GeoDataFrame
    merged_gdf = gdf.merge(total_hectares, left_on='NOMBDEP', right_on='NOMBREDD')
    merged_gdf = merged_gdf.merge(total_production, left_on='NOMBDEP', right_on='NOMBREDD',
                                  suffixes=('_hectares', '_production'))
    merged_gdf = merged_gdf.merge(total_sales, left_on='NOMBDEP', right_on='NOMBREDD', suffixes=('', '_sales'))
    merged_gdf = merged_gdf.merge(cooperatives_factor, left_on='NOMBDEP', right_on='NOMBREDD', how='left')

    # Mapping of data_type to column names in merged_gdf
    data_type_to_column = {
        'hectares': 'P217_SUP_ha',
        'production': 'total_kg2',
        'sales': 'venta_kg2',
        'cooperatives': 'FACTOR'  # Add this line
    }

    # Select the correct column for coloring based on data_type
    color_column = data_type_to_column.get(data_type, 'P217_SUP_ha')  # Default to 'hectares' if data_type is not found

    # Update labels in the px.choropleth() function
    labels = {
        'P217_SUP_ha': 'Hectáreas cultivadas',
        'total_kg2': 'Producción total',
        'venta_kg2': 'Ventas totales',
        'FACTOR': 'Número de UAs en cooperativas'
    }

    # Create the choropleth map
    fig = px.choropleth(
        merged_gdf,
        geojson=geojson,
        locations='NOMBDEP',
        color=color_column,  # Use the correct column for coloring
        featureidkey="properties.NOMBDEP",
        projection="mercator",
        labels=labels  # Use the updated labels
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig

@app.callback(
    Output('maquinaria-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_maquinaria_equipos_chart(departamento, provincia, distrito, toggle_state, region):
    # Filter based on selected area
    filtered_df = merged_df2.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]

    # Ensure 'P1228_N' is numeric for multiplication
    filtered_df['P1228_N'] = pd.to_numeric(filtered_df['P1228_N'], errors='coerce').fillna(0)

    # Group by unique production units and get the first 'Total_Ha' for each
    unique_productions = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA'])['Total_Ha'].first().reset_index()

    hectare_range_data = []
    for lower, upper in hectare_ranges:
        # Select units within the current hectare range
        range_productions = unique_productions[
            (unique_productions['Total_Ha'] >= lower) & (unique_productions['Total_Ha'] < upper)]
        # Merge back to the main DataFrame to get data for selected units
        range_df = pd.merge(range_productions, filtered_df, on=['CONGLOMERADO', 'NSELUA', 'UA'], how='left')

        # Calculate the count multiplied by 'P1228_N' for Maquinaria and Equipo
        maquinaria_count = (range_df[range_df['P1228_TIPO'] == 'Maquinaria']['FACTOR'] * range_df['P1228_N']).sum()
        equipo_count = (range_df[range_df['P1228_TIPO'] == 'Equipo']['FACTOR'] * range_df['P1228_N']).sum()

        hectare_range_data.append({
            'Hectare Range': f'{lower}-{upper if upper != float("inf") else "∞"}',
            'Maquinaria': maquinaria_count,
            'Equipo': equipo_count
        })

    plot_df = pd.DataFrame(hectare_range_data)

    if 'toggle' in toggle_state:
        # Stacked bar chart for the toggle state
        fig = px.bar(plot_df, x='Hectare Range', y=['Maquinaria', 'Equipo'],
                     title='Cantidad de Maquinarias y Equipos por Rango de Hectáreas Cultivadas',
                     text_auto=True,
                     barmode='stack')
    else:
        # Pie chart for the default state
        total_maquinaria = plot_df['Maquinaria'].sum()
        total_equipo = plot_df['Equipo'].sum()
        fig = px.pie(names=['Maquinaria', 'Equipo'], values=[total_maquinaria, total_equipo],
                     title='Cantidad Total de Maquinarias y Equipos')

    return fig

@app.callback(
    Output('ua-maquinaria-equipos-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_ua_maquinaria_equipos_chart(departamento, provincia, distrito, toggle_state, region):
    # Filter based on selected area
    filtered_df = merged_df2.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]

    # Get the first Total_Ha and FACTOR for each unique production unit
    unique_productions = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA'])[
        'Total_Ha', 'FACTOR'].first().reset_index()

    # Check for Maquinaria or Equipo in any row of each unique production unit
    has_maquinaria_or_equipo = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA'])['P1228_TIPO'].apply(
        lambda x: any(item in ['Maquinaria', 'Equipo'] for item in x)).reset_index(name='Has_Maquinaria_Equipo')

    # Merge the information back into unique_productions
    unique_productions = pd.merge(unique_productions, has_maquinaria_or_equipo, on=['CONGLOMERADO', 'NSELUA', 'UA'])

    # Convert boolean to a more readable format
    unique_productions['Has_Maquinaria_Equipo'] = unique_productions['Has_Maquinaria_Equipo'].map(
        {True: 'Tiene maquinaria o equipo', False: 'No tiene'})

    hectare_range_data = []
    for lower, upper in hectare_ranges:
        # Select units within the current hectare range
        range_productions = unique_productions[
            (unique_productions['Total_Ha'] >= lower) & (unique_productions['Total_Ha'] < upper)]

        # Count the number of UAs with and without Maquinaria/Equipo
        count_has = range_productions[range_productions['Has_Maquinaria_Equipo'] == 'Tiene maquinaria o equipo'][
            'FACTOR'].sum()
        count_doesnt_have = range_productions[range_productions['Has_Maquinaria_Equipo'] == 'No tiene']['FACTOR'].sum()

        hectare_range_data.append({
            'Hectare Range': f'{lower}-{upper if upper != float("inf") else "∞"}',
            'Tiene maquinaria o equipo': count_has,
            'No tiene': count_doesnt_have
        })

    plot_df = pd.DataFrame(hectare_range_data)

    if 'toggle' in toggle_state:
        # Stacked bar chart for the toggle state
        fig = px.bar(plot_df, x='Hectare Range', y=['Tiene maquinaria o equipo', 'No tiene'],
                     title='Cantidad de UA con y sin Maquinaria o Equipo por Rango de Hectáreas Cultivadas',
                     text_auto=True,
                     barmode='stack')
    else:
        # Pie chart for the default state
        total_has = plot_df['Tiene maquinaria o equipo'].sum()
        total_doesnt_have = plot_df['No tiene'].sum()
        fig = px.pie(names=['Tiene maquinaria o equipo', 'No tiene'], values=[total_has, total_doesnt_have],
                     title='Cantidad de UA con y sin Maquinaria o Equipo')

    return fig

@app.callback(
    Output('unique-maquinaria-equipos-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_maquinaria_equipos_chart(departamento, provincia, distrito, toggle_state, region):
    # Filter based on selected area
    filtered_df = merged_df2.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]

    # Convert 'P1228_N' to numeric
    filtered_df['P1228_N'] = pd.to_numeric(filtered_df['P1228_N'], errors='coerce').fillna(0)

    # Define specific machinery and equipment names
    common_types = [
        'TRACTOR AGRICOLA',
        'MOCHILA FUMIGADORA',
        'MOTOGUADAÑA',
        'DESPULPADORA',
        'COSECHADORA',
        'MOTOBOMBA'
    ]

    # Categorize each item in 'P1228_NOM'
    filtered_df['Item_Category'] = filtered_df['P1228_NOM'].apply(lambda x: x if x in common_types else 'Otra maquinaria o equipo')

    # Group by unique production units and get the first 'Total_Ha' for each
    unique_productions = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA'])['Total_Ha'].first().reset_index()

    hectare_range_data = []
    for lower, upper in hectare_ranges:
        # Select units within the current hectare range
        range_productions = unique_productions[
            (unique_productions['Total_Ha'] >= lower) & (unique_productions['Total_Ha'] < upper)]
        # Merge back to the main DataFrame to get data for selected units
        range_df = pd.merge(range_productions, filtered_df, on=['CONGLOMERADO', 'NSELUA', 'UA'], how='left')
        # Multiply each 'FACTOR' by 'P1228_N' and sum up for each category within the hectare range
        range_df['Weighted_COUNT'] = range_df['FACTOR'] * range_df['P1228_N']
        category_counts = range_df.groupby('Item_Category')['Weighted_COUNT'].sum()

        # Prepare data for the chart
        chart_data = category_counts.reindex(common_types + ['Otra maquinaria o equipo'], fill_value=0).to_dict()
        chart_data['Hectare Range'] = f'{lower}-{upper if upper != float("inf") else "∞"}'
        hectare_range_data.append(chart_data)

    plot_df = pd.DataFrame(hectare_range_data)

    if 'toggle' in toggle_state:
        # Stacked bar chart for the toggle state
        fig = px.bar(plot_df, x='Hectare Range', y=common_types + ['Otra maquinaria o equipo'],
                     title='Cantidad de Maquinarias y Equipos por Rango de Hectáreas Cultivadas',
                     text_auto=True,
                     barmode='stack')
    else:
        # Pie chart for the default state
        total_counts = plot_df[common_types + ['Otra maquinaria o equipo']].sum()
        fig = px.pie(names=total_counts.index, values=total_counts.values,
                     title='Cantidad de Maquinarias y Equipos')

    return fig

@app.callback(
    Output('maquinaria-obtencion-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_maquinaria_obtencion_chart(departamento, provincia, distrito, toggle_state, region):
    # Filter based on selected area
    filtered_df = merged_df2.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]

    # Convert 'P1229' to categorical
    obtencion_map = {
    'Propia?': 'Propia',
    'Propio, de la organización?': 'De la Organización',
    'Alquilada?': 'Alquilada',
    'Alquilada de la Dirección Regional de Agricultura?': 'Alquilada de DRA',
    'Cedido?': 'Cedido'
    }
    
    filtered_df['P1229'] = filtered_df['P1229'].map(obtencion_map).astype('category')

    # Convert 'P1228_N' to numeric
    filtered_df['P1228_N'] = pd.to_numeric(filtered_df['P1228_N'], errors='coerce').fillna(0)

    # Group by unique production units and get the first 'Total_Ha' for each
    unique_productions = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA'])['Total_Ha'].first().reset_index()

    hectare_range_data = []
    for lower, upper in hectare_ranges:
        # Select units within the current hectare range
        range_productions = unique_productions[
            (unique_productions['Total_Ha'] >= lower) & (unique_productions['Total_Ha'] < upper)]
        # Merge back to the main DataFrame to get data for selected units
        range_df = pd.merge(range_productions, filtered_df, on=['CONGLOMERADO', 'NSELUA', 'UA'], how='left')
        # Multiply each 'FACTOR' by 'P1228_N' and sum up for each category within the hectare range
        range_df['Weighted_COUNT'] = range_df['FACTOR'] * range_df['P1228_N']
        category_counts = range_df.groupby('P1229')['Weighted_COUNT'].sum()

        # Prepare data for the chart
        chart_data = category_counts.to_dict()
        chart_data['Hectare Range'] = f'{lower}-{upper if upper != float("inf") else "∞"}'
        hectare_range_data.append(chart_data)

    plot_df = pd.DataFrame(hectare_range_data)

    if 'toggle' in toggle_state:
        # Stacked bar chart for the toggle state
        fig = px.bar(plot_df, x='Hectare Range', y=list(obtencion_map.values()),
             title='¿Cómo obtuvo la maquinaria y equipo? Por Rango de Hectáreas Cultivadas',
             text_auto=True,
             barmode='stack')
    else:
        # Pie chart for the default state
        total_counts = plot_df[list(obtencion_map.values())].sum()
        fig = px.pie(names=total_counts.index, values=total_counts.values,
                     title='¿Cómo obtuvo la maquinaria y equipo?')

    return fig

@app.callback(
    Output('sales-proportion-table', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_sales_proportion_table(departamento, provincia, distrito, region):
    # Filter based on selected area
    filtered_df = df.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]

    # Ensure venta_kg and total_kg are numeric and replace NaN and infinities
    filtered_df['venta_kg'] = pd.to_numeric(filtered_df['venta_kg'], errors='coerce').fillna(0)
    filtered_df['total_kg'] = pd.to_numeric(filtered_df['total_kg'], errors='coerce').fillna(0)

    # Group by 'CONGLOMERADO', 'NSELUA', 'UA'
    unique_productions = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA']).agg({
        'Total_Ha': 'first',
        'FACTOR': 'first',
        'venta_kg': 'sum',
        'total_kg': 'sum'
    }).reset_index()

    # Calculate the proportion of sales and handle divide by zero by replacing with NaN
    unique_productions['sales_proportion'] = (
        unique_productions['venta_kg'] / unique_productions['total_kg'].replace(0, np.nan)
    ) * 100

    # Replace NaN with 0 after proportion calculation
    unique_productions['sales_proportion'].fillna(0, inplace=True)

    # Define hectare ranges and create 'Hectare Range' column
    hectare_range_bins = [0] + [upper for _, upper in hectare_ranges]  # Define bins based on hectare ranges
    unique_productions['Hectare Range'] = pd.cut(unique_productions['Total_Ha'], bins=hectare_range_bins, include_lowest=True, right=False)

    # Create bins for the sales proportion
    proportion_bins = [0, 10, 20, 30, 50, 75, np.inf]
    bins = pd.cut(unique_productions['sales_proportion'], bins=proportion_bins, include_lowest=True, right=False)

    # Group by hectare range and sales proportion bins, and sum the 'FACTOR' values
    table_df = unique_productions.groupby(['Hectare Range', bins])['FACTOR'].sum().unstack(fill_value=0)

    # Convert counts to percentages
    table_df_percentage = table_df.div(table_df.sum(axis=1), axis=0) * 100

    # Convert Interval objects to strings
    # Convert Interval objects to strings
    table_df_percentage.columns = [f'{int(interval.left)}-{int(interval.right) if interval.right != np.inf else "+"}%' for interval in table_df_percentage.columns]

    # Create a table figure
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Tamaño parcelas (ha)"] + list(table_df_percentage.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[table_df_percentage.index.astype(str)] + [table_df_percentage[col].round(2).astype(str) + '%' for col in table_df_percentage.columns],
                fill_color='lavender',
                align='left'))
    ])

    fig.update_layout(
        title="Porcentaje de la producción agrícola vendida (kg) en 2022 por tamaño de parcelas",
        title_x=0.5
    )

    return fig


@app.callback(
    Output('problemas-cultivo-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('cultivo-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_problemas_cultivo_chart(selected_departamento, selected_provincia, selected_distrito, selected_cultivo, toggle_state, region):
    # Filter based on selections
    filtered_df = df.copy()
    for dropdown, column in zip([selected_departamento, selected_provincia, selected_distrito, selected_cultivo, region],
                                ['NOMBREDD', 'NOMBREPV', 'NOMBREDI', 'P204_NOM', 'REGION']):
        if dropdown:
            filtered_df = filtered_df[filtered_df[column] == dropdown]


    # Group and aggregate data
    agg_df = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA']).agg({
        'Total_Ha': 'first',
        'FACTOR': 'first',
        'P223A': lambda x: (x == 'Sí').any()  # Check if any 'Sí' in the group
    }).reset_index()
    agg_df.rename(columns={'P223A': 'Tuvo problemas'}, inplace=True)

    # Initialize data structure for hectare range information
    hectare_range_data = {range_label: {'Tuvo problemas': 0, 'No tuvo problemas': 0} for range_label in hectare_ranges}

    # Categorize each production unit and sum 'FACTOR' for each category
    for _, row in agg_df.iterrows():
        total_ha = row['Total_Ha']
        for (lower, upper) in hectare_ranges:
            if lower <= total_ha < upper:
                if row['Tuvo problemas']:
                    hectare_range_data[(lower, upper)]['Tuvo problemas'] += row['FACTOR']
                else:
                    hectare_range_data[(lower, upper)]['No tuvo problemas'] += row['FACTOR']
                break

    # Prepare DataFrame for visualization
    plot_df = pd.DataFrame([
        {**{'Hectare Range': f'{lower}-{upper}'}, **counts}
        for (lower, upper), counts in hectare_range_data.items()
    ])
    

    # Generate the figure based on toggle state
    if 'toggle' in toggle_state:
        # Stacked bar chart for the toggle state
        fig = px.bar(plot_df, x='Hectare Range',
                     y=['Tuvo problemas', 'No tuvo problemas'],
                     title="Unidades Agropecuarias que tuvieron al menos un cultivo afectado, por rango de hectáreas",
                     text_auto=True,
                     barmode='stack')
    else:
        # Pie chart for the default state
        total_counts = plot_df[['Tuvo problemas', 'No tuvo problemas']].sum()
        fig = px.pie(names=total_counts.index, values=total_counts.values,
                     title='Unidades Agropecuarias que tuvieron al menos un cultivo afectado')

    return fig

@app.callback(
    Output('crops-affected-reasons-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('cultivo-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_crops_affected_reasons_chart(selected_departamento, selected_provincia, selected_distrito, selected_cultivo, toggle_state, region):
    # Filter based on selections
    filtered_df = df.copy()
    for dropdown, column in zip([selected_departamento, selected_provincia, selected_distrito, selected_cultivo, region],
                                ['NOMBREDD', 'NOMBREPV', 'NOMBREDI', 'P204_NOM', 'REGION']):
        if dropdown:
            filtered_df = filtered_df[filtered_df[column] == dropdown]

    # Define the mapping for reasons
    reasons_map = {
        'Sequía': 'Sequía',
        'Bajas temperaturas': 'Bajas temperaturas',
        'Heladas': 'Heladas',
        'Granizada': 'Granizada',
        'Friaje': 'Friaje',
        'Lluvias a destiempo': 'Lluvias a destiempo',
        'Plagas y enfermedades': 'Plagas y enfermedades',
        'Otro': 'Otro'
    }


    # Initialize data structure for hectare range information
    hectare_range_data = []

    for lower, upper in hectare_ranges:
        range_df = filtered_df[(filtered_df['Total_Ha'] >= lower) & (filtered_df['Total_Ha'] < upper)]

        # Initialize count dictionary
        count_dict = {'No presentó problemas': 0}
        count_dict.update({reason: 0 for reason in reasons_map.values()})

        for _, row in range_df.iterrows():
            factor = row['FACTOR']  # Get the expansion factor for the row
            if row['P223A'] == 'No':
                count_dict['No presentó problemas'] += factor  # Multiply by the factor
            else:
                # Check each reason column
                for col in ['P223B_1', 'P223B_2', 'P223B_3', 'P223B_4', 'P223B_5', 'P223B_6', 'P223B_7', 'P223B_8']:
                    reason_str = row[col]
                    if reason_str and reason_str in reasons_map:
                        count_dict[reasons_map[reason_str]] += factor  # Multiply by the factor


        # Add hectare range label
        count_dict['Hectare Range'] = f'{lower}-{upper}'
        hectare_range_data.append(count_dict)

    # Prepare DataFrame for visualization
    plot_df = pd.DataFrame(hectare_range_data)
    # Generate the figure based on toggle state
    if 'toggle' in toggle_state:
        # Stacked bar chart for the toggle state
        fig = px.bar(plot_df, x='Hectare Range',
                     y=list(plot_df.columns)[:-1],  # Exclude 'Hectare Range' from values
                     title="Número de cultivos afectados por razón y rango de hectáreas",
                     text_auto=True,
                     barmode='stack')
    else:
        # Pie chart for the default state
        total_counts = plot_df[list(plot_df.columns)[:-1]].sum()
        fig = px.pie(names=total_counts.index, values=total_counts.values,
                     title='Número de cultivos afectados por razón')

    return fig

@app.callback(
    Output('jornaleros-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_jornaleros_chart(selected_departamento, selected_provincia, selected_distrito, toggle_state, region):
    # Filter based on selections
    filtered_df = df_cap1000.copy()
    for dropdown, column in zip([selected_departamento, selected_provincia, selected_distrito, region],
                                ['NOMBREDD', 'NOMBREPV', 'NOMBREDI', 'REGION']):
        if dropdown:
            filtered_df = filtered_df[filtered_df[column] == dropdown]

    # Initialize data structure for hectare range information
    hectare_range_data = {range_label: {'Contrata Jornaleros Permanentes o Eventuales': 0, 'No contrata mano de obra': 0} for range_label in hectare_ranges}

    # Categorize each production unit and sum 'FACTOR' for each category
    for _, row in filtered_df.iterrows():
        total_ha = row['Total_Ha']
        for (lower, upper) in hectare_ranges:
            if lower <= total_ha < upper:
                if row[['P1001A_2A_1', 'P1001A_2A_2', 'P1001A_2B_1', 'P1001A_2B_2']].gt(0).any():
                    hectare_range_data[(lower, upper)]['Contrata Jornaleros Permanentes o Eventuales'] += row['FACTOR']
                else:
                    hectare_range_data[(lower, upper)]['No contrata mano de obra'] += row['FACTOR']
                break

    # Prepare DataFrame for visualization
    plot_df = pd.DataFrame([
        {**{'Hectare Range': f'{lower}-{upper}'}, **counts}
        for (lower, upper), counts in hectare_range_data.items()
    ])

    # Generate the figure based on toggle state
    if 'toggle' in toggle_state:
        # Stacked bar chart for the toggle state
        fig = px.bar(plot_df, x='Hectare Range',
                     y=['Contrata Jornaleros Permanentes o Eventuales', 'No contrata mano de obra'],
                     title="Unidades Agropecuarias que contratan jornaleros, por rango de hectáreas",
                     text_auto=True,
                     barmode='stack')
    else:
        # Pie chart for the default state
        total_counts = plot_df[['Contrata Jornaleros Permanentes o Eventuales', 'No contrata mano de obra']].sum()
        fig = px.pie(names=total_counts.index, values=total_counts.values,
                     title='Unidades Agropecuarias que contratan jornaleros')

    return fig

@app.callback(
    Output('jornaleros-desagregado-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_jornaleros_detail_chart(selected_departamento, selected_provincia, selected_distrito, toggle_state, region):
    # Filter based on selections
    filtered_df = df_cap1000.copy()
    for dropdown, column in zip([selected_departamento, selected_provincia, selected_distrito, region],
                                ['NOMBREDD', 'NOMBREPV', 'NOMBREDI', 'REGION']):
        if dropdown:
            filtered_df = filtered_df[filtered_df[column] == dropdown]

    # Initialize data structure for hectare range information
    categories = ['Permanente Hombre', 'Permanente Mujer', 'Permanente Ambos', 'Eventual Hombre', 'Eventual Mujer', 'Eventual Ambos']
    hectare_range_data = {range_label: {category: 0 for category in categories} for range_label in hectare_ranges}

    # Categorize each production unit and sum 'FACTOR' for each category
    for _, row in filtered_df.iterrows():
        total_ha = row['Total_Ha']
        for (lower, upper) in hectare_ranges:
            if lower <= total_ha < upper:
                if row['P1001A_2A_1'] > 0 and row['P1001A_2A_2'] > 0:
                    hectare_range_data[(lower, upper)]['Permanente Ambos'] += row['FACTOR']
                elif row['P1001A_2A_1'] > 0:
                    hectare_range_data[(lower, upper)]['Permanente Hombre'] += row['FACTOR']
                elif row['P1001A_2A_2'] > 0:
                    hectare_range_data[(lower, upper)]['Permanente Mujer'] += row['FACTOR']

                if row['P1001A_2B_1'] > 0 and row['P1001A_2B_2'] > 0:
                    hectare_range_data[(lower, upper)]['Eventual Ambos'] += row['FACTOR']
                elif row['P1001A_2B_1'] > 0:
                    hectare_range_data[(lower, upper)]['Eventual Hombre'] += row['FACTOR']
                elif row['P1001A_2B_2'] > 0:
                    hectare_range_data[(lower, upper)]['Eventual Mujer'] += row['FACTOR']
                break

    # Prepare DataFrame for visualization
    plot_df = pd.DataFrame([
        {**{'Hectare Range': f'{lower}-{upper}'}, **counts}
        for (lower, upper), counts in hectare_range_data.items()
    ])

    # Generate the figure based on toggle state
    if 'toggle' in toggle_state:
        # Stacked bar chart for the toggle state
        fig = px.bar(plot_df, x='Hectare Range',
                    y=categories,
                    title="Unidades Agropecuarias que contratan jornaleros, por rango de hectáreas",
                    text_auto=True,
                    barmode='stack')
    else:
        # Pie chart for the default state
        total_counts = plot_df[categories].sum()
        fig = px.pie(names=total_counts.index, values=total_counts.values,
                title='Unidades Agropecuarias que contratan jornaleros')

        # Update the layout of the figure
        fig.update_layout(
            autosize=True,
            margin=dict(l=50, r=50, b=100, t=100, pad=4),
            paper_bgcolor="LightSteelBlue",
        )

        return fig
    
@app.callback(
    Output('employed-people-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_employed_people_chart(selected_departamento, selected_provincia, selected_distrito, toggle_state, region):
    # Filter based on selections
    filtered_df = df_cap1000.copy()
    for dropdown, column in zip([selected_departamento, selected_provincia, selected_distrito, region],
                                ['NOMBREDD', 'NOMBREPV', 'NOMBREDI', 'REGION']):
        if dropdown:
            filtered_df = filtered_df[filtered_df[column] == dropdown]

    # Calculate the amount of people employed
    filtered_df['People Employed'] = filtered_df[['P1001A_2A_1C', 'P1001A_2A_2C', 'P1001A_2B_1C', 'P1001A_2B_2C']].fillna(0).sum(axis=1) * filtered_df['FACTOR']
    # Prepare DataFrame for visualization
    hectare_range_data = {}
    for lower, upper in hectare_ranges:
        range_df = filtered_df[(filtered_df['Total_Ha'] >= lower) & (filtered_df['Total_Ha'] < upper)]
        hectare_range_data[f'{lower}-{upper}'] = range_df['People Employed'].sum()

    plot_df = pd.DataFrame(list(hectare_range_data.items()), columns=['Hectare Range', 'People Employed'])

    
    fig = px.bar(plot_df, x='Hectare Range', y='People Employed', title="Personas totales empleadas por rango de hectáreas", text_auto=True)
   
    return fig

@app.callback(
    Output('employed-peoples-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_employed_peoples_chart(selected_departamento, selected_provincia, selected_distrito, toggle_state, region):
    # Filter based on selections
    filtered_df = df_cap1000.copy()
    for dropdown, column in zip([selected_departamento, selected_provincia, selected_distrito, region],
                                ['NOMBREDD', 'NOMBREPV', 'NOMBREDI', 'REGION']):
        if dropdown:
            filtered_df = filtered_df[filtered_df[column] == dropdown]

    # Calculate the amount of people employed for each type of worker
    worker_types = ['P1001A_2A_1C', 'P1001A_2A_2C', 'P1001A_2B_1C', 'P1001A_2B_2C']
    worker_names = ['Permanente Hombre', 'Permanente Mujer', 'Eventual Hombre', 'Eventual Mujer']
    for worker_type, worker_name in zip(worker_types, worker_names):
        filtered_df[worker_name] = filtered_df[worker_type].fillna(0) * filtered_df['FACTOR']

    # Prepare DataFrame for visualization
    hectare_range_data = []
    for lower, upper in hectare_ranges:
        range_df = filtered_df[(filtered_df['Total_Ha'] >= lower) & (filtered_df['Total_Ha'] < upper)]
        for worker_name in worker_names:
            sum_people = range_df[worker_name].sum()
            hectare_range_data.append({'Hectare Range': f'{lower}-{upper}', 'Worker Type': worker_name, 'People Employed': sum_people})

    plot_df = pd.DataFrame(hectare_range_data)

    # Generate the figure based on toggle state
    if 'toggle' in toggle_state:
        # Stacked bar chart for the toggle state
        fig = px.bar(plot_df, x='Hectare Range', y='People Employed', color='Worker Type', title="Personas empleadas por rango de hectáreas", barmode='stack')
    else:
        # Pie chart for the default state
        fig = px.pie(plot_df, names='Worker Type', values='People Employed', title='Personas empleadas por tipo de trabajador')

    return fig

@app.callback(
    Output('venta-contribution-table', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('cultivo-dropdown', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_venta_contribution_table(departamento, provincia, distrito, cultivo, region):
    # Filter based on selected area
    filtered_df = df.copy()

    if cultivo:
        filtered_df = filtered_df[filtered_df['P204_NOM'] == cultivo].reset_index(drop=True)
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region].reset_index(drop=True)
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]

    # Ensure venta_kg and total_kg are numeric and replace NaN and infinities
    filtered_df['venta_kg'] = pd.to_numeric(filtered_df['venta_kg'], errors='coerce').fillna(0)
    filtered_df['total_kg'] = pd.to_numeric(filtered_df['total_kg'], errors='coerce').fillna(0)

    # Group by 'CONGLOMERADO', 'NSELUA', 'UA'
    unique_productions = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA']).agg({
        'Total_Ha': 'first',
        'venta_kg': 'sum',
        'total_kg': 'sum'
    }).reset_index()

    # Calculate the proportion of sales to total
    unique_productions['sales_proportion'] = (unique_productions['venta_kg'] / unique_productions['total_kg'].replace(0, np.nan))

    # Define categories based on Total_Ha and sales_proportion
    conditions = [
        (unique_productions['Total_Ha'] < 5) & (unique_productions['sales_proportion'] < 0.5),
        (unique_productions['Total_Ha'] < 5) & (unique_productions['sales_proportion'] >= 0.5),
        (unique_productions['Total_Ha'] >= 5) & (unique_productions['Total_Ha'] < 20),
        (unique_productions['Total_Ha'] >= 20)
    ]
    # Define ordered categories
    ordered_categories = ['<5 ha & <50% ventas', '<5 ha & >=50% ventas', '5-20 ha', '>=20 ha']

    # Define the 'Category' column based on your conditions
    unique_productions['Category'] = np.select(conditions, ordered_categories, default='Unknown')


    # Filter out the 'Unknown' category and order the categories
    unique_productions = unique_productions[unique_productions['Category'] != 'Unknown']
    unique_productions['Category'] = pd.Categorical(unique_productions['Category'], ordered=True, categories=ordered_categories)
    unique_productions = unique_productions.sort_values('Category')

    # Filter rows for local market
    local_market_df = unique_productions[filtered_df['P223_1'].str.contains('Mercado local', na=False)]

    # Calculate total venta_kg for each category
    total_venta_kg = unique_productions.groupby('Category')['venta_kg'].sum()
    local_market_venta_kg = local_market_df.groupby('Category')['venta_kg'].sum()

    # Calculate the overall total of venta_kg
    overall_total_venta_kg = total_venta_kg.sum()
    overall_local_market_venta_kg = local_market_venta_kg.sum()

    # Calculate the percentage contribution for each category
    total_percentages = (total_venta_kg / overall_total_venta_kg * 100).fillna(0).round(2)
    local_market_percentages = (local_market_venta_kg / overall_local_market_venta_kg * 100).fillna(0).round(2)

    # Prepare data for the table
    table_data = {
        'Categoría': total_percentages.index,
        'Ventas totales (%)': total_percentages.astype(str) + '%',
        'Ventas en Mercado Local (%)': local_market_percentages.reindex(total_percentages.index, fill_value=0).astype(str) + '%'
    }

    # Create a table figure without 'Unknown' category
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Categoría', 'Ventas totales (%)', 'Ventas en Mercado Local (%)'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[table_data[key] for key in table_data if key != 'Unknown'],  # Exclude 'Unknown' data
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(
        title="Porcentaje de contribución a las ventas totales en el país y mercado local",
        title_x=0.5
    )

    return fig

@app.callback(
    Output('proportions-category-table', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('cultivo-dropdown', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_proportion_category_table(departamento, provincia, distrito, cultivo, region):
    # Filter based on selected area
    filtered_df = df.copy()

    if cultivo:
        filtered_df = filtered_df[filtered_df['P204_NOM'] == cultivo].reset_index(drop=True)
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region].reset_index(drop=True)
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    
    

    # Ensure venta_kg and total_kg are numeric and replace NaN and infinities
    filtered_df['venta_kg'] = pd.to_numeric(filtered_df['venta_kg'], errors='coerce').fillna(0)
    filtered_df['total_kg'] = pd.to_numeric(filtered_df['total_kg'], errors='coerce').fillna(0)

    # Group by 'CONGLOMERADO', 'NSELUA', 'UA'
    unique_productions = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA']).agg({
        'Total_Ha': 'first',
        'venta_kg': 'sum',
        'total_kg': 'sum'
    }).reset_index()

    # Calculate the proportion of sales to total
    unique_productions['sales_proportion'] = (unique_productions['venta_kg'] / unique_productions['total_kg'].replace(0, np.nan))

    # Define categories based on Total_Ha and sales_proportion
    conditions = [
        (unique_productions['Total_Ha'] < 5) & (unique_productions['sales_proportion'] < 0.3),
        (unique_productions['Total_Ha'] < 5) & (unique_productions['sales_proportion'] >= 0.3),
        (unique_productions['Total_Ha'] >= 5) & (unique_productions['Total_Ha'] < 20),
        (unique_productions['Total_Ha'] >= 20)
    ]
    # Define ordered categories
    ordered_categories = ['<5 ha & <30% ventas', '<5 ha & >=30% ventas', '5-20 ha', '>=20 ha']

    # Define the 'Category' column based on your conditions
    unique_productions['Category'] = np.select(conditions, ordered_categories, default='Unknown')


    # Filter out the 'Unknown' category and order the categories
    unique_productions = unique_productions[unique_productions['Category'] != 'Unknown']
    unique_productions['Category'] = pd.Categorical(unique_productions['Category'], ordered=True, categories=ordered_categories)
    unique_productions = unique_productions.sort_values('Category')

    # Filter rows for local market
    local_market_df = unique_productions[filtered_df['P223_1'].str.contains('Mercado local', na=False)]

    # Calculate total venta_kg for each category
    total_venta_kg = unique_productions.groupby('Category')['venta_kg'].sum()
    local_market_venta_kg = local_market_df.groupby('Category')['venta_kg'].sum()

    # Calculate the overall total of venta_kg
    overall_total_venta_kg = total_venta_kg.sum()
    overall_local_market_venta_kg = local_market_venta_kg.sum()

    # Calculate the percentage contribution for each category
    total_percentages = (total_venta_kg / overall_total_venta_kg * 100).fillna(0).round(2)
    local_market_percentages = (local_market_venta_kg / overall_local_market_venta_kg * 100).fillna(0).round(2)

    # Prepare data for the table
    table_data = {
        'Categoría': total_percentages.index,
        'Ventas totales (%)': total_percentages.astype(str) + '%',
        'Ventas en Mercado Local (%)': local_market_percentages.reindex(total_percentages.index, fill_value=0).astype(str) + '%'
    }

    # Create a table figure without 'Unknown' category
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Categoría', 'Ventas totales (%)', 'Ventas en Mercado Local (%)'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[table_data[key] for key in table_data if key != 'Unknown'],  # Exclude 'Unknown' data
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(
        title="Porcentaje de contribución a las ventas totales en el país y mercado local",
        title_x=0.5
    )

    return fig


filtered_df = df.copy()

# Ensure venta_kg and total_kg are numeric and replace NaN and infinities
filtered_df['venta_kg'] = pd.to_numeric(filtered_df['venta_kg'], errors='coerce').fillna(0)
filtered_df['total_kg'] = pd.to_numeric(filtered_df['total_kg'], errors='coerce').fillna(0)

# Group by 'CONGLOMERADO', 'NSELUA', 'UA'
unique_productions = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA']).agg({
    'Total_Ha': 'first',
    'venta_kg': 'sum',
    'total_kg': 'sum'
}).reset_index()

# Calculate the proportion of sales to total
unique_productions['sales_proportion'] = unique_productions['venta_kg'] / unique_productions['total_kg'].replace(0, np.nan)

# Perform an inner merge to include only the rows that match
merged_df2 = pd.merge(merged_df2, unique_productions[['CONGLOMERADO', 'NSELUA', 'UA', 'sales_proportion']],
                      on=['CONGLOMERADO', 'NSELUA', 'UA'],
                      how='inner')

@app.callback(
    Output('maquinaria1', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_proportions_owned_maquinaria_table(departamento, provincia, distrito, region):
    # Assume merged_df2 is already merged with sales_proportion as shown in previous steps

    # Filter based on selected area and ownership
    filtered_df = merged_df2.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]

    # Filter only rows with 'Propia?' in column 'P1229'
    filtered_df = filtered_df[filtered_df['P1229'] == 'Propia?']

    # Ensure 'P1228_N' and 'FACTOR' are numeric for multiplication
    filtered_df['P1228_N'] = pd.to_numeric(filtered_df['P1228_N'], errors='coerce').fillna(0)
    filtered_df['FACTOR'] = pd.to_numeric(filtered_df['FACTOR'], errors='coerce').fillna(0)

    # Define categories based on Total_Ha and sales_proportion
    conditions = [
        (filtered_df['Total_Ha'] < 5) & (filtered_df['sales_proportion'] < 0.3),
        (filtered_df['Total_Ha'] < 5) & (filtered_df['sales_proportion'] >= 0.3),
        (filtered_df['Total_Ha'] >= 5) & (filtered_df['Total_Ha'] < 20),
        (filtered_df['Total_Ha'] >= 20)
    ]
    ordered_categories = ['<5 ha & <30% ventas', '<5 ha & >=30% ventas', '5-20 ha', '>=20 ha']

    filtered_df['Category'] = np.select(conditions, ordered_categories, default='Unknown')

    # Exclude 'Unknown' category
    filtered_df = filtered_df[filtered_df['Category'] != 'Unknown']

    # Specify the order of categories
    filtered_df['Category'] = pd.Categorical(filtered_df['Category'], categories=ordered_categories, ordered=True)

    filtered_df['Total_Factor_P1228_N'] = filtered_df['FACTOR'] * filtered_df['P1228_N']

    # Group the data by 'Category' and 'P1228_TIPO' and sum 'Total_Factor_P1228_N'
    category_sum = filtered_df.groupby(['Category', 'P1228_TIPO'])['Total_Factor_P1228_N'].sum().unstack(fill_value=0)

    # Calculate the total count for owned Maquinaria and Equipo
    category_sum['Total Owned Maquinaria'] = category_sum.get('Maquinaria', 0)
    category_sum['Total Owned Equipo'] = category_sum.get('Equipo', 0)

    # Calculate the percentages for owned Maquinaria and Equipo
    total_owned_maquinaria = category_sum['Total Owned Maquinaria'].sum()
    total_owned_equipo = category_sum['Total Owned Equipo'].sum()
    category_sum['Maquinaria %'] = (category_sum['Total Owned Maquinaria'] / total_owned_maquinaria * 100).round(2)
    category_sum['Equipo %'] = (category_sum['Total Owned Equipo'] / total_owned_equipo * 100).round(2)

    # Prepare data for the table
    table_data = {
        'Categoría': category_sum.index,
        'Cantidad de Maquinaria Propia': category_sum['Total Owned Maquinaria'].astype(int),
        'Maquinaria Propia %': category_sum['Maquinaria %'].astype(str) + '%',
        'Cantidad de Equipo Propio': category_sum['Total Owned Equipo'].astype(int),
        'Equipo Propio %': category_sum['Equipo %'].astype(str) + '%'
    }

    # Create a table figure
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Categoría', 'Cantidad de Maquinaria Propia', 'Maquinaria Propia %', 'Cantidad de Equipo Propio', 'Equipo Propio %'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[table_data[key] for key in table_data],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(
        title="Porcentaje de maquinarias y equipos agrícolas propios por categoría de hectáreas y proporción de ventas",
        title_x=0.5
    )

    return fig

@app.callback(
    Output('maquinaria2', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_maquinaria_sales_proportion_table(departamento, provincia, distrito, region):
    # Filter based on selected area, assume merged_df2 already has sales_proportion merged in it
    filtered_df = merged_df2.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]

    # Ensure 'P1228_N' and 'FACTOR' are numeric for multiplication
    filtered_df['P1228_N'] = pd.to_numeric(filtered_df['P1228_N'], errors='coerce').fillna(0)
    filtered_df['FACTOR'] = pd.to_numeric(filtered_df['FACTOR'], errors='coerce').fillna(0)

    # Define categories based on Total_Ha and sales_proportion
    conditions = [
        (filtered_df['Total_Ha'] < 5) & (filtered_df['sales_proportion'] < 0.3),
        (filtered_df['Total_Ha'] < 5) & (filtered_df['sales_proportion'] >= 0.3),
        (filtered_df['Total_Ha'] >= 5) & (filtered_df['Total_Ha'] < 20),
        (filtered_df['Total_Ha'] >= 20)
    ]
    ordered_categories = ['<5 ha & <30% ventas', '<5 ha & >=30% ventas', '5-20 ha', '>=20 ha']

    filtered_df['Category'] = np.select(conditions, ordered_categories, default='Unknown')

    # Exclude 'Unknown' category
    filtered_df = filtered_df[filtered_df['Category'] != 'Unknown']

    # Specify the order of categories
    filtered_df['Category'] = pd.Categorical(filtered_df['Category'], categories=ordered_categories, ordered=True)

    # Calculate the total 'FACTOR' * 'P1228_N' for each type of equipment
    filtered_df['Total_Factor_P1228_N'] = filtered_df['FACTOR'] * filtered_df['P1228_N']

    # Group the data by 'Category' and 'P1228_TIPO' and sum 'Total_Factor_P1228_N'
    category_sum = filtered_df.groupby(['Category', 'P1228_TIPO'])['Total_Factor_P1228_N'].sum().unstack(fill_value=0)

    # Calculate the percentages for Maquinaria and Equipo
    total_maquinaria = category_sum.get('Maquinaria', 0).sum()
    total_equipo = category_sum.get('Equipo', 0).sum()
    category_sum['Maquinaria %'] = (category_sum.get('Maquinaria', 0) / total_maquinaria * 100).round(2)
    category_sum['Equipo %'] = (category_sum.get('Equipo', 0) / total_equipo * 100).round(2)

    # Prepare data for the table
    table_data = {
        'Categoría': category_sum.index,
        'Cantidad de Maquinaria': category_sum.get('Maquinaria', 0).astype(int),
        'Maquinaria %': category_sum['Maquinaria %'].astype(str) + '%',
        'Cantidad de Equipo': category_sum.get('Equipo', 0).astype(int),
        'Equipo %': category_sum['Equipo %'].astype(str) + '%'
    }

    # Create a table figure
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Categoría', 'Cantidad de Maquinaria', 'Maquinaria %', 'Cantidad de Equipo', 'Equipo %'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[table_data[key] for key in table_data],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(
        title="Porcentaje de tenencia de maquinarias y equipos agrícolas por categoría de hectáreas y proporción de ventas",
        title_x=0.5
    )

    return fig

@app.callback(
    Output('maquinaria3', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_proportions_owned_tractor_table(departamento, provincia, distrito, region):
    # Assume merged_df2 is already merged with sales_proportion as shown in previous steps

    # Filter based on selected area and ownership
    filtered_df = merged_df2.copy()
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBREDD'] == departamento]
    if provincia:
        filtered_df = filtered_df[filtered_df['NOMBREPV'] == provincia]
    if distrito:
        filtered_df = filtered_df[filtered_df['NOMBREDI'] == distrito]
    if region:
        filtered_df = filtered_df[filtered_df['REGION'] == region]

    
    # Filter only rows with 'Propia?' in column 'P1229'
    filtered_df = filtered_df[(filtered_df['P1229'] == 'Propia?') & (filtered_df['P1228_NOM'] == 'TRACTOR AGRICOLA')]
    
    # Ensure 'P1228_N' and 'FACTOR' are numeric for multiplication
    filtered_df['P1228_N'] = pd.to_numeric(filtered_df['P1228_N'], errors='coerce').fillna(0)
    filtered_df['FACTOR'] = pd.to_numeric(filtered_df['FACTOR'], errors='coerce').fillna(0)

    # Define categories based on Total_Ha and sales_proportion
    conditions = [
        (filtered_df['Total_Ha'] < 5) & (filtered_df['sales_proportion'] < 0.3),
        (filtered_df['Total_Ha'] < 5) & (filtered_df['sales_proportion'] >= 0.3),
        (filtered_df['Total_Ha'] >= 5) & (filtered_df['Total_Ha'] < 20),
        (filtered_df['Total_Ha'] >= 20)
    ]
    ordered_categories = ['<5 ha & <30% ventas', '<5 ha & >=30% ventas', '5-20 ha', '>=20 ha']

    filtered_df['Category'] = np.select(conditions, ordered_categories, default='Unknown')

    # Exclude 'Unknown' category
    filtered_df = filtered_df[filtered_df['Category'] != 'Unknown']

    # Specify the order of categories
    filtered_df['Category'] = pd.Categorical(filtered_df['Category'], categories=ordered_categories, ordered=True)
    
    filtered_df['Total_Factor_P1228_N'] = filtered_df['FACTOR'] * filtered_df['P1228_N']
    


    # Group the data by 'Category' and sum 'Total_Factor_P1228_N'
    category_sum = filtered_df.groupby('Category')['Total_Factor_P1228_N'].sum()

    # Calculate the total count for owned Tractores Agrícolas
    total_owned_tractores = category_sum.sum()

    # Calculate the percentages for owned Tractores Agrícolas
    category_percentage = category_sum / total_owned_tractores * 100

    # Prepare data for the table
    table_data = {
        'Categoría': category_sum.index,
        'Cantidad de Tractores Agrícolas Propios': category_sum.values.round(0).astype(int),
        'Tractores Agrícolas Propios %': np.char.add(category_percentage.values.round(2).astype(str), '%')
    }

    # Create a table figure
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Categoría', 'Cantidad de Tractores Agrícolas Propios', 'Tractores Agrícolas Propios %'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[table_data[key] for key in table_data],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(
        title="Porcentaje de tractores agrícolas propios por categoría de hectáreas y proporción de ventas",
        title_x=0.5
    )

    return fig

@app.callback(
    Output('worked-days-chart', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('chart-toggle', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_worked_days_chart(selected_departamento, selected_provincia, selected_distrito, toggle_state, region):
    # Filter based on selections
    filtered_df = df_cap1000.copy()
    for dropdown, column in zip([selected_departamento, selected_provincia, selected_distrito, region],
                                ['NOMBREDD', 'NOMBREPV', 'NOMBREDI', 'REGION']):
        if dropdown:
            filtered_df = filtered_df[filtered_df[column] == dropdown]

    # Calculate the days worked for each gender
    filtered_df['dias_hombre'] = ((filtered_df['P1001A_2A_1'].fillna(0) + filtered_df['P1001A_2B_1'].fillna(0)) * filtered_df['FACTOR']) / 40
    filtered_df['dias_mujer'] = ((filtered_df['P1001A_2A_2'].fillna(0) + filtered_df['P1001A_2B_2'].fillna(0)) * filtered_df['FACTOR']) / 30
    filtered_df['dias'] = filtered_df['dias_hombre'] + filtered_df['dias_mujer']

    # Prepare DataFrame for visualization
    hectare_range_data = []
    for lower, upper in hectare_ranges:
        range_df = filtered_df[(filtered_df['Total_Ha'] >= lower) & (filtered_df['Total_Ha'] < upper)]
        sum_dias_hombre = range_df['dias_hombre'].sum()
        sum_dias_mujer = range_df['dias_mujer'].sum()
        hectare_range_data.append({'Hectare Range': f'{lower}-{upper}', 'Gender': 'Hombre', 'Days Worked': sum_dias_hombre})
        hectare_range_data.append({'Hectare Range': f'{lower}-{upper}', 'Gender': 'Mujer', 'Days Worked': sum_dias_mujer})

    plot_df = pd.DataFrame(hectare_range_data)

    # Generate the figure based on toggle state
    if 'toggle' in toggle_state:
        # Stacked bar chart for the toggle state
        fig = px.bar(plot_df, x='Hectare Range', y='Days Worked', color='Gender', title="Días trabajados por rango de hectáreas y género", barmode='stack')
    else:
        # Pie chart for the default state, summing up all days for men and women
        total_days = {
            'Gender': ['Hombre', 'Mujer'],
            'Days Worked': [plot_df[plot_df['Gender'] == 'Hombre']['Days Worked'].sum(), plot_df[plot_df['Gender'] == 'Mujer']['Days Worked'].sum()]
        }
        fig = px.pie(pd.DataFrame(total_days), names='Gender', values='Days Worked', title='Días trabajados por género')

    return fig

filtered_df = df.copy()

# Ensure venta_kg and total_kg are numeric and replace NaN and infinities
filtered_df['venta_kg'] = pd.to_numeric(filtered_df['venta_kg'], errors='coerce').fillna(0)
filtered_df['total_kg'] = pd.to_numeric(filtered_df['total_kg'], errors='coerce').fillna(0)

# Group by 'CONGLOMERADO', 'NSELUA', 'UA'
unique_productions = filtered_df.groupby(['CONGLOMERADO', 'NSELUA', 'UA']).agg({
    'Total_Ha': 'first',
    'venta_kg': 'sum',
    'total_kg': 'sum'
}).reset_index()

# Calculate the proportion of sales to total
unique_productions['sales_proportion'] = unique_productions['venta_kg'] / unique_productions['total_kg'].replace(0, np.nan)

# Perform an inner merge to include only the rows that match
df_cap1000 = pd.merge(df_cap1000, unique_productions[['CONGLOMERADO', 'NSELUA', 'UA', 'sales_proportion']],
                      on=['CONGLOMERADO', 'NSELUA', 'UA'],
                      how='inner')

@app.callback(
    Output('dias-contribution-table', 'figure'),
    [Input('departamento-dropdown', 'value'),
     Input('provincia-dropdown', 'value'),
     Input('distrito-dropdown', 'value'),
     Input('region-dropdown', 'value')]
)
@cache.memoize(timeout=60 * 60)
def update_dias_contribution_table(departamento, provincia, distrito, region):
    # Filter based on selected area
    filtered_df = df_cap1000.copy()
    for dropdown, column in zip([departamento, provincia, distrito, region],
                                ['NOMBREDD', 'NOMBREPV', 'NOMBREDI', 'REGION']):
        if dropdown:
            filtered_df = filtered_df[filtered_df[column] == dropdown]

    # Ensure FACTOR is numeric for multiplication
    filtered_df['FACTOR'] = pd.to_numeric(filtered_df['FACTOR'], errors='coerce').fillna(0)

    # Calculate days for each type of worker
    filtered_df['dias_hombre'] = ((filtered_df['P1001A_2A_1'].fillna(0) + filtered_df['P1001A_2B_1'].fillna(0)) * filtered_df['FACTOR']) / 40000
    filtered_df['dias_mujer'] = ((filtered_df['P1001A_2A_2'].fillna(0) + filtered_df['P1001A_2B_2'].fillna(0)) * filtered_df['FACTOR']) / 30000
    filtered_df['dias'] = filtered_df['dias_hombre'] + filtered_df['dias_mujer']

    # Define categories based on Total_Ha and sales_proportion
    conditions = [
        (filtered_df['Total_Ha'] < 5) & (filtered_df['sales_proportion'] < 0.3),
        (filtered_df['Total_Ha'] < 5) & (filtered_df['sales_proportion'] >= 0.3),
        (filtered_df['Total_Ha'] >= 5) & (filtered_df['Total_Ha'] < 20),
        (filtered_df['Total_Ha'] >= 20)
    ]
    ordered_categories = ['<5 ha & <30% ventas', '<5 ha & >=30% ventas', '5-20 ha', '>=20 ha']

    filtered_df['Category'] = np.select(conditions, ordered_categories, default='Unknown')

    # Exclude 'Unknown' category
    filtered_df = filtered_df[filtered_df['Category'] != 'Unknown']

    # Specify the order of categories
    filtered_df['Category'] = pd.Categorical(filtered_df['Category'], categories=ordered_categories, ordered=True)

    # Group the data by 'Category' and sum days
    category_sum = filtered_df.groupby('Category')[['dias_hombre', 'dias_mujer', 'dias']].sum()

    # Calculate the percentages for dias, dias_hombre, and dias_mujer
    total_dias = category_sum['dias'].sum()
    total_dias_hombre = category_sum['dias_hombre'].sum()
    total_dias_mujer = category_sum['dias_mujer'].sum()
    category_sum['dias %'] = (category_sum['dias'] / total_dias * 100).round(2)
    category_sum['dias_hombre %'] = (category_sum['dias_hombre'] / total_dias_hombre * 100).round(2)
    category_sum['dias_mujer %'] = (category_sum['dias_mujer'] / total_dias_mujer * 100).round(2)

    # Prepare data for the table
    table_data = {
        'Categoría': category_sum.index,
        'Miles de Días Totales': category_sum['dias'].astype(int),
        'Días Totales %': category_sum['dias %'].astype(str) + '%',
        'Miles de Días Hombre': category_sum['dias_hombre'].astype(int),
        'Días Hombre %': category_sum['dias_hombre %'].astype(str) + '%',
        'Miles de Días Mujer': category_sum['dias_mujer'].astype(int),
        'Días Mujer %': category_sum['dias_mujer %'].astype(str) + '%'
    }

    # Create a table figure
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(table_data.keys()),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[table_data[key] for key in table_data.keys()],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(
        title="Contribución de días (en miles) trabajados generado por Unidades Agropecuarias, por categoría de hectáreas y proporción de ventas",
        title_x=0.5
    )

    return fig

# Ejecutar aplicación
#if __name__ == '__main__':
    #app.run_server(debug=True)
