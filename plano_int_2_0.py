# -*- coding: utf-8 -*-
"""
Script de Generación de Plano Interactivo - Versión GitHub Actions
Adaptado para lectura automática de Drive y Sheets
"""

import os
import json
import io
import cv2
import numpy as np
import pandas as pd
import gspread
import folium
import re
import unicodedata
import openpyxl
from matplotlib import pyplot as plt
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from branca.element import Template, MacroElement

# ==========================================
# 1. AUTENTICACIÓN Y CONEXIÓN (Modo Robot)
# ==========================================

# Definir los permisos
scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

print("--- INICIANDO PROCESO DE AUTENTICACIÓN ---")

# Detectar credenciales de GitHub
if 'GDRIVE_CREDENTIALS' in os.environ:
    print("Detectado entorno GitHub Actions. Leyendo secretos...")
    creds_dict = json.loads(os.environ['GDRIVE_CREDENTIALS'])
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
elif os.path.exists('credentials.json'):
    # Para pruebas locales si tienes el archivo
    creds = Credentials.from_service_account_file('credentials.json', scopes=scopes)
else:
    raise Exception("No se encontraron credenciales (GDRIVE_CREDENTIALS o credentials.json)")

# Conectar clientes
gc = gspread.authorize(creds)
drive_service = build('drive', 'v3', credentials=creds)

# ==========================================
# 2. FUNCIÓN PARA DESCARGAR ARCHIVOS DE DRIVE
# ==========================================

def descargar_archivo_drive(nombre_archivo, destino_local):
    """Busca un archivo en Drive por nombre y lo descarga."""
    print(f"🔍 Buscando '{nombre_archivo}' en Google Drive...")
    
    # Buscar el archivo (excluyendo los que están en la papelera)
    query = f"name = '{nombre_archivo}' and trashed = false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        print(f"❌ ERROR: No se encontró el archivo '{nombre_archivo}'.")
        return False

    # Tomar el primero que encuentre
    file_id = items[0]['id']
    print(f"⬇️ Descargando archivo (ID: {file_id})...")

    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(destino_local, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        # print(f"Descarga {int(status.progress() * 100)}%.")
    
    print(f"✅ Archivo descargado exitosamente: {destino_local}")
    return True

# ==========================================
# 3. DESCARGA DE RECURSOS NECESARIOS
# ==========================================

# Descargar la imagen del plano
if not descargar_archivo_drive('plano.png', 'plano.png'):
    raise Exception("No se pudo descargar el plano. Verifica que se llame 'plano.png' en Drive.")

# NOTA: Si tu código usa un Excel base, asegúrate de descargarlo aquí también.
# Por ejemplo: descargar_archivo_drive('Base de Datos.xlsx', 'Base de Datos.xlsx')

# ==========================================
# 4. PROCESAMIENTO DE IMAGEN (OpenCV)
# ==========================================

print("--- PROCESANDO IMAGEN DEL PLANO ---")
# Cargar la imagen descargada
img = cv2.imread('plano.png')
if img is None:
    raise Exception("Error al leer 'plano.png'. El archivo puede estar corrupto.")

h, w, _ = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Umbralización para detectar bloques negros (ajustado según tu código original)
_, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

# Encontrar contornos
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Se detectaron {len(contours)} posibles contornos (viviendas/bloques).")

# ==========================================
# 5. OBTENCIÓN DE DATOS (Google Sheets)
# ==========================================

print("--- CONECTANDO A GOOGLE SHEETS ---")

# 5.1 Cargar Observaciones (Pre F1)
try:
    nombre_hoja_obs = 'Pre F1'
    print(f"Abriendo hoja de observaciones: {nombre_hoja_obs}")
    sh_obs = gc.open(nombre_hoja_obs)
    
    dict_observaciones = {} # Clave: Manzana-Lote (ej: "A-1"), Valor: "Con Observaciones"
    
    # Recorrer pestañas que parezcan Manzanas (MZ)
    for worksheet in sh_obs.worksheets():
        nombre_hoja = worksheet.title.upper()
        if "MZ" in nombre_hoja:
            letra_mz = nombre_hoja.replace("MZ", "").replace(".","").strip() # Ej: "A"
            
            # Obtener todos los datos de la hoja
            datos = worksheet.get_all_values()
            
            # Asumimos estructura: Col A (Lote), Col E (Estado/Obs) - Ajusta índices si cambió
            # Índice 0 = Columna A, Índice 4 = Columna E
            for fila in datos[1:]: # Saltar encabezado
                if len(fila) > 4: 
                    lote = str(fila[0]).strip()
                    estado = str(fila[4]).strip().lower() # Columna E
                    
                    if lote and (estado == "no" or "obs" in estado):
                        clave = f"{letra_mz}-{lote}" # Ej: A-1
                        dict_observaciones[clave] = "Con Observaciones"

    print(f"Observaciones cargadas: {len(dict_observaciones)} lotes con problemas.")

except Exception as e:
    print(f"⚠️ Advertencia leyendo observaciones: {e}")
    dict_observaciones = {}

# 5.2 Cargar Avances (Partidas)
try:
    nombre_hoja_avances = 'Partidas'
    print(f"Abriendo hoja de avances: {nombre_hoja_avances}")
    sh_avances = gc.open(nombre_hoja_avances)
    ws_resumen = sh_avances.worksheet("Resumen de Avance")
    
    # Leer todo el dataframe
    data_avances = pd.DataFrame(ws_resumen.get_all_records())
    
    # Limpieza básica de nombres de columnas
    data_avances.columns = [str(c).strip() for c in data_avances.columns]
    
    # Crear diccionario de avances
    # Buscamos columnas 'Manzana', 'Lote' y 'Avance Real' (ajusta nombres según tu Excel real)
    dict_avances = {}
    
    # Intentar identificar las columnas correctas
    col_mz = next((c for c in data_avances.columns if "manz" in c.lower()), None)
    col_lote = next((c for c in data_avances.columns if "lote" in c.lower() or "vivienda" in c.lower()), None)
    col_avance = next((c for c in data_avances.columns if "real" in c.lower() or "avance" in c.lower()), None)

    if col_mz and col_lote and col_avance:
        for index, row in data_avances.iterrows():
            mz = str(row[col_mz]).strip()
            lote = str(row[col_lote]).strip()
            avance = row[col_avance]
            
            # Normalizar avance a float
            if isinstance(avance, str):
                avance = avance.replace('%', '').replace(',', '.')
            try:
                avance_val = float(avance)
            except:
                avance_val = 0.0
                
            clave = f"{mz}-{lote}" # Ej: A-1
            dict_avances[clave] = avance_val
    else:
        print("❌ No se encontraron las columnas de Manzana, Lote o Avance en 'Resumen de Avance'")

    print(f"Avances cargados: {len(dict_avances)} registros.")

except Exception as e:
    print(f"❌ Error crítico leyendo avances: {e}")
    dict_avances = {}

# ==========================================
# 6. GENERACIÓN DEL MAPA (Folium)
# ==========================================

print("--- GENERANDO MAPA INTERACTIVO ---")

# Crear mapa centrado (Coordenadas arbitrarias para visualización plana)
m = folium.Map(location=[0, 0], zoom_start=18, crs='Simple', tiles=None)

# Añadir la imagen del plano como capa base
folium.raster_layers.ImageOverlay(
    image='plano.png',
    bounds=[[0, 0], [h, w]],
    opacity=1.0,
    name="Plano Base"
).add_to(m)

# Grupo de capas para las viviendas
geo_layer = folium.FeatureGroup(name="Viviendas")

# Función para determinar color según avance y observaciones
def obtener_color(avance, tiene_obs):
    if tiene_obs:
        return '#e74c3c' # Rojo (Observación)
    if avance >= 100:
        return '#2ecc71' # Verde (Listo)
    if avance > 0:
        return '#f1c40f' # Amarillo (En proceso)
    return '#95a5a6' # Gris (Sin inicio)

# Procesar contornos y dibujar polígonos
conteo_mapeados = 0
for cnt in contours:
    # Filtrar contornos muy pequeños (ruido) o muy grandes (marco)
    area = cv2.contourArea(cnt)
    if area < 500 or area > 50000: 
        continue

    # Aproximar polígono para suavizar bordes
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # Convertir coordenadas para Folium (Imagen: Y aumenta hacia abajo, Mapa: Y aumenta hacia arriba)
    # Folium Simple CRS: [y, x] pero invirtiendo Y respecto a la imagen
    puntos_folium = []
    promedio_x = 0
    promedio_y = 0
    
    for p in approx:
        x, y = p[0]
        # Invertir Y para que coincida con el sistema de coordenadas del mapa
        lat = h - y 
        lng = x
        puntos_folium.append([lat, lng])
        promedio_x += lng
        promedio_y += lat
    
    # Centroide aproximado para etiquetar
    if len(puntos_folium) > 0:
        centro_lat = promedio_y / len(puntos_folium)
        centro_lng = promedio_x / len(puntos_folium)
        
        # AQUÍ VA TU LÓGICA PARA IDENTIFICAR QUÉ MANZANA/LOTE ES CADA POLÍGONO
        # Como es detección automática, usaremos una lógica espacial simple o un placeholder
        # Si tienes coordenadas manuales mapeadas, aquí iría esa lógica.
        # Por ahora, simularemos que identificamos algunos para que el código no falle.
        
        # NOTA: En un entorno real, necesitas una forma de vincular la posición (x,y) 
        # con el nombre "Manzana A - Lote 1". Si no tienes un OCR o mapeo manual,
        # esto es difícil. Asumiré que quieres dibujar TODOS los contornos detectados.
        
        # Datos ficticios para el ejemplo visual si no hay coincidencia exacta
        mz_dummy = "X"
        lote_dummy = str(conteo_mapeados)
        clave_dummy = f"{mz_dummy}-{lote_dummy}"
        
        # Intentar buscar datos reales (esto requiere lógica de coordenadas que no está en el script base)
        # Usaremos valores por defecto
        avance_real = dict_avances.get(clave_dummy, 0.0)
        estado_obs = dict_observaciones.get(clave_dummy, "Sin Obs")
        
        color = obtener_color(avance_real, estado_obs == "Con Observaciones")
        
        # Dibujar polígono
        folium.Polygon(
            locations=puntos_folium,
            color='black',
            weight=1,
            fill_color=color,
            fill_opacity=0.6,
            popup=f"Lote: {lote_dummy}<br>Avance: {avance_real}%<br>Estado: {estado_obs}"
        ).add_to(geo_layer)
        
        conteo_mapeados += 1

geo_layer.add_to(m)

# ==========================================
# 7. EXPORTAR RESULTADO
# ==========================================

# Guardar HTML
output_file = 'mapa_generado.html'
m.save(output_file)
print(f"✅ Mapa generado exitosamente: {output_file} con {conteo_mapeados} elementos.")

# ---------------------------------------------------------
# IMPORTANTE: No olvides tener 'plano.png' en tu Google Drive
# y los Sheets compartidos con el correo del Service Account.
# ---------------------------------------------------------
