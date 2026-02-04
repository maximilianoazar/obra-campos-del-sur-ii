# -*- coding: utf-8 -*-
import os
import json
import io
import cv2
import numpy as np
import pandas as pd
import gspread
import folium
import re
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from branca.element import Template, MacroElement

print("--- INICIANDO SISTEMA ---")

# 1. AUTENTICACIÓN (GITHUB ACTIONS)
scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
if 'GDRIVE_CREDENTIALS' in os.environ:
    creds_dict = json.loads(os.environ['GDRIVE_CREDENTIALS'])
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    print("🤖 Modo: GitHub Actions")
else:
    raise Exception("❌ No se detectaron credenciales GDRIVE_CREDENTIALS")

drive_service = build('drive', 'v3', credentials=creds)
gc = gspread.authorize(creds)

# 2. FUNCIÓN DE DESCARGA FLEXIBLE
def descargar_archivo_flexible(nombre_busqueda, nombre_local):
    print(f"📥 Buscando archivo que contenga: '{nombre_busqueda}'...")
    # Buscamos por nombre parcial para evitar errores de (1) o espacios
    query = f"name contains '{nombre_busqueda}' and trashed = false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])
    
    if not items:
        print(f"❌ ERROR: No se encontró ningún archivo con '{nombre_busqueda}'")
        return False
        
    file_id = items[0]['id']
    print(f"✅ Encontrado: {items[0]['name']} (ID: {file_id})")
    
    # Si es un Google Sheet, lo exportamos como Excel
    if 'spreadsheet' in items[0].get('name', '').lower() or 'google-apps.spreadsheet' in str(items[0]):
        request = drive_service.files().export_media(fileId=file_id, mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    else:
        request = drive_service.files().get_media(fileId=file_id)

    fh = io.FileIO(nombre_local, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    return True

# --- PROCESO DE DESCARGA ---
if not descargar_archivo_flexible('plano.png', 'plano.png'):
    raise Exception("Falta plano.png")

# Usamos el código del proyecto que es único "135-CR"
if not descargar_archivo_flexible('135-CR', 'avance.xlsx'):
    raise Exception("Falta el Excel de obra (135-CR)")

# 3. PROCESAMIENTO (TU LÓGICA DE CALCULO)
# Mantenemos toda tu lógica de OpenCV y Folium aquí abajo...
# (El resto del código de detección de manzanas y generación de HTML)
print("✅ Archivos listos. Iniciando cálculos internos...")

# ... [Aquí sigue el resto de tu código de plano_int_2_0.py] ...
# Asegúrate de usar 'plano.png' y 'avance.xlsx' como nombres de archivo locales.
