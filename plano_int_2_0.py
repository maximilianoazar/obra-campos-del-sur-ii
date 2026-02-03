# -*- coding: utf-8 -*-
"""
Script de Generación de Plano Interactivo - Automatizado para GitHub Actions
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

# Definir los alcances (permisos) que necesita el robot
scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

print("--- INICIANDO PROCESO DE AUTENTICACIÓN ---")

# Detectar credenciales: O vienen de GitHub (Variable de Entorno) o de un archivo local
if 'GDRIVE_CREDENTIALS' in os.environ:
    print("Detectado entorno GitHub Actions. Leyendo secretos...")
    creds_dict = json.loads(os.environ['GDRIVE_CREDENTIALS'])
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
elif os.path.exists('credentials.json'):
    print("Detectado entorno local. Leyendo credentials.json...")
    creds = Credentials.from_service_account_file('credentials.json', scopes=scopes)
else:
    raise Exception("❌ NO SE ENCONTRARON CREDENCIALES. Configura el secreto GDRIVE_CREDENTIALS en GitHub.")

# Crear clientes de conexión
drive_service = build('drive', 'v3', credentials=creds)
gc = gspread.authorize(creds)
print("✅ Autenticación exitosa.")

# ==========================================
# 2. FUNCIÓN DE DESCARGA DE DRIVE
# ==========================================

def descargar_archivo_drive(nombre_exacto_drive, nombre_destino_local):
    """Busca un archivo por nombre en Drive y lo descarga localmente."""
    try:
        # Buscar ID del archivo
        query = f"name = '{nombre_exacto_drive}' and trashed = false"
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print(f"⚠️ ADVERTENCIA: No se encontró '{nombre_exacto_drive}' en el Drive del robot.")
            return False

        # Si hay duplicados, tomamos el primero
        file_id = items[0]['id']
        print(f"⬇️ Descargando '{nombre_exacto_drive}' (ID: {file_id})...")

        request = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        # Guardar en disco local del servidor
        with open(nombre_destino_local, 'wb') as f:
            f.write(fh.getbuffer())
        
        print(f"✅ Archivo guardado como: {nombre_destino_local}")
        return True

    except Exception as e:
        print(f"❌ Error descargando {nombre_exacto_drive}: {e}")
        return False

# ==========================================
# 3. DESCARGA DE RECURSOS (Imágenes y Excel)
# ==========================================
print("\n--- DESCARGANDO RECURSOS ---")

# 1. Descargar la imagen del plano
if not descargar_archivo_drive('plano.png', 'plano.png'):
    raise Exception("No se pudo descargar el plano.png")

# 2. Descargar el Excel de avances
# Nota: Usamos el nombre largo para buscarlo, pero lo guardamos con nombre corto 'avance.xlsx'
nombre_excel_drive = '135-CR-CAMPOS DEL SUR 2 (VIVIENDAS_SEDE SOCIAL.1).xlsx'
if not descargar_archivo_drive(nombre_excel_drive, 'avance.xlsx'):
    raise Exception("No se pudo descargar el Excel de obra.")

# ==========================================
# 4. PROCESAMIENTO DE IMAGEN (OpenCV)
# ==========================================
print("\n--- PROCESANDO IMAGEN ---")

# Cargar la imagen descargada
img = cv2.imread('plano.png')
if img is None:
    raise Exception("Error al leer plano.png con OpenCV")

h, w, _ = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Umbralización y limpieza
_, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Contornos
contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

casas_geometria = []
centroides_casas = []

def pixel_to_folium(pt, h):
    px_x, px_y = pt
    lat = float(h - px_y)
    lng = float(px_x)
    return [lng, lat]

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    if perimetro == 0: continue

    circularidad = (4 * np.pi * area) / (perimetro ** 2)

    if 200 < area < 4000 and circularidad > 0.4:
        epsilon = 0.03 * perimetro
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if 4 <= len(approx) <= 10:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                centroides_casas.append((cx, cy))

            coords = []
            for pt in approx:
                px_x = float(pt[0][0])
                px_y = float(pt[0][1])
                coords.append(pixel_to_folium((px_x, px_y), h))
            
            coords.append(coords[0]) # Cerrar polígono
            casas_geometria.append(coords)

print(f"ÉXITO: Se detectaron {len(casas_geometria)} viviendas.")

# ==========================================
# 5. LÓGICA DE MANZANAS (Asignación)
# ==========================================

centroides = []
for i, geo in enumerate(casas_geometria):
    xs = [p[0] for p in geo[:-1]]
    ys = [p[1] for p in geo[:-1]]
    cx = sum(xs) / len(xs)
    cy_mapa = sum(ys) / len(ys)
    cy_pixel = h - cy_mapa
    centroides.append({"idx": i, "cx": cx, "cy": cy_pixel})

casas = []
for c in centroides:
    casas.append({
        "idx": c["idx"],
        "cx": c["cx"],
        "cy": c["cy"],
        "geometry": casas_geometria[c["idx"]]
    })

MANZANAS = {
    "A": lambda x,y: x > 1400 and y < 500,
    "B": lambda x,y: x > 1400 and 500 <= y < 850,
    "C": lambda x,y: 770 < x <= 1400 and y < 500,
    "D": lambda x,y: 1250 < x <= 1400 and 550 <= y < 730,
    "E": lambda x,y: 770 < x <= 1400 and y >= 800,
    "F": lambda x,y: 1000 < x <= 1230 and 550 <= y < 730,
    "G": lambda x,y: 80 <= x < 775 and y < 500,
    "H": lambda x,y: 570 < x < 775 and 550 <= y < 750,
    "I": lambda x,y: 380 < x <= 710 and 550 <= y < 750,
    "J": lambda x,y: x <= 780 and y >= 780,
    "K": lambda x,y: 220 < x <= 380 and 550 <= y < 750,
    "L": lambda x,y: x <= 215 and 550 <= y < 750
}

from collections import defaultdict
casas_por_manzana = defaultdict(list)

for casa in centroides:
    x, y = casa["cx"], casa["cy"]
    asignada = False
    for letra, regla in MANZANAS.items():
        if regla(x, y):
            casas_por_manzana[letra].append(casa)
            asignada = True
            break
    if not asignada:
        casas_por_manzana["SIN_MANZANA"].append(casa)

# ==========================================
# 6. ORDENAMIENTO (Numeración)
# ==========================================

def agrupar_en_filas(casas, tolerancia=25):
    filas = []
    for c in sorted(casas, key=lambda x: x["cy"]):
        agregado = False
        for fila in filas:
            if abs(fila[0]["cy"] - c["cy"]) < tolerancia:
                fila.append(c); agregado = True; break
        if not agregado: filas.append([c])
    return filas

def ordenar_rectangular(casas):
    filas = agrupar_en_filas(casas)
    casas_ordenadas = []
    for fila in filas:
        fila_ordenada = sorted(fila, key=lambda c: c["cx"])
        casas_ordenadas.extend(fila_ordenada)
    return casas_ordenadas

def ordenar_lineal(casas, modo):
    if modo == "LR_T": return sorted(casas, key=lambda c: (c["cy"], c["cx"]))
    if modo == "RL_T": return sorted(casas, key=lambda c: (c["cy"], -c["cx"]))
    return casas

def ordenar_perimetro(casas, es_especial=False):
    if not casas: return []
    min_x, max_x = min(c['cx'] for c in casas), max(c['cx'] for c in casas)
    min_y, max_y = min(c['cy'] for c in casas), max(c['cy'] for c in casas)
    tol = 30
    muro_izq, muro_sup, muro_der, muro_inf = [], [], [], []
    procesadas = set()

    candidatos_izq = sorted([c for c in casas if c['cx'] < min_x + tol], key=lambda x: x['cy'], reverse=True)
    for c in candidatos_izq: muro_izq.append(c); procesadas.add(c['idx'])

    candidatos_sup = sorted([c for c in casas if c['cy'] < min_y + tol and c['idx'] not in procesadas], key=lambda x: x['cx'])
    for c in candidatos_sup: muro_sup.append(c); procesadas.add(c['idx'])

    candidatos_der = sorted([c for c in casas if c['cx'] > max_x - tol and c['idx'] not in procesadas], key=lambda x: x['cy'])
    for c in candidatos_der: muro_der.append(c); procesadas.add(c['idx'])

    candidatos_inf = sorted([c for c in casas if c['cy'] > max_y - tol and c['idx'] not in procesadas], key=lambda x: x['cx'], reverse=True)
    for c in candidatos_inf: muro_inf.append(c); procesadas.add(c['idx'])

    orden_base = muro_izq + muro_sup + muro_der + muro_inf
    for c in casas:
        if c['idx'] not in procesadas: orden_base.append(c)

    if es_especial and len(orden_base) > 2:
        return [orden_base[0], orden_base[-1]] + orden_base[1:-1]
    return orden_base

mapa_numeros = {}
mapa_manzanas = {} # Importante para guardar la relación índice -> manzana
MANZANAS_ESPIRAL_ESPECIAL = {"D", "F", "H"}
ORDEN_MANZANA_LINEAL = {"A": "LR_T", "C": "LR_T", "E": "RL_T", "G": "LR_T", "J": "LR_T"}

for manzana, casas_lista in casas_por_manzana.items():
    if manzana == "SIN_MANZANA": continue
    casas_ord = []

    if manzana == "L":
        casas_ord = ordenar_perimetro(casas_lista, es_especial=False)
        if len(casas_ord) >= 2: casas_ord[-1], casas_ord[-2] = casas_ord[-2], casas_ord[-1]
    elif manzana in ["I", "K"]:
        casas_ord = ordenar_perimetro(casas_lista, es_especial=True)
        if len(casas_ord) >= 2: casas_ord[-1], casas_ord[-2] = casas_ord[-2], casas_ord[-1]
    elif manzana in MANZANAS_ESPIRAL_ESPECIAL:
        casas_ord = ordenar_perimetro(casas_lista, es_especial=True)
    elif manzana == "B":
        casas_ord = ordenar_rectangular(casas_lista)
    else:
        if manzana == "C":
            filas = agrupar_en_filas(casas_lista, tolerancia=40)
            casas_ord = sorted(filas[0], key=lambda c: c["cx"])
        else:
            modo = ORDEN_MANZANA_LINEAL.get(manzana, "LR_T")
            casas_ord = ordenar_lineal(casas_lista, modo)

    for n, casa in enumerate(casas_ord, start=1):
        mapa_numeros[casa["idx"]] = n
        mapa_manzanas[casa["idx"]] = manzana

# ==========================================
# 7. PROCESAMIENTO EXCEL (Pandas)
# ==========================================
print("\n--- ANALIZANDO EXCEL ---")

file_path = 'avance.xlsx' # Usamos el archivo descargado localmente
excel_file = pd.ExcelFile(file_path)
dict_avances = {}

def es_partida_real(codigo):
    if not isinstance(codigo, str): return False
    return len(re.findall(r'\.', codigo.strip())) >= 2

for sheet_name in excel_file.sheet_names:
    if "MANZ." in sheet_name.upper():
        letra_mz = sheet_name.split('.')[-1].strip()
        df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)

        idx_titulo = None
        for i, row in df_raw.iterrows():
            row_str = " ".join([str(v).upper() for v in row.values])
            if "VIVIENDA" in row_str and "LOTE" in row_str:
                idx_titulo = i; break

        if idx_titulo is None: continue

        idx_numeros = idx_titulo + 1
        fila_numeros = df_raw.iloc[idx_numeros]
        columnas_casas_info = []

        for col_idx, valor in enumerate(fila_numeros):
            val_clean = str(valor).strip()
            if val_clean.isdigit():
                columnas_casas_info.append((col_idx, int(val_clean)))

        df_datos = pd.read_excel(excel_file, sheet_name=sheet_name, skiprows=idx_numeros + 1)
        col_item_nombre = df_datos.columns[0]
        df_partidas = df_datos[df_datos[col_item_nombre].apply(es_partida_real)]
        total_p = len(df_partidas)

        if total_p > 0:
            for col_idx, num_casa in columnas_casas_info:
                col_data = df_partidas.iloc[:, col_idx]
                completadas = col_data.notna().sum()
                porcentaje = round((completadas / total_p) * 100, 1)
                dict_avances[(letra_mz, num_casa)] = porcentaje

# ==========================================
# 8. CONEXIÓN GOOGLE SHEETS (Gspread)
# ==========================================
print("\n--- CONECTANDO A GOOGLE SHEETS ---")

# Cargar Maestro de Partidas
try:
    sh_maestro = gc.open('Partidas').sheet1
    lista_partidas_maestras = [str(p).strip().upper() for p in sh_maestro.col_values(1) if p]
    print(f"✅ Maestro 'Partidas' cargado.")
except Exception as e:
    print(f"⚠️ Error cargando 'Partidas'. Asegúrate de compartir el sheet con el robot. Error: {e}")
    lista_partidas_maestras = []

# Cargar Observaciones
try:
    nombre_hoja_obs = 'Pre F1'
    sh_obs = gc.open(nombre_hoja_obs)
    dict_observaciones = {}
    casas_con_obs = set()

    for hoja in sh_obs.worksheets():
        nombre_hoja = hoja.title.strip().upper()
        if "MZ" in nombre_hoja:
            letra_mz = nombre_hoja.replace("MZ", "").replace(".","").strip() # Limpieza extra
            filas = hoja.get_all_values()
            if len(filas) < 2: continue

            for i, fila in enumerate(filas[1:], start=2):
                if len(fila) < 3: continue
                lote_raw = str(fila[0]).strip()
                partida = str(fila[1]).strip()
                estado = str(fila[2]).strip()
                comentario = str(fila[3]).strip() if len(fila) > 3 else "Sin detalle"

                try: num_casa = int(float(lote_raw))
                except: continue

                if estado.lower() == "en proceso":
                    key_partida = (letra_mz, num_casa, partida)
                    dict_observaciones[key_partida] = comentario
                    casas_con_obs.add((letra_mz, num_casa))
    print(f"✅ Observaciones 'Pre F1' cargadas.")

except Exception as e:
    print(f"⚠️ Error cargando 'Pre F1': {e}")
    dict_observaciones = {}

# ==========================================
# 9. PROCESO DE CRUCE DE DATOS (OpenPyxl)
# ==========================================

# Carga de lista maestra para filtro estricto
try:
    # Reutilizamos la conexión de gspread hecha arriba
    filas_maestras = sh_maestro.get_all_values()
    lista_maestra_llaves = set()
    for fila in filas_maestras:
        if len(fila) >= 2:
            item_m = str(fila[0]).strip().upper()
            nom_m = str(fila[1]).strip().upper()
            if item_m and nom_m:
                lista_maestra_llaves.add(f"{item_m}-{nom_m}")
except:
    lista_maestra_llaves = set()

def es_partida_maestra_estricta(codigo, nombre):
    if not lista_maestra_llaves: return True
    llave_actual = f"{str(codigo).strip().upper()}-{str(nombre).strip().upper()}"
    return llave_actual in lista_maestra_llaves

def normalizar(txt):
    if not txt: return ""
    txt = str(txt).lower()
    txt = "".join(c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn')
    txt = re.sub(r'[^a-z0-9]', '', txt)
    return txt

wb = openpyxl.load_workbook(file_path, data_only=True)
dict_detalles_casas = {}

for sheet_name in wb.sheetnames:
    if "MANZ" in sheet_name.upper():
        ws = wb[sheet_name]
        letra_mz = sheet_name.replace("MANZ.", "").replace("MANZ", "").strip().upper()

        fila_item = None
        for r in range(1, 50):
            if str(ws.cell(row=r, column=1).value).strip().upper() == "ITEM":
                fila_item = r; break
        if not fila_item: continue

        columnas_casas = []
        for r_search in range(fila_item - 2, fila_item + 3):
            for c in range(3, 150):
                val = ws.cell(row=r_search, column=c).value
                if val and str(val).strip().isdigit():
                    num_casa = int(str(val).strip())
                    if not any(x[1] == num_casa for x in columnas_casas):
                        columnas_casas.append((c, num_casa))

        titulo_act, sub_act = "", ""
        for r in range(fila_item + 1, ws.max_row + 1):
            cell_it = ws.cell(row=r, column=1)
            item_val = str(cell_it.value).strip() if cell_it.value else ""
            desc_val = str(ws.cell(row=r, column=2).value).strip() if ws.cell(row=r, column=2).value else ""

            if not item_val or item_val == "None": continue

            if cell_it.font.bold:
                if "." not in item_val: titulo_act = desc_val; sub_act = ""
                else: sub_act = desc_val

            elif es_partida_maestra_estricta(item_val, desc_val):
                for col_idx, num_casa in columnas_casas:
                    v_celda = ws.cell(row=r, column=col_idx).value
                    terminado = pd.notna(v_celda) if hasattr(v_celda, 'notna') else (v_celda is not None and str(v_celda).strip() != "")

                    match_encontrado = False
                    comentario_texto = ""
                    partida_excel_norm = normalizar(desc_val)

                    for (obs_mzn, obs_casa, obs_partida), obs_coment in dict_observaciones.items():
                        # Normalización agresiva para comparar 'A' con 'A '
                        if obs_mzn.upper().strip() == letra_mz.strip() and int(obs_casa) == int(num_casa):
                            if normalizar(obs_partida) in partida_excel_norm or partida_excel_norm in normalizar(obs_partida):
                                match_encontrado = True
                                comentario_texto = obs_coment
                                break

                    key = (letra_mz, num_casa)
                    if key not in dict_detalles_casas: dict_detalles_casas[key] = []
                    dict_detalles_casas[key].append({
                        'titulo': titulo_act, 'subtitulo': sub_act,
                        'partida': f"[{item_val}] {desc_val}",
                        'estado': "✅" if terminado else "❌",
                        'tiene_obs': match_encontrado, 'comentario': comentario_texto
                    })

# ==========================================
# 10. CLASIFICACIÓN DE TIPOS Y RECALCULO
# ==========================================

tipos_ref = {
    "Tipo B": ["D1", "F1", "F8", "I1", "I8", "K1", "K8", "L1", "L7"],
    "Tipo C": ["B3", "B4", "B5", "B6", "B7", "G1"],
    "Tipo D": ["B1", "B2"],
    "Tipo A2": ["E16", "E17"],
    "Tipo A1-N": ["E1", "D11", "F11", "I12", "J21"]
}
dict_tipos_vivienda = {}

for i in range(len(mapa_manzanas)):
    mz_val = str(mapa_manzanas.get(i, "SIN")).strip()
    num_val = str(mapa_numeros.get(i, 0)).strip()
    id_busqueda = f"{mz_val}{num_val}".replace("MANZ.", "").replace(" ", "")

    v_tipo = "Tipo A1"
    for t_nombre, lista in tipos_ref.items():
        if id_busqueda in lista:
            v_tipo = t_nombre; break
    dict_tipos_vivienda[(mapa_manzanas.get(i), mapa_numeros.get(i))] = v_tipo

REGLAS_PARTIDAS = {
    "B.4.4.1": {"tipos": {"Tipo A1", "Tipo A1-N", "Tipo A2"}},
    "B.4.4.2": {"tipos": {"Tipo A1", "Tipo A1-N", "Tipo A2"}},
    "B.5.3.1": {"tipos": {"Tipo A1", "Tipo A1-N", "Tipo A2"}},
    "C.2.3.1.B": {"tipos": {"Tipo A1-N"}},
    "C.5.4": {"tipos": {"Tipo A1", "Tipo A1-N", "Tipo A2", "Tipo B"}},
    "C.7.1": {"tipos": {"Tipo A1", "Tipo A1-N", "Tipo A2"}},
    "C.9.3.1": {"tipos": {"Tipo A1", "Tipo A1-N", "Tipo A2", "Tipo B"}},
    "C.EX.3":  {"tipos": {"Tipo A1-N", "Tipo D"}},
    "C.EX.14.1": {"tipos": {"Tipo A1-N", "Tipo B", "Tipo C", "Tipo D"}},
    "C.EX.15": {"tipos": {"Tipo A1-N", "Tipo B"}},
    "C.EX.16": {"tipos": {"Tipo C", "Tipo D"}},
    "C.EX.18": {"tipos": {"Tipo B", "Tipo C"}},
    "D.1.3": {"tipos": {"Tipo C", "Tipo D"}},
    "D.1.4": {"tipos": {"Tipo A1", "Tipo A1-N", "Tipo A2"}},
    "D.1.5": {"tipos": {"Tipo C", "Tipo D"}},
    "D.1.8": {"tipos": {"Tipo C", "Tipo D"}},
    "D.1.9": {"tipos": {"Tipo C", "Tipo D"}},
    "D.1.10": {"tipos": {"Tipo C", "Tipo D"}},
    "D.1.11": {"tipos": {"Tipo C", "Tipo D"}},
    "D.1.12": {"tipos": {"Tipo C", "Tipo D"}},
    "D.4.5.4": {"tipos": {"Tipo D"}},
    "D.EX.3": {"tipos": {"Tipo A1-N", "Tipo D"}},
    "D.EX.4": {"tipos": {"Tipo B"}},
}

def partida_aplica_a_vivienda(codigo_partida, tipo_v, mz, casa):
    if not codigo_partida: return True
    if codigo_partida not in REGLAS_PARTIDAS: return True
    regla = REGLAS_PARTIDAS[codigo_partida]
    return tipo_v in regla.get("tipos", set())

dict_avances_filtrado = {}

for key, lista_partidas in dict_detalles_casas.items():
    mz, casa = key
    tipo_v = dict_tipos_vivienda.get(key, "Tipo A1")
    filtradas = []
    for p in lista_partidas:
        partida_raw = p.get("partida", "")
        match = re.search(r'\[(.*?)\]', partida_raw)
        codigo = match.group(1).strip() if match else ""
        if partida_aplica_a_vivienda(codigo, tipo_v, mz, casa):
            filtradas.append(p)
    
    if filtradas:
        total_p = len(filtradas)
        hechas = sum(1 for p in filtradas if p['estado'] == "✅")
        dict_avances_filtrado[key] = round((hechas / total_p) * 100, 1)

# ==========================================
# 11. GENERACIÓN DEL MAPA FOLIUM
# ==========================================
print("\n--- GENERANDO MAPA INTERACTIVO ---")

limites = [[0, 0], [h, w]]
m = folium.Map(location=[h/2, w/2], zoom_start=0, crs='Simple', tiles=None)
folium.raster_layers.ImageOverlay(image='plano.png', bounds=limites, zindex=1).add_to(m)

def obtener_color_estatico(avance, tiene_obs):
    if tiene_obs: return "#f2ca27" # Amarillo
    if avance > 80: return "#36d278" # Verde
    if 30 <= avance <= 80: return "#409ad5" # Azul
    return "#d65548" # Rojo

def generar_html_popup(manzana, casa_num, detalles, tipo_vivienda, avance):
    # Filtrar detalles para el popup también
    detalles_popup = []
    for d in detalles:
        match = re.search(r'\[(.*?)\]', d['partida'])
        codigo = match.group(1).strip() if match else ""
        if partida_aplica_a_vivienda(codigo, tipo_vivienda, manzana, casa_num):
            detalles_popup.append(d)

    resumen = {}
    for d in detalles_popup:
        t, s = d['titulo'], d['subtitulo']
        if t not in resumen: resumen[t] = {'total': 0, 'listo': 0, 'subs': {}, 'obs': False}
        resumen[t]['total'] += 1
        if d['estado'] == "✅": resumen[t]['listo'] += 1
        if d.get('tiene_obs'): resumen[t]['obs'] = True
        if s:
            if s not in resumen[t]['subs']: resumen[t]['subs'][s] = {'total': 0, 'listo': 0, 'obs': False}
            resumen[t]['subs'][s]['total'] += 1
            if d['estado'] == "✅": resumen[t]['subs'][s]['listo'] += 1
            if d.get('tiene_obs'): resumen[t]['subs'][s]['obs'] = True

    html = f"""
    <div style="font-family: 'Segoe UI', Arial; width: 520px; background: white; margin: -15px -10px -10px -10px;">
        <div style="background: #2c3e50; color: white; padding: 15px 10px; display: flex; justify-content: space-between; align-items: center;">
            <h4 style="margin: 0; font-size: 16px;">MZ {manzana} - Casa {casa_num} - {tipo_vivienda}</h4>
            <div style="width: 160px;">
                <div style="font-size: 12px; font-weight: bold; text-align: right;">{avance}%</div>
                <div style="background: #dcdde1; border-radius: 6px; height: 8px; overflow: hidden;">
                    <div style="width: {avance}%; height: 100%; background: linear-gradient(90deg, #2980b9, #27ae60);"></div>
                </div>
            </div>
        </div>
        <div style="display: flex; height: 380px;">
            <div style="flex: 1.8; overflow-y: auto; padding: 10px; border-right: 1px solid #eee;" id="lista_partidas">
                <table style="width: 100%; border-collapse: collapse; table-layout: fixed;">
                    <colgroup><col style="width: 85%;"><col style="width: 15%;"></colgroup>
    """

    current_tit, current_sub = None, None
    for item in detalles_popup:
        if item['titulo'] != current_tit:
            current_tit = item['titulo']
            html += f'<tr style="background: #edeff0;"><td colspan="2" style="padding: 10px 5px; font-weight: bold; color: #2c3e50; border-top: 2px solid #2c3e50;">{current_tit.upper()}</td></tr>'
        if item['subtitulo'] != current_sub:
            current_sub = item['subtitulo']
            if current_sub:
                html += f'<tr style="background: #f9f9f9;"><td colspan="2" style="padding: 6px 8px; font-weight: bold; color: #7f8c8d; font-style: italic; border-bottom: 1px solid #eee;">↳ {current_sub}</td></tr>'

        if item.get('tiene_obs'):
            color_st = "#d4a017"; icono = "⚠️"
            comentario = item.get("comentario", "Sin detalle")
            nombre = f'<div style="padding: 2px 0;"><b style="color: #d4a017;">{item["partida"]}</b><details style="margin-top: 4px;"><summary style="cursor: pointer; color: #856404; font-size: 10px; font-weight: bold;">Ver nota [+]</summary><div style="margin-top: 4px; padding: 8px; background: #fff9e6; border-left: 3px solid #d4a017; color: #856404; font-size: 10px;">{comentario}</div></details></div>'
        else:
            color_st = "#27ae60" if item['estado'] == "✅" else "#e74c3c"; icono = item['estado']
            nombre = f"<span style='color: #444; font-size: 11px;'>{item['partida']}</span>"
        
        html += f'<tr style="border-bottom: 1px solid #f2f2f2;"><td style="padding: 8px 10px; vertical-align: top;">{nombre}</td><td style="padding: 8px 5px; text-align: center; color: {color_st}; font-weight: bold; font-size: 14px;">{icono}</td></tr>'
    
    html += "</table></div>"
    # (Omitimos el índice lateral para simplificar código, pero los datos están ahí)
    html += "</div></div>"
    return html

for i, geo in enumerate(casas_geometria):
    mz = str(mapa_manzanas.get(i, "SIN"))
    try: num = int(float(mapa_numeros.get(i, 0)))
    except: num = 0
    key = (mz, num)

    avance_val = dict_avances_filtrado.get(key, 0)
    detalles = dict_detalles_casas.get(key, [])
    tiene_observacion = any(d.get('tiene_obs') for d in detalles)
    tipo_v = dict_tipos_vivienda.get(key, "Tipo A1")

    color_casa = obtener_color_estatico(avance_val, tiene_observacion)

    if detalles:
        popup_html = generar_html_popup(mz, num, detalles, tipo_v, avance_val)
    else:
        popup_html = f"<div style='font-family:Arial;width:200px;'><b>Mzn {mz} Casa {num}</b><br>Avance: {avance_val}%<br>Sin detalles.</div>"

    feature_data = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [geo]},
        "properties": {
            "manzana": mz, "numero": num, "tipo": tipo_v,
            "etiqueta_avance": f"{avance_val}%"
        }
    }

    geo_layer = folium.GeoJson(
        feature_data,
        style_function=lambda x, c=color_casa: {
            "fillColor": c, "fillOpacity": 0.5, "weight": 1.2, "color": "black"
        },
        highlight_function=lambda x: {"fillOpacity": 0.8, "weight": 2.5},
        tooltip=folium.GeoJsonTooltip(
            fields=["manzana", "numero", "tipo", "etiqueta_avance"],
            aliases=["Manzana:", "Casa Nº:", "Tipo:", "Avance:"],
            sticky=True,
            style="background-color: white; border: 1px solid black; font-family: Arial; font-size: 12px;"
        )
    )
    geo_layer.add_child(folium.Popup(popup_html, max_width=520))
    geo_layer.add_to(m)

# Macro Overlay (Resumen Global)
promedio_total = round(sum(dict_avances_filtrado.values()) / len(dict_avances_filtrado), 1) if dict_avances_filtrado else 0
overlay_html = f"""
{{% macro html(this, kwargs) %}}
<div style="position: fixed; top: 20px; right: 20px; z-index: 9999; background: white; padding: 16px; border-radius: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.25); font-family: Arial; width: 200px;">
    <div style="font-weight: bold; font-size: 13px;">Avance Global</div>
    <div style="font-size: 26px; font-weight: bold; color: #2c7be5; text-align: center;">{promedio_total}%</div>
    <div style="background: #e0e0e0; border-radius: 8px; height: 10px; overflow: hidden;">
        <div style="width: {promedio_total}%; height: 100%; background: linear-gradient(90deg, #27ae60, #2ecc71);"></div>
    </div>
</div>
{{% endmacro %}}
"""
macro = MacroElement()
macro._template = Template(overlay_html)
m.get_root().add_child(macro)

m.fit_bounds(limites)

# ==========================================
# 12. GUARDADO FINAL
# ==========================================
print("\n--- GUARDANDO ARCHIVO HTML ---")
m.save("mapa_generado.html")
print("✅ ¡Mapa generado con éxito como 'mapa_generado.html'!")
