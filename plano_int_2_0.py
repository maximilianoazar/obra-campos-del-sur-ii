# -*- coding: utf-8 -*-
import os
import json
import re
import cv2
import numpy as np
import pandas as pd
import gspread
import folium
import unicodedata
from collections import defaultdict
from matplotlib import pyplot as plt
from branca.element import Template, MacroElement

# ==========================================
# 1. CONFIGURACIÓN Y AUTENTICACIÓN
# ==========================================

try:
    if "GDRIVE_CREDENTIALS" in os.environ:
        print("🔑 Cargando credenciales desde Variable de Entorno (GitHub)...")
        creds_dict = json.loads(os.environ["GDRIVE_CREDENTIALS"])
        gc = gspread.service_account_from_dict(creds_dict)
    else:
        print("💻 Buscando archivo 'credentials.json' localmente...")
        gc = gspread.service_account(filename="credentials.json")
except Exception as e:
    print(f"❌ Error de autenticación: {e}")
    exit()

# Cargar imagen
archivo_plano = 'plano.png'
if not os.path.exists(archivo_plano):
    print(f"❌ No se encontró {archivo_plano} en el directorio actual.")
    exit()

img = cv2.imread(archivo_plano)
h, w, _ = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ==========================================
# 2. DETECCIÓN DE VIVIENDAS (OPENCV)
# ==========================================

_, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

casas_geometria = []
centroides = []

def pixel_to_folium(pt, h):
    px_x, px_y = pt
    lat = float(h - px_y)
    lng = float(px_x)
    return [lng, lat]

index_geo = 0
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
                
                coords = []
                for pt in approx:
                    coords.append(pixel_to_folium((float(pt[0][0]), float(pt[0][1])), h))
                coords.append(coords[0])
                casas_geometria.append(coords)
                
                centroides.append({"idx": index_geo, "cx": cx, "cy": cy})
                index_geo += 1

print(f"🏠 Viviendas detectadas: {len(casas_geometria)}")

# ==========================================
# 3. LÓGICA DE MANZANAS Y ORDENAMIENTO
# ==========================================

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

casas_por_manzana = defaultdict(list)
mapa_manzanas = {}

for casa in centroides:
    x, y = casa["cx"], casa["cy"]
    asignada = False
    for letra, regla in MANZANAS.items():
        if regla(x, y):
            casas_por_manzana[letra].append(casa)
            mapa_manzanas[casa["idx"]] = letra
            asignada = True
            break
    if not asignada: mapa_manzanas[casa["idx"]] = "SIN"

def agrupar_en_filas(casas, tol=25):
    filas = []
    for c in sorted(casas, key=lambda x: x["cy"]):
        agregado = False
        for fila in filas:
            if abs(fila[0]["cy"] - c["cy"]) < tol:
                fila.append(c); agregado = True; break
        if not agregado: filas.append([c])
    return filas

mapa_numeros = {}
for mz, lista in casas_por_manzana.items():
    # Ordenamiento simplificado para compatibilidad
    ordenadas = sorted(lista, key=lambda c: (c["cy"], c["cx"]))
    for n, casa in enumerate(ordenadas, start=1):
        mapa_numeros[casa["idx"]] = n

# ==========================================
# 4. CARGA DE DATOS DESDE GOOGLE SHEETS
# ==========================================

# A. Cargar Maestro de Partidas
try:
    sh_maestro = gc.open('Partidas').sheet1
    # Usamos get_all_values para evitar múltiples llamadas a la API
    rows = sh_maestro.get_all_values()
    lista_partidas_maestras = [str(r[0]).strip().upper() for r in rows if r]
    print(f"✅ Maestro cargado: {len(lista_partidas_maestras)} partidas.")
except Exception as e:
    print(f"⚠️ Error cargando 'Partidas': {e}")
    lista_partidas_maestras = []

# B. Cargar Observaciones (Pre F1)
dict_observaciones = {}
try:
    sh_obs = gc.open('Pre F1')
    for hoja in sh_obs.worksheets():
        m_name = hoja.title.strip().upper().replace("MZ", "").strip()
        filas = hoja.get_all_values()
        if len(filas) < 2: continue
        for f in filas[1:]:
            if len(f) < 3: continue
            try:
                num_c = int(float(str(f[0]).strip()))
                partida_obs = str(f[1]).strip().upper()
                estado_obs = str(f[2]).strip().lower()
                if estado_obs == "en proceso":
                    dict_observaciones[(m_name, num_c, partida_obs)] = str(f[3]) if len(f)>3 else ""
            except: continue
except Exception as e:
    print(f"⚠️ Error en Pre F1: {e}")

# C. Cargar Avances
try:
    sh = gc.open('135-CR-CAMPOS DEL SUR 2 (VIVIENDAS_SEDE SOCIAL.1)(1)')
    dict_detalles_casas = {}
    
    for worksheet in sh.worksheets():
        sheet_name = worksheet.title.upper()
        if "MANZ" in sheet_name:
            letra_mz = sheet_name.replace("MANZ.", "").replace("MANZ", "").strip()
            datos = worksheet.get_all_values()
            if len(datos) < 5: continue
            
            # Buscar fila de Items y columnas de casas
            idx_item = next((i for i, f in enumerate(datos[:40]) if f and str(f[0]).strip().upper() == "ITEM"), None)
            if idx_item is None: continue
            
            col_casas = []
            for r_idx in range(max(0, idx_item-2), idx_item+2):
                for c_idx, val in enumerate(datos[r_idx]):
                    if c_idx > 1 and str(val).strip().isdigit():
                        num_v = int(str(val).strip())
                        if not any(x[1] == num_v for x in col_casas): col_casas.append((c_idx, num_v))

            for i in range(idx_item + 1, len(datos)):
                fila = datos[i]
                if not fila or not str(fila[0]).strip(): continue
                item_cod = str(fila[0]).strip().upper()
                desc_p = str(fila[1]).strip().upper()
                
                # Solo procesar si está en el maestro
                if f"{item_cod}-{desc_p}" in [p for p in lista_partidas_maestras]:
                    for c_idx, num_v in col_casas:
                        val_celda = fila[c_idx] if c_idx < len(fila) else ""
                        terminado = (str(val_celda).strip() != "")
                        key = (letra_mz, num_v)
                        if key not in dict_detalles_casas: dict_detalles_casas[key] = []
                        dict_detalles_casas[key].append({
                            'partida': f"[{item_cod}] {desc_p}",
                            'estado': "✅" if terminado else "❌"
                        })
except Exception as e:
    print(f"❌ Error en Spreadsheet principal: {e}")

# ==========================================
# 5. GENERACIÓN DE MAPA (FOLIUM)
# ==========================================

m = folium.Map(location=[h/2, w/2], zoom_start=0, crs='Simple', tiles=None)
folium.raster_layers.ImageOverlay(image=archivo_plano, bounds=[[0,0], [h,w]]).add_to(m)

dict_avances_filtrado = {}

for i, geo in enumerate(casas_geometria):
    mz = str(mapa_manzanas.get(i, "SIN"))
    num = mapa_numeros.get(i, 0)
    key = (mz, num)
    
    detalles = dict_detalles_casas.get(key, [])
    hechas = sum(1 for d in detalles if d['estado'] == "✅")
    total = len(detalles)
    avance = round((hechas/total)*100, 1) if total > 0 else 0
    dict_avances_filtrado[key] = avance
    
    # Verificar si tiene observaciones "En Proceso"
    tiene_obs = any(True for (m_o, c_o, p_o) in dict_observaciones.keys() if m_o == mz and c_o == num)
    
    color = "#d65548" # Rojo
    if tiene_obs: color = "#f2ca27" # Amarillo
    elif avance > 80: color = "#36d278" # Verde
    elif avance >= 30: color = "#409ad5" # Azul

    # Popup simple
    html_p = f"<b>MZ {mz} - Casa {num}</b><br>Avance: {avance}%"
    if tiene_obs: html_p += "<br>⚠️ Tiene observaciones pendientes"

    folium.GeoJson(
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [geo]}},
        style_function=lambda x, c=color: {"fillColor": c, "color": "black", "weight": 1, "fillOpacity": 0.5},
        tooltip=f"MZ {mz} - #{num}: {avance}%"
    ).add_child(folium.Popup(html_p)).add_to(m)

m.fit_bounds([[0,0], [h,w]])
m.save("mapa_generado.html")
print("✅ Proceso finalizado. Mapa guardado como index.html")
