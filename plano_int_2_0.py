# -*- coding: utf-8 -*-
import os
import json  # Agregado para procesar el secreto JSON
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import re
import gspread
import folium
from branca.element import Template, MacroElement
from collections import defaultdict
import unicodedata

# ========================================================
# CONFIGURACIÓN INICIAL (ADAPTADO PARA GITHUB)
# ========================================================

# 1. ELIMINADO: drive.mount('/content/drive') -> Genera incompatibilidad en GitHub.
# 2. ELIMINADO: Cambio de directorio con os.chdir. Se asume que el script corre en la raíz.

print(f"Directorio de trabajo actual: {os.getcwd()}")
print("Archivos encontrados:", os.listdir())

# 1. Cargar la imagen
# Se asume que plano2.png está en la raíz del repositorio
img = cv2.imread('plano2.png')

if img is None:
    raise FileNotFoundError("❌ Error: No se encontró 'plano2.png'. Asegúrate de que está en el repositorio.")

h, w, _ = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Umbralización para detectar bloques negros
_, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

# 3. Limpieza de ruido
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# 4. Encontrar contornos
contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

casas_geometria = []
debug_img = img.copy()

def pixel_to_folium(pt, h):
    px_x, px_y = pt
    # latitud = Altura_Total - pixel_y (esto invierte el eje para Folium)
    # longitud = pixel_x
    lat = float(h - px_y)
    lng = float(px_x)
    return [lng, lat]

centroides_casas = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)

    if perimetro == 0: continue

    # Circularidad para descartar líneas
    circularidad = (4 * np.pi * area) / (perimetro ** 2)

    # Filtros de área y forma
    if 200 < area < 4000 and circularidad > 0.4:
        epsilon = 0.03 * perimetro
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if 4 <= len(approx) <= 10:
            cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 2)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
              cx = M["m10"] / M["m00"]
              cy = M["m01"] / M["m00"]
              centroides_casas.append((cx, cy))

            # --- CORRECCIÓN CLAVE PARA FOLIUM ---
            coords = []
            for pt in approx:
              px_x = float(pt[0][0])
              px_y = float(pt[0][1])
              coords.append(pixel_to_folium((px_x, px_y), h))

            coords.append(coords[0]) # Cerrar polígono
            casas_geometria.append(coords)

print(f"ÉXITO: Se detectaron {len(casas_geometria)} viviendas.")

# Visualización de Debug
# COMENTADO: plt.show() detiene la ejecución en servidores/GitHub Actions
# plt.figure(figsize=(15, 7))
# plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
# plt.title("Detección de Viviendas (Verde)")
# plt.axis('off')
# plt.show()

centroides = []

for i, geo in enumerate(casas_geometria):
    xs = [p[0] for p in geo[:-1]]
    ys = [p[1] for p in geo[:-1]]

    cx = sum(xs) / len(xs)
    cy_mapa = sum(ys) / len(ys)

    # --- CORRECCIÓN ---
    cy_pixel = h - cy_mapa

    centroides.append({
        "idx": i,
        "cx": cx,
        "cy": cy_pixel # Usamos la coordenada de pixel para la lógica
    })

# =========================
# 1. ESTRUCTURA DE CASAS
# =========================

casas = []

for c in centroides:
    casas.append({
        "idx": c["idx"],
        "cx": c["cx"],
        "cy": c["cy"],
        "geometry": casas_geometria[c["idx"]]
    })


# =========================
# 2. DEFINICIÓN DE MANZANAS
# =========================

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


# =========================
# 3. ASIGNACIÓN CASA → MANZANA
# =========================

mapa_manzanas = {}
casas_por_manzana = {}

for casa in casas:
    asignada = False

    for manzana, condicion in MANZANAS.items():
        if condicion(casa["cx"], casa["cy"]):
            mapa_manzanas[casa["idx"]] = manzana
            casas_por_manzana.setdefault(manzana, []).append(casa)
            asignada = True
            break

    if not asignada:
        mapa_manzanas[casa["idx"]] = "SIN"
        casas_por_manzana.setdefault("SIN", []).append(casa)


# DEBUG OPCIONAL
for m, lst in casas_por_manzana.items():
    print(f"Manzana {m}: {len(lst)} casas")

total = sum(len(casas) for casas in casas_por_manzana.values())
print(f"Total de casas asignadas: {total}")

if "SIN_MANZANA" in casas_por_manzana:
    print("⚠️ Casas sin manzana:")
    for c in casas_por_manzana["SIN_MANZANA"]:
        print(f"idx={c['idx']} cx={int(c['cx'])} cy={int(c['cy'])}")

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

total = 0
for m, casas in casas_por_manzana.items():
    print(f"Manzana {m}: {len(casas)} casas")
    total += len(casas)

print("TOTAL:", total)

# ========================================================
# 1. FUNCIONES DE ORDENAMIENTO (ESTRUCTURA BASE)
# ========================================================

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

# ========================================================
# 2. EJECUCIÓN CON CONDICIONALES INDEPENDIENTES
# ========================================================
mapa_numeros = {}
MANZANAS_ESPIRAL_ESPECIAL = {"D", "F", "H"}
ORDEN_MANZANA_LINEAL = {"A": "LR_T", "C": "LR_T", "E": "RL_T", "G": "LR_T", "J": "LR_T"}

for manzana, casas_lista in casas_por_manzana.items():
    if manzana == "SIN_MANZANA": continue

    casas_ord = []

    # --- CONDICIONAL MANZANA L (Casas 10 y 11) ---
    if manzana == "L":
        casas_ord = ordenar_perimetro(casas_lista, es_especial=False)
        if len(casas_ord) >= 2:
            # Swap de las últimas dos para corregir inversión
            casas_ord[-1], casas_ord[-2] = casas_ord[-2], casas_ord[-1]

    # --- CONDICIONAL MANZANAS I y K (Casas 11 y 12, modo especial) ---
    elif manzana in ["I", "K"]:
        casas_ord = ordenar_perimetro(casas_lista, es_especial=True)
        if len(casas_ord) >= 2:
            # Swap de las últimas dos de la lista corregida (sin tocar la casa 2)
            casas_ord[-1], casas_ord[-2] = casas_ord[-2], casas_ord[-1]

    # --- RESTO DE MANZANAS ---
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

    # ASIGNACIÓN FINAL
    for n, casa in enumerate(casas_ord, start=1):
        mapa_numeros[casa["idx"]] = n

print(f"✅ mapa_numeros corregido: L (10-11), I y K (11-12) con casa 2 fija.")

debug = img.copy()

for c in centroides:
    cx, cy = int(c["cx"]), int(c["cy"])
    cv2.circle(debug, (cx, cy), 6, (0,0,255), -1)

# COMENTADO: plt.show() genera incompatibilidad en entornos sin pantalla (headless)
# plt.figure(figsize=(12,6))
# plt.imshow(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.show()


# ========================================================
# AUTENTICACIÓN GOOGLE SHEETS (MODIFICADO PARA SECRETO)
# ========================================================

# MODIFICACIÓN: En lugar de auth.authenticate_user() o leer un archivo,
# leemos la variable de entorno donde GitHub inyecta el Secreto.
try:
    if "GDRIVE_CREDENTIALS" in os.environ:
        # Si existe la variable de entorno (GitHub Actions)
        print("🔑 Detectado Secreto GDRIVE_CREDENTIALS. Iniciando sesión...")
        creds_json = os.environ["GDRIVE_CREDENTIALS"]
        creds_dict = json.loads(creds_json)
        gc = gspread.service_account_from_dict(creds_dict)
    else:
        # Fallback: Si no está la variable, intentamos buscar el archivo localmente (Para pruebas en tu PC)
        print("⚠️ No se detectó variable de entorno. Buscando archivo 'GDRIVE_CREDENTIALS.json'...")
        gc = gspread.service_account(filename='GDRIVE_CREDENTIALS.json')

    print("✅ Autenticación exitosa.")

except Exception as e:
    raise Exception(f"❌ Error crítico de autenticación: {e}. Revisa tus Secretos de GitHub.")


# 2. Abrir el Google Sheet
spreadsheet_name = '135-CR-CAMPOS DEL SUR 2 (VIVIENDAS_SEDE SOCIAL.1)(1)'
sh = gc.open(spreadsheet_name)

dict_avances = {}

def es_partida_real(codigo):
    if not isinstance(codigo, str): return False
    return len(re.findall(r'\.', codigo.strip())) >= 2

print(f"--- INICIO DE DEBUG (ESCANEO DESDE GOOGLE SHEETS) ---")

# 3. Iterar por las hojas
worksheets = sh.worksheets()

for ws in worksheets:
    sheet_name = ws.title
    if "MANZ." in sheet_name.upper():
        letra_mz = sheet_name.split('.')[-1].strip()

        data = ws.get_all_values()
        df_raw = pd.DataFrame(data)

        # 1. Localizar la fila del título "VIVIENDA LOTE"
        idx_titulo = None
        for i, row in df_raw.iterrows():
            row_str = " ".join([str(v).upper() for v in row.values])
            if "VIVIENDA" in row_str and "LOTE" in row_str:
                idx_titulo = i
                break

        if idx_titulo is None:
            print(f"⚠️ Manzana {letra_mz}: No se halló el texto 'VIVIENDA LOTE'")
            continue

        # 2. Los números están en la fila SIGUIENTE (idx_titulo + 1)
        idx_numeros = idx_titulo + 1
        fila_numeros = df_raw.iloc[idx_numeros]

        columnas_casas_info = [] # (indice_columna, numero_casa)

        for col_idx, valor in enumerate(fila_numeros):
            val_clean = str(valor).strip()
            if val_clean.isdigit():
                num_casa = int(val_clean)
                columnas_casas_info.append((col_idx, num_casa))

        if not columnas_casas_info:
            print(f"⚠️ Manzana {letra_mz}: Fila de título en {idx_titulo}, pero fila {idx_numeros} no tiene números.")
            continue

        print(f"✅ Manzana {letra_mz}: {len(columnas_casas_info)} casas encontradas { [c[1] for c in columnas_casas_info] }")

        # 3. Procesar partidas
        df_datos = df_raw.iloc[idx_numeros + 1:].copy()

        # Filtrar solo partidas X.x.x
        mask_partidas = df_datos[0].apply(es_partida_real)
        df_partidas = df_datos[mask_partidas]

        total_p = len(df_partidas)

        if total_p > 0:
            for col_idx, num_casa in columnas_casas_info:
                col_data = df_partidas.iloc[:, col_idx]

                # Lógica de conteo original
                completadas = col_data.apply(lambda x: str(x).strip() if pd.notna(x) else "").replace("", pd.NA).notna().sum()
                porcentaje = round((completadas / total_p) * 100, 1)
                dict_avances[(letra_mz, num_casa)] = porcentaje

print(f"\n--- FIN DE DEBUG ---")
print(f"Resultado: {len(dict_avances)} registros de avance creados.")

# ========================================================
# BLOQUE: CARGA DE MAESTRO DE PARTIDAS (FILTRO)
# ========================================================

try:
    # Usamos 'gc' ya autenticado
    sh_maestro = gc.open('Partidas').sheet1

    lista_partidas_maestras = [str(p).strip().upper() for p in sh_maestro.col_values(1) if p]

    def es_partida_valida(nombre_partida):
        return str(nombre_partida).strip().upper() in lista_partidas_maestras

    print(f"✅ Maestro de partidas cargado con éxito.")
    print(f"📋 Total de partidas maestras detectadas para filtrar: {len(lista_partidas_maestras)}")

except Exception as e:
    print(f"❌ Error al cargar el archivo 'Partidas': {e}")
    # En producción esto debería detenerse, pero lo dejaremos pasar por compatibilidad con tu logica
# ========================================================

nombre_hoja_obs = 'Pre F1'
try:
    sh_obs = gc.open(nombre_hoja_obs)
    dict_observaciones = {}
    casas_con_obs = set()

    print(f"--- Iniciando Escaneo de Pestañas ---")

    for hoja in sh_obs.worksheets():
        nombre_hoja = hoja.title.strip().upper()

        # Procesamos solo las que tienen MZ en el nombre
        if "MZ" in nombre_hoja:
            letra_mz = nombre_hoja.replace("MZ", "").strip()
            filas = hoja.get_all_values()
            if len(filas) < 2: continue

            for i, fila in enumerate(filas[1:], start=2):
                if len(fila) < 3: continue
                lote_raw = str(fila[0]).strip()
                partida = str(fila[1]).strip()
                estado = str(fila[2]).strip()
                comentario = str(fila[3]).strip() if len(fila) > 3 else "Sin detalle"

                try:
                    num_casa = int(float(lote_raw))
                except:
                    continue

                if estado.lower() == "en proceso":
                    key_partida = (letra_mz, num_casa, partida)
                    dict_observaciones[key_partida] = comentario
                    casas_con_obs.add((letra_mz, num_casa))

    print(f"\n--- RESUMEN FINAL ---")
    print(f"Total observaciones: {len(dict_observaciones)}")

except Exception as e:
    print(f"Advertencia: No se pudo cargar 'Pre F1': {e}")
    dict_observaciones = {}


# ========================================================
# 1. CARGA DE LISTA MAESTRA (DOBLE FILTRO: ITEM + NOMBRE)
# ========================================================
try:
    sh_maestro = gc.open('Partidas').sheet1
    filas_maestras = sh_maestro.get_all_values()

    lista_maestra_llaves = set()
    for fila in filas_maestras:
        if len(fila) >= 2:
            item_m = str(fila[0]).strip().upper()
            nom_m = str(fila[1]).strip().upper()
            if item_m and nom_m:
                lista_maestra_llaves.add(f"{item_m}-{nom_m}")

    print(f"✅ Maestro cargado: {len(lista_maestra_llaves)} combinaciones únicas.")

except Exception as e:
    lista_maestra_llaves = set()
    print(f"⚠️ Error cargando maestro: {e}")

def es_partida_maestra_estricta(codigo, nombre):
    return True

def normalizar(txt):
    if not txt: return ""
    txt = str(txt).lower()
    txt = "".join(c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn')
    txt = re.sub(r'[^a-z0-9]', '', txt)
    return txt

# ========================================================
# 2. PROCESO DE CRUCE
# ========================================================

dict_avances = {}
dict_detalles_casas = {}

for worksheet in sh.worksheets():
    sheet_name = worksheet.title
    if "MANZ" in sheet_name.upper():
        datos = worksheet.get_all_values()
        if not datos: continue

        letra_mz = sheet_name.replace("MANZ.", "").replace("MANZ", "").strip().upper()

        fila_item_idx = next((i for i, f in enumerate(datos[:50]) if f and str(f[0]).strip().upper() == "ITEM"), None)
        if fila_item_idx is None: continue

        columnas_casas = []
        for i_s in range(max(0, fila_item_idx - 2), min(len(datos), fila_item_idx + 3)):
            for c_idx, val in enumerate(datos[i_s]):
                if c_idx > 1 and str(val).strip().isdigit():
                    num_casa = int(str(val).strip())
                    if not any(x[1] == num_casa for x in columnas_casas):
                        columnas_casas.append((c_idx, num_casa))

        titulo_act = ""
        sub_act = ""

        for i in range(fila_item_idx + 1, len(datos)):
            fila = datos[i]
            if not fila or not str(fila[0]).strip(): continue

            item_val = str(fila[0]).strip()
            desc_val = str(fila[1]).strip()
            llave_actual = f"{item_val.upper()}-{desc_val.upper()}"

            if llave_actual not in lista_maestra_llaves:
                if "." not in item_val:
                    titulo_act = desc_val
                    sub_act = ""
                else:
                    sub_act = desc_val
                continue

            for col_idx, num_casa in columnas_casas:
                v_celda = fila[col_idx] if col_idx < len(fila) else ""
                terminado = (v_celda is not None and str(v_celda).strip() != "")

                key = (letra_mz, num_casa)
                if key not in dict_detalles_casas: dict_detalles_casas[key] = []

                dict_detalles_casas[key].append({
                    'titulo': titulo_act,
                    'subtitulo': sub_act,
                    'partida': f"[{item_val}] {desc_val}",
                    'estado': "✅" if terminado else "❌",
                    'tiene_obs': False,
                    'comentario': ""
                })

# 3. Cálculo de porcentajes
for key, lista in dict_detalles_casas.items():
    total = len(lista)
    hechas = sum(1 for p in lista if p['estado'] == "✅")
    dict_avances[key] = round((hechas / total) * 100, 1) if total > 0 else 0

# ========================================================
# BLOQUE: RE-ASIGNACIÓN DE IDS Y CLASIFICACIÓN (FORZADO)
# ========================================================

tipos_ref = {
    "Tipo B": ["D1", "F1", "F8", "I1", "I8", "K1", "K8", "L1", "L7"],
    "Tipo C": ["B3", "B4", "B5", "B6", "B7", "G1"],
    "Tipo D": ["B1", "B2"],
    "Tipo A2": ["E16", "E17"],
    "Tipo A1-N": ["E1", "D11", "F11", "I12", "J21"]
}

dict_tipos_vivienda = {}
nuevas_llaves_coherentes = []

print("🛠️ Re-vinculando geometrías con nombres de manzana reales...")

for i in range(len(mapa_manzanas)):
    mz_val = str(mapa_manzanas[i]).strip()
    num_val = str(mapa_numeros[i]).strip()
    id_busqueda = f"{mz_val}{num_val}".replace("MANZ.", "").replace(" ", "")

    v_tipo = "Tipo A1"
    for t_nombre, lista in tipos_ref.items():
        if id_busqueda in lista:
            v_tipo = t_nombre
            break

    dict_tipos_vivienda[(mapa_manzanas[i], mapa_numeros[i])] = v_tipo

print(f"✅ Clasificación forzada completada para {len(dict_tipos_vivienda)} polígonos.")

# ========================================================
# RECONSTRUCCIÓN DE AVANCES CON FILTRO POR TIPO DE VIVIENDA
# ========================================================

dict_detalles_casas_filtrado = {}
dict_avances_filtrado = {}

def partida_aplica_a_vivienda(codigo_partida, tipo_v, mz, casa):
    reglas = {
        "B.4.4.1": {"A1", "A1-N", "A2"},
        "B.4.4.2": {"A1", "A1-N", "A2"},
        "B.5.3.1": {"A1", "A1-N", "A2"},
        "C.2.3.1.B": {"A1-N"},
        "C.5.4": {"A1", "A1-N", "A2", "B"},
        "C.7.1": {"A1", "A1-N", "A2"},
        "C.9.3.1": {"A1", "A1-N", "A2", "B"},
        "C.12.1.4": {"C", "D"},
        "C.EX.3": {"A1-N", "D"},
        "C.EX.14.1": {"A1-N", "B", "C", "D"},
        "C.EX.15": {"A1-N", "B"},
        "C.EX.16": {"C", "D"},
        "D.1.2": {"A1", "A1-N", "A2", "B"},
        "D.1.3": {"C", "D"},
        "D.1.4": {"A1", "A1-N", "A2"},
        "D.1.5": {"B", "C", "D"},
        "D.1.7": {"A1", "A1-N", "A2", "B"},
        "D.1.8": {"C", "D"},
        "D.1.9": {"C", "D"},
        "D.1.10": {"C", "D"},
        "D.1.11": {"C", "D"},
        "D.1.12": {"C", "D"},
        "D.4.5.4": {"D"},
        "D.EX.3": {"A1-N", "D"},
        "D.EX.4": {"B"},
    }

    if codigo_partida not in reglas:
        return True

    return tipo_v in reglas[codigo_partida]


for key, lista_partidas in dict_detalles_casas.items():
    mz, casa = key
    tipo_v = dict_tipos_vivienda.get(key, "A1")
    filtradas = []

    for p in lista_partidas:
        partida_raw = p.get("partida", "")
        match = re.search(r'\[(.*?)\]', partida_raw)
        codigo = match.group(1).strip() if match else ""

        if partida_aplica_a_vivienda(codigo, tipo_v, mz, casa):
            filtradas.append(p)

    if filtradas:
        dict_detalles_casas_filtrado[key] = filtradas
        total_p = len(filtradas)
        hechas = sum(1 for p in filtradas if p['estado'] == "✅")
        dict_avances_filtrado[key] = round((hechas / total_p) * 100, 1)

avance_total_obra = round(
    sum(dict_avances_filtrado.values()) / len(dict_avances_filtrado), 1
)

print(f"🏗️ Avance total de la obra: {avance_total_obra}%")

# --- PASO FINAL: VINCULAR DATOS DEL EXCEL AL MAPA ---

count = 0
for (mz, casa_num), lista_partidas_excel in dict_detalles_casas.items():
    letra_mz_mapa = mz.replace("MZ", "").strip().upper()
    for p_excel in lista_partidas_excel:
        nombre_completo_excel = p_excel['partida'].strip().upper()
        for (mz_obs, casa_obs, partida_obs) in dict_observaciones.keys():
            if mz_obs == letra_mz_mapa and int(casa_obs) == int(casa_num):
                if partida_obs.upper() in nombre_completo_excel:
                    p_excel['tiene_obs'] = True
                    p_excel['comentario'] = dict_observaciones[(mz_obs, casa_obs, partida_obs)]
                    count += 1
                    break
        else:
            if 'tiene_obs' not in p_excel:
                p_excel['tiene_obs'] = False

REGLAS_PARTIDAS = {
    "B.4.4.1": {"tipos": {"Tipo A1", "Tipo A1-N", "Tipo A2"}},
    "B.4.4.2": {"tipos": {"Tipo A1", "Tipo A1-N", "Tipo A2"}},
    "B.5.3.1": {"tipos": {"Tipo A1", "Tipo A1-N", "Tipo A2"}},
    "C.2.3.1.B": {"tipos": {"Tipo A1-N"}},
    "C.5.4": {"tipos": {"Tipo A1", "Tipo A1-N", "Tipo A2", "Tipo B"}},
    "C.7.1": {"tipos": {"Tipo A1", "Tipo A1-N", "Tipo A2"}},
    "C.9.3.1": {"tipos": {"Tipo A1", "Tipo A1-N", "Tipo A2", "Tipo B"}},
    "C.12.1.4": {"tipos": {"Tipos C", "Tipo D"}},
    "C.EX.3":  {"tipos": {"Tipo A1-N", "Tipo D"}},
    "C.EX.14.1": {"tipos": {"Tipo A1-N", "Tipo B", "Tipo C", "Tipo D"}},
    "C.EX.15": {"tipos": {"Tipo A1-N", "Tipo B"}},
    "C.EX.16": {"tipos": {"Tipo C", "Tipo D"}},
    "C.EX.18": {"tipos": {"Tipo B", "Tipo C"}},
    "D.1.2": {"tipos": {"Tipo A1", "Tipo A1-N", "Tipo A2", "Tipo B"}},
    "D.1.3": {"tipos": {"Tipo C", "Tipo D"}},
    "D.1.4": {"tipos": {"Tipo A1", "Tipo A1-N", "Tipo A2"}},
    "D.1.5": {"tipos": {"Tipo B", "Tipo C", "Tipo D"}},
    "D.1.7": {"tipos": {"Tipo A1", "Tipo A1-N", "Tipo A2", "Tipo B"}},
    "D.1.8": {"tipos": {"Tipo C", "Tipo D"}},
    "D.1.9": {"tipos": {"Tipo C", "Tipo D"}},
    "D.1.10": {"tipos": {"Tipo C", "Tipo D"}},
    "D.1.11": {"tipos": {"Tipo C", "Tipo D"}},
    "D.1.12": {"tipos": {"Tipo C", "Tipo D"}},
    "D.4.5.4": {"tipos": {"Tipo D"}},
    "D.EX.3": {"tipos": {"Tipo A1-N", "Tipo D"}},
    "D.EX.4": {"tipos": {"Tipo B"}},
}

def extraer_codigo_partida(partida_raw):
    if not partida_raw: return None
    match = re.search(r'\[([A-Z0-9\.]+)\]', partida_raw)
    return match.group(1) if match else None

def partida_aplica(partida_raw, tipo_vivienda, manzana, casa):
    codigo = extraer_codigo_partida(partida_raw)
    if not codigo: return True
    if codigo not in REGLAS_PARTIDAS: return True
    regla = REGLAS_PARTIDAS[codigo]
    if tipo_vivienda in regla.get("tipos", set()): return True
    if "excepciones" in regla:
        if (str(manzana), str(casa)) in regla["excepciones"]: return True
    return False

def generar_html_popup(manzana, casa_num, detalles, tipo_vivienda, avance):
    detalles = [
        d for d in detalles
        if partida_aplica(d.get("partida"), tipo_vivienda, manzana, casa_num)
    ]
    letra_mz_mapa = manzana.replace("MZ", "").strip().upper()
    for d in detalles:
        nombre_excel = d.get('partida', '').strip().upper()
        match_encontrado = False
        for (mz_obs, casa_obs, partida_obs) in dict_observaciones.keys():
            if mz_obs == letra_mz_mapa and int(casa_obs) == int(casa_num):
                if partida_obs.upper() in nombre_excel:
                    d['tiene_obs'] = True
                    d['comentario'] = dict_observaciones[(mz_obs, casa_obs, partida_obs)]
                    match_encontrado = True
                    break
        if not match_encontrado: d['tiene_obs'] = False

    resumen = {}
    for d in detalles:
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
    for item in detalles:
        if item['titulo'] != current_tit:
            current_tit = item['titulo']
            anchor_tit = f"tit_{abs(hash(current_tit))}"
            html += f'<tr id="{anchor_tit}" style="background: #edeff0;"><td colspan="2" style="padding: 10px 5px; font-weight: bold; color: #2c3e50; border-top: 2px solid #2c3e50;">{current_tit.upper()}</td></tr>'
        if item['subtitulo'] != current_sub:
            current_sub = item['subtitulo']
            if current_sub:
                anchor_sub = f"sub_{abs(hash(current_sub))}"
                html += f'<tr id="{anchor_sub}" style="background: #f9f9f9;"><td colspan="2" style="padding: 6px 8px; font-weight: bold; color: #7f8c8d; font-style: italic; border-bottom: 1px solid #eee;"> ↳ {current_sub}</td></tr>'

        if item.get('tiene_obs'):
            color_st = "#d4a017"
            icono_mostrado = "⚠️"
            comentario = item.get("comentario", "Sin detalle")
            nombre_partida = f"""<div style="padding: 2px 0;"><b style="color: #d4a017;">{item['partida']}</b><details style="margin-top: 4px;"><summary style="cursor: pointer; color: #856404; font-size: 10px; font-weight: bold;">Ver nota [+]</summary><div style="margin-top: 4px; padding: 8px; background: #fff9e6; border-left: 3px solid #d4a017; color: #856404; font-size: 10px; line-height: 1.4;">{comentario}</div></details></div>"""
        else:
            color_st = "#27ae60" if item['estado'] == "✅" else "#e74c3c"
            icono_mostrado = item['estado']
            nombre_partida = f"<span style='color: #444; font-size: 11px;'>{item['partida']}</span>"
        html += f"""<tr style="border-bottom: 1px solid #f2f2f2;"><td style="padding: 8px 10px; vertical-align: top;">{nombre_partida}</td><td style="padding: 8px 5px; text-align: center; color: {color_st}; font-weight: bold; font-size: 14px;">{icono_mostrado}</td></tr>"""

    html += """</table></div><div style="flex: 1.2; background: #f4f7f8; padding: 10px; overflow-y: auto; border-left: 1px solid #ddd;"><div style="font-size: 11px; font-weight: bold; color: #95a5a6; margin-bottom: 10px; text-align: center; border-bottom: 1px solid #ccc; padding-bottom: 5px;">ÍNDICE DE CONTROL</div>"""
    for tit, datos in resumen.items():
        anchor_tit = f"tit_{abs(hash(tit))}"
        bg_tit = "#fff3cd" if datos['obs'] else "#fff"
        html += f"""<div onclick="document.getElementById('{anchor_tit}').scrollIntoView({{behavior:'smooth'}})" style="cursor: pointer; padding: 6px; background: {bg_tit}; border: 1px solid #dcdde1; border-radius: 4px; margin-bottom: 4px;"><div style="font-weight: bold; color: #2c3e50; font-size: 10px;">{tit}</div><div style="font-size: 9px; color: {'#856404' if datos['obs'] else '#27ae60'};">{datos['listo']}/{datos['total']} completados</div></div>"""
        for subtit, sdatos in datos['subs'].items():
            anchor_sub = f"sub_{abs(hash(subtit))}"
            estilo_s = "color:#856404;font-weight:bold;" if sdatos['obs'] else "color:#636e72;"
            html += f"""<div onclick="document.getElementById('{anchor_sub}').scrollIntoView({{behavior:'smooth'}})" style="cursor:pointer; padding:4px 6px 4px 15px; margin-bottom:3px; border-left:2px solid {'#f1c40f' if sdatos['obs'] else '#bdc3c7'}; font-size:9px; {estilo_s}">{subtit} {'(!)' if sdatos['obs'] else ''}</div>"""
    html += "</div></div></div>"
    return html

# =======================================================
# PASO 1: CARGA DE LA MATRIZ DE PRECIOS DE TRATOS
# =======================================================
nombre_archivo_tratos = 'Tratos - Campos del Sur II 3.0 AGOSTO 2025 modificado ultimo'
try:
    sh_tratos = gc.open(nombre_archivo_tratos)
    ws_tratos = sh_tratos.worksheet('TRATOS VIVIENDA')
    datos_tratos = ws_tratos.get_all_values()
    print("✅ Archivo de Tratos detectado.")
except Exception as e:
    print(f"⚠️ Error al abrir el archivo: {e}")
    datos_tratos = []

precios_tratos = {}
estructura_tratos = []
titulo_actual = ""
subtitulo_actual = ""

def limpiar_monto(valor):
    try:
        if not valor: return 0.0
        v = str(valor).replace('$', '').replace('.', '').replace(',', '.').strip()
        return float(v)
    except:
        return 0.0

if datos_tratos:
    # Lista extendida de títulos para asegurar que no se escape ninguno
    titulos_validos = [
        "OBRA GRUESA",
        "OBRAS DE TERMINACIÓN",
        "INSTALACIONES"
    ]

    for i, fila in enumerate(datos_tratos):
        # Empezamos a leer desde la fila 7 (índice 6)
        if i < 6: continue

        nombre_col = str(fila[1]).strip() if len(fila) > 1 else ""
        if not nombre_col: continue

        unidad = str(fila[3]).strip() if len(fila) > 3 else ""

        # --- NUEVA LÓGICA DE DETECCIÓN PRIORIZADA ---

        # 1. Prioridad 1: ¿Es un Título Principal?
        # (Ignoramos la unidad porque "OBRA GRUESA" tiene "A-1" en esa columna)
        if nombre_col.upper() in titulos_validos:
            titulo_actual = nombre_col.upper()
            subtitulo_actual = ""
            continue # Pasamos a la siguiente fila

        # 2. Prioridad 2: ¿Es un Subtítulo?
        # (Unidad vacía y texto en mayúsculas suele ser subtítulo como "FUNDACIONES")
        if unidad == "" and nombre_col.isupper():
            subtitulo_actual = nombre_col
            continue

        # 3. Prioridad 3: ¿Es una Partida?
        # (Si tiene unidad y no es un título de los de arriba, es cobrable)
        if unidad != "" and unidad not in ["A-1", "A-2", "B", "C", "D"]:
            precios_tratos[nombre_col] = {
                "Tipo A1": limpiar_monto(fila[5]), "Tipo A1-N": limpiar_monto(fila[5]),
                "Tipo A2": limpiar_monto(fila[9]), "Tipo B": limpiar_monto(fila[13]),
                "Tipo C": limpiar_monto(fila[17]), "Tipo D": limpiar_monto(fila[21])
            }
            estructura_tratos.append({
                "titulo": titulo_actual,
                "subtitulo": subtitulo_actual,
                "partida": nombre_col
            })

# =======================================================
# LECTURA DE TRATOS Y ASIGNACIÓN DE CUADRILLAS
# =======================================================
nombre_archivo_sheets = "Asignación Tratos"

try:
    sh_asignacion = gc.open(nombre_archivo_sheets)
    print("✅ Archivo 'Asignación Tratos' conectado.")
except Exception as e:
    print(f"⚠️ Error: {e}")

ws_cuadrillas = sh_asignacion.worksheet('CUADRILLAS')
datos_cuadrillas = ws_cuadrillas.get_all_values()

# CAMBIO: Usamos f[0] (Columna CUADRILLA) en lugar de f[1] (JEFE CUADRILLA)
dict_maestro_cuadrillas = {str(f[2]).strip(): str(f[0]).strip() for f in datos_cuadrillas[1:] if len(f) >= 3 and f[2]}

estado_tratos = {}
cuadrillas_tratos = {}
manzanas_a_procesar = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

for letra in manzanas_a_procesar:
    nombre_hoja = f"MZ {letra}"
    try:
        ws_mz = sh_asignacion.worksheet(nombre_hoja)
        datos_mz = ws_mz.get_all_values()
        if len(datos_mz) < 3: continue

        encabezados = datos_mz[2]

        for i in range(1, len(encabezados), 2):
            raw_casa = str(encabezados[i]).upper().replace("CASA", "").replace("LOTE", "").replace("N°", "").strip()
            if not raw_casa: continue
            try:
                casa_num = int(float(raw_casa))
            except:
                continue

            for fila in datos_mz[3:]:
                if not fila or not fila[0].strip(): continue

                # Normalización del nombre para comparar
                partida_nombre = " ".join(str(fila[0]).split()).strip()

                if partida_nombre in ["TRATO", "FUNDACIONES", "RADIER", "MUROS 1ER PISO", "TERMINACIONES"]:
                    continue

                id_en_celda = str(fila[i]).strip() if i < len(fila) else ""
                fecha_en_celda = str(fila[i+1]).strip() if (i+1) < len(fila) and str(fila[i+1]).strip() else "-"

                llave = (letra, casa_num, partida_nombre)

                if id_en_celda and id_en_celda != "0":
                    nombre_cuadrilla = dict_maestro_cuadrillas.get(id_en_celda, f"ID: {id_en_celda}")
                    estado_tratos[llave] = {"terminada": True, "fecha": fecha_en_celda}
                    cuadrillas_tratos[llave] = nombre_cuadrilla
                else:
                    estado_tratos[llave] = {"terminada": False, "fecha": "-"}
                    cuadrillas_tratos[llave] = "-"

    except Exception as e:
        print(f"Aviso en MZ {letra}: {e}")

# =======================================================
# PASO 3: FUNCIONES DE INTERFAZ (PESTAÑA TRATOS) - CORREGIDO
# =======================================================

def color_gradiente_plata(ganado, total):
    if total <= 0: return "#ecf0f1"

    p = max(0.0, min(1.0, float(ganado) / float(total)))

    if p < 0.5:
        p_local = p * 2
        r = int(52 + (241 - 52) * p_local)
        g = int(152 + (196 - 152) * p_local)
        b = int(219 + (15 - 219) * p_local)
    else:
        p_local = (p - 0.5) * 2
        r = int(241 + (39 - 241) * p_local)
        g = int(196 + (174 - 196) * p_local)
        b = int(15 + (96 - 15) * p_local)

    return f"#{r:02x}{g:02x}{b:02x}"

def formatear_plata(valor):
    if valor <= 0: return "$ -"
    return f"${int(valor):,} pesos".replace(",", ".")

def generar_html_popup_tratos(manzana, casa_num, tipo_vivienda):
    resumen = {}
    plata_ganada = 0.0
    plata_total = 0.0
    detalles_html = ""
    current_tit, current_sub = None, None

    for item in estructura_tratos:
        tit = item['titulo']
        sub = item['subtitulo']
        partida = item['partida']
        partida_limpia = " ".join(str(partida).split()).strip()

        precio_partida = precios_tratos.get(partida, {}).get(tipo_vivienda, 0.0)
        llave = (manzana, casa_num, partida_limpia)
        estado = estado_tratos.get(llave, {"terminada": False, "fecha": "-"})
        cuadrilla = cuadrillas_tratos.get(llave, "-")

        plata_total += precio_partida
        if estado["terminada"]:
            plata_ganada += precio_partida

        if tit not in resumen:
            resumen[tit] = {'total': 0, 'ganado': 0, 'subs': {}}

        resumen[tit]['total'] += precio_partida
        if estado["terminada"]:
            resumen[tit]['ganado'] += precio_partida

        if sub:
            if sub not in resumen[tit]['subs']:
                resumen[tit]['subs'][sub] = {'total': 0, 'ganado': 0}
            resumen[tit]['subs'][sub]['total'] += precio_partida
            if estado["terminada"]:
                resumen[tit]['subs'][sub]['ganado'] += precio_partida

        if tit != current_tit:
            current_tit = tit
            anchor_tit = f"tratos_tit_{abs(hash(current_tit))}"
            detalles_html += f'<tr id="{anchor_tit}" style="background: #edeff0;"><td colspan="4" style="padding: 10px 5px; font-weight: bold; color: #2c3e50; border-top: 2px solid #2c3e50;">{current_tit.upper()}</td></tr>'

        if sub != current_sub:
            current_sub = sub
            if current_sub:
                anchor_sub = f"tratos_sub_{abs(hash(current_sub))}"
                detalles_html += f'<tr id="{anchor_sub}" style="background: #fdfdfd;"><td colspan="4" style="padding: 6px 8px; font-weight: bold; color: #7f8c8d; font-style: italic; border-bottom: 1px solid #eee;"> ↳ {current_sub}</td></tr>'

        color_st = "#27ae60" if estado["terminada"] else "#e74c3c"
        icono_mostrado = "✅" if estado["terminada"] else "❌"

        detalles_html += f"""
        <tr class="fila-trato" data-cuadrilla="{cuadrilla}" style="border-bottom: 1px solid #f2f2f2;">
            <td style="padding: 8px 5px; vertical-align: middle;"><span style='color: #444; font-size: 10px;'>{partida}</span></td>
            <td style="padding: 8px 5px; text-align: center; font-size: 10px; color: #555;"><b>{formatear_plata(precio_partida)}</b></td>
            <td style="padding: 8px 5px; text-align: center; font-size: 9px; color: #777;">{cuadrilla}<br><span style="color:#aaa">{estado['fecha']}</span></td>
            <td style="padding: 8px 5px; text-align: center; color: {color_st}; font-weight: bold; font-size: 12px;">{icono_mostrado}</td>
        </tr>"""

    html_final = f"""
    <div style="font-family: 'Segoe UI', Arial; width: 720px; background: white; margin: -15px -10px -10px -10px; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
        <div style="background: #1abc9c; color: white; padding: 15px; display: flex; justify-content: space-between; align-items: center;">
            <h4 style="margin: 0; font-size: 16px;">MZ {manzana} - Casa {casa_num} - {tipo_vivienda}</h4>
            <div style="text-align: right;">
                <div style="font-size: 11px; opacity: 0.9;">Total Pagado</div>
                <div style="font-size: 18px; font-weight: bold;">{formatear_plata(plata_ganada)}</div>
            </div>
        </div>

        <div style="display: flex; height: 450px;">
            <div style="flex: 2; overflow-y: auto; padding: 10px; border-right: 1px solid #ddd;">
                <table style="width: 100%; border-collapse: collapse; table-layout: fixed;">
                    <colgroup><col style="width: 45%;"><col style="width: 20%;"><col style="width: 25%;"><col style="width: 10%;"></colgroup>
                    <thead><tr style="background:#f4f7f8; font-size:10px; color:#555; border-bottom: 2px solid #ddd;"><th style="padding:8px; text-align:left;">Partida</th><th style="padding:8px;">Precio</th><th style="padding:8px;">Cuadrilla</th><th style="padding:8px;">Est.</th></tr></thead>
                    <tbody>{detalles_html}</tbody>
                </table>
            </div>

            <div style="flex: 1.1; background: #f8f9fa; padding: 10px; overflow-y: auto;">
                <div style="font-size: 11px; font-weight: bold; color: #7f8c8d; margin-bottom: 12px; text-align: center; border-bottom: 1px solid #ccc; padding-bottom: 5px;">RESUMEN Y SUBTOTALES</div>
    """

    for tit, datos in resumen.items():
        if datos['total'] == 0: continue
        anchor_tit = f"tratos_tit_{abs(hash(tit))}"
        pct_tit = (datos['ganado'] / datos['total']) * 100

        # AÑADIMOS class="btn-indice"
        html_final += f"""
        <div class="btn-indice" onclick="document.getElementById('{anchor_tit}').scrollIntoView({{behavior:'smooth'}})"
             style="cursor: pointer; padding: 8px; background: #2c3e50; border-radius: 4px; margin-bottom: 5px; color: white;">
            <div style="font-weight: bold; font-size: 10px; text-transform: uppercase;">{tit}</div>
            <div style="font-size: 10px; color: #1abc9c; font-weight: bold;">{formatear_plata(datos['ganado'])}</div>
            <div style="width: 100%; background: rgba(255,255,255,0.2); height: 4px; border-radius: 2px; margin-top: 4px;">
                <div style="width: {pct_tit}%; background: #1abc9c; height: 100%; border-radius: 2px;"></div>
            </div>
        </div>"""

        for sub, dsub in datos['subs'].items():
            if dsub['total'] == 0: continue
            anchor_sub = f"tratos_sub_{abs(hash(sub))}"
            pct_sub = (dsub['ganado'] / dsub['total']) * 100

            # AÑADIMOS class="btn-indice"
            html_final += f"""
            <div class="btn-indice" onclick="document.getElementById('{anchor_sub}').scrollIntoView({{behavior:'smooth'}})"
                  style="cursor: pointer; padding: 6px 6px 6px 12px; background: white; border: 1px solid #dcdde1; border-radius: 4px; margin-bottom: 4px; margin-left: 10px;">
                <div style="font-weight: bold; color: #34495e; font-size: 9px;">↳ {sub}</div>
                <div style="font-size: 9px; color: #27ae60;">{formatear_plata(dsub['ganado'])} / {int(pct_sub)}%</div>
                <div style="width: 100%; background: #eee; height: 3px; border-radius: 2px; margin-top: 3px;">
                    <div style="width: {pct_sub}%; background: #27ae60; height: 100%; border-radius: 2px;"></div>
                </div>
            </div>"""

    html_final += "</div></div></div>"
    return html_final, plata_ganada, plata_total

print("✅ PASO 3: Clases 'btn-indice' añadidas para habilitar el resaltado.")

# =======================================================
# PASO PREVIO: PREPARACIÓN DE LISTA DE CUADRILLAS
# =======================================================

# Creamos un diccionario para saber qué cuadrilla tiene cada casa (Mz, Num)
# y una lista única de nombres de cuadrillas para el menú
dict_cuadrillas_casas = {}
nombres_cuadrillas_unicas = set()

for (mz, num, partida), nombre_c in cuadrillas_tratos.items():
    if nombre_c and nombre_c != "Sin Asignar":
        dict_cuadrillas_casas[(mz, num)] = nombre_c
        nombres_cuadrillas_unicas.add(nombre_c)

# Generamos el HTML de los botones Radio
html_opciones_cuadrillas = '<label style="display:block; margin:5px; cursor:pointer;"><input type="radio" name="c_sel" onclick="filtrarC(\'TODAS\')" checked> <b>VER TODAS</b></label>'
for nombre in sorted(list(nombres_cuadrillas_unicas)):
    html_opciones_cuadrillas += f'<label style="display:block; margin:5px; cursor:pointer;"><input type="radio" name="c_sel" onclick="filtrarC(\'{nombre}\')"> {nombre}</label>'

print(f"✅ Se han detectado {len(nombres_cuadrillas_unicas)} cuadrillas listas para el mapa.")

# =======================================================
# PASO 3.5: PROCESAMIENTO DE DATOS PARA PANEL DE CUADRILLAS
# =======================================================

# 1. Información Maestra de Cuadrillas
dict_info_maestra_cuadrillas = {}
if 'datos_cuadrillas' in locals():
    for f in datos_cuadrillas[1:]:
        if len(f) >= 3:
            c_name = str(f[0]).strip().upper()
            dict_info_maestra_cuadrillas[c_name] = {
                "representante": str(f[1]).strip(),
                "id": str(f[2]).strip()
            }
    print(f"DEBUG: {len(dict_info_maestra_cuadrillas)} cuadrillas cargadas.")

# 2. Inicializar variables para el mapa y JS
info_cuadrillas_js = {}
dict_cuadrillas_por_casa = {} # <--- ESTO ES LO QUE FALTABA
todas_cuadrillas_set = set()

print("DEBUG: Iniciando mapeo y cálculo de montos...")
conteo_exitos = 0
conteo_errores = 0

# 3. Recorrido de asignaciones (cuadrillas_tratos viene de tu celda de Sheets)
for (letra_mz, casa_num, partida_nombre), nombre_cuadrilla in cuadrillas_tratos.items():
    if nombre_cuadrilla and nombre_cuadrilla != "-":
        c_limpia = str(nombre_cuadrilla).strip().upper()
        todas_cuadrillas_set.add(c_limpia)

        # A. Llenar dict_cuadrillas_por_casa para el GeoJson del mapa
        # Usamos la misma llave que usará el mapa: (MZ_LIMPIA, NUMERO)
        letra_limpia = str(letra_mz).upper().replace("MZ", "").strip()
        key_casa = (letra_limpia, int(casa_num))

        if key_casa not in dict_cuadrillas_por_casa:
            dict_cuadrillas_por_casa[key_casa] = set()
        dict_cuadrillas_por_casa[key_casa].add(c_limpia)

        # B. Calcular montos para el Panel Lateral
        if c_limpia not in info_cuadrillas_js:
            info_m = dict_info_maestra_cuadrillas.get(c_limpia, {"representante": "No asignado", "id": "-"})
            info_cuadrillas_js[c_limpia] = {
                "representante": info_m["representante"],
                "id": info_m["id"],
                "total_pagado": 0.0,
                "tratos_realizados": []
            }

        # Buscar precio
        mz_formateada = f"MZ {letra_mz}"
        tipo_v = dict_tipos_vivienda.get((mz_formateada, int(casa_num)), "Tipo A1")

        valor_partida = 0.0
        if partida_nombre in precios_tratos:
            valor_partida = precios_tratos[partida_nombre].get(tipo_v, 0.0)

        if valor_partida > 0:
            info_cuadrillas_js[c_limpia]["total_pagado"] += valor_partida
            info_cuadrillas_js[c_limpia]["tratos_realizados"].append({
                "casa": f"Mz {letra_mz} - Casa {casa_num} ({partida_nombre})",
                "monto": valor_partida
            })
            conteo_exitos += 1
        else:
            conteo_errores += 1

# Convertir los sets de cuadrillas a listas para que sean serializables a JSON
for k in dict_cuadrillas_por_casa:
    dict_cuadrillas_por_casa[k] = list(dict_cuadrillas_por_casa[k])

print(f"DEBUG: Mapeo terminado.")
print(f"DEBUG: Partidas con monto: {conteo_exitos}")
print(f"DEBUG: Partidas sin monto: {conteo_errores}")

# 4. Generar HTML de opciones
html_opciones_cuadrillas = '<div onclick="filtrarC(\'TODAS\')" style="cursor:pointer; padding:8px; border-bottom:1px solid #eee; font-weight:bold; color:#2c3e50;">• TODAS</div>'
for c in sorted(info_cuadrillas_js.keys()):
    html_opciones_cuadrillas += f'''
    <div class="item-cuadrilla" style="border-bottom:1px solid #eee; display:flex; align-items:center;">
        <div onclick="filtrarC(\'{c}\')" style="cursor:pointer; padding:8px; flex-grow:1; font-size:12px;">• {c}</div>
        <div onclick="toggleDetalleCuadrilla(\'{c}\')" style="cursor:pointer; padding:8px 12px; color:#1abc9c; font-weight:bold; border-left:1px solid #eee;">→</div>
    </div>'''


# --- 1. REPARACIÓN DE DATOS DE CUADRILLAS ---
dict_info_maestra_cuadrillas = {}
try:
    for fila in datos_cuadrillas[1:]:
        if len(fila) >= 3:
            c_name = str(fila[0]).strip().upper()
            dict_info_maestra_cuadrillas[c_name] = {
                "representante": str(fila[1]).strip(),
                "id": str(fila[2]).strip()
            }
except: pass

# --- 2. MAPEO MULTI-CUADRILLA POR CASA ---
dict_cuadrillas_por_casa = {}
todas_cuadrillas_set = set()

for (mzn_t, casa_t, partida_t), cuad in cuadrillas_tratos.items():
    if cuad and cuad != "-":
        c_limpia = str(cuad).strip().upper()
        key_casa = (str(mzn_t).upper().replace("MZ", "").strip(), int(casa_t))
        if key_casa not in dict_cuadrillas_por_casa:
            dict_cuadrillas_por_casa[key_casa] = set()
        dict_cuadrillas_por_casa[key_casa].add(c_limpia)
        todas_cuadrillas_set.add(c_limpia)

# --- LÓGICA DE COLORES SEGÚN TU SOLICITUD ---
def obtener_color_estatico(avance, tiene_obs):
    if tiene_obs: return "#F2FF0D"
    if avance >= 100: return "#00FF19"
    if 85 < avance <= 99: return "#00F2FF"
    if 70 < avance <= 85: return "#000DFF"
    if 50 < avance <= 70: return "#C300FF"
    if 30 <= avance <= 50: return "#FF00AA"
    if 10 <= avance < 30: return "#FF8400"
    if avance < 10: return "#D10000"
    return "#D10000"

# --- 3. PREPARACIÓN DE OPCIONES E INFO PARA PANEL ---
info_cuadrillas_js = {}
todas_cuadrillas = sorted(list(todas_cuadrillas_set))
html_opciones_cuadrillas = '<div onclick="filtrarC(\'TODAS\')" style="cursor:pointer; padding:8px; border-bottom:1px solid #eee; font-weight:bold; color:#2c3e50;">• TODAS</div>'

for c in todas_cuadrillas:
    info_m = dict_info_maestra_cuadrillas.get(c, {"representante": "No asignado", "id": "-"})
    info_cuadrillas_js[c] = {"representante": info_m["representante"], "id": info_m["id"], "total_pagado": 0.0, "tratos_realizados": []}
    html_opciones_cuadrillas += f'''
    <div class="item-cuadrilla" style="border-bottom:1px solid #eee; display:flex; align-items:center;">
        <div onclick="filtrarC('{c}')" style="cursor:pointer; padding:8px; flex-grow:1; font-size:12px;">• {c}</div>
        <div onclick="toggleDetalleCuadrilla('{c}')" style="cursor:pointer; padding:8px 12px; color:#1abc9c; font-weight:bold; border-left:1px solid #eee;">→</div>
    </div>'''

# --- 4. CONFIGURACIÓN DEL MAPA ---
limites = [[0, 0], [h, w]]
m = folium.Map(location=[h/2, w/2], zoom_start=0, crs='Simple', tiles=None)

fg_fisico = folium.FeatureGroup(name="Avance Físico", show=True)
fg_tratos = folium.FeatureGroup(name="Avance Tratos", show=False)

for grupo in [fg_fisico, fg_tratos]:
    folium.raster_layers.ImageOverlay(image='plano2.png', bounds=[[0, 0], [h, w]], opacity=1, zindex=1).add_to(grupo)

total_plata_obra = 0.0
total_posible_obra = 0.0

# --- 5. DIBUJO DE CASAS ---
for i, geo in enumerate(casas_geometria):
    mz = str(mapa_manzanas.get(i, "SIN"))
    try: num = int(float(mapa_numeros.get(i, 0)))
    except: num = 0
    key = (mz, num)
    tipo_v = dict_tipos_vivienda.get(key, "Tipo A1")
    letra_mz_limpia = mz.replace("MZ", "").strip().upper()
    key_busqueda = (letra_mz_limpia, num)

    # ----- A. VISTA AVANCE FÍSICO -----
    avance_fisico = dict_avances_filtrado.get(key, 0)
    detalles_fisicos = dict_detalles_casas_filtrado.get(key, [])
    tiene_observacion = False
    for d in detalles_fisicos:
        nombre_excel = d.get('partida', '').strip().upper()
        for (mz_obs, casa_obs, partida_obs) in dict_observaciones.keys():
            if str(mz_obs).upper() == letra_mz_limpia and int(casa_obs) == int(num):
                if str(partida_obs).upper() in nombre_excel:
                    d['tiene_obs'] = True
                    d['comentario'] = dict_observaciones[(mz_obs, casa_obs, partida_obs)]
                    tiene_observacion = True
                    break
        else:
            if 'tiene_obs' not in d: d['tiene_obs'] = False

    color_fisico = obtener_color_estatico(avance_fisico, tiene_observacion)
    popup_html_fisico = generar_html_popup(mz, num, detalles_fisicos, tipo_v, avance_fisico)

    folium.GeoJson(
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [geo]}, "properties": {"manzana": mz, "numero": num, "tipo": tipo_v, "avance": avance_fisico, "etiqueta": f"""<div style="font-size:12px;font-weight:bold;text-align:right;">{avance_fisico}%</div><div style="background:#e0e0e0;height:6px;border-radius:4px;overflow:hidden;"><div style="width:{avance_fisico}%;height:100%;background:linear-gradient(90deg,#2980b9,#27ae60);"></div></div>"""}},
        style_function=lambda x, c=color_fisico: {"fillColor": c, "fillOpacity": 0.5, "weight": 1.2, "color": "black"},
        highlight_function=lambda x: {"fillOpacity": 0.8, "weight": 2.5},
        tooltip=folium.GeoJsonTooltip(fields=["manzana", "numero", "tipo", "etiqueta"], aliases=["Manzana:", "Casa Nº:", "Tipo:", "Físico:"], style="background-color: white; border: 1px solid black; border-radius: 6px; font-family: Arial; font-size: 12px;")
    ).add_child(folium.Popup(popup_html_fisico, max_width=520)).add_to(fg_fisico)

    # ----- B. VISTA TRATOS -----
    popup_html_tratos, plata_g, plata_t = generar_html_popup_tratos(mz, num, tipo_v)
    total_plata_obra += plata_g
    total_posible_obra += plata_t
    color_tratos_val = color_gradiente_plata(plata_g, plata_t)

    lista_cuadrillas_casa = list(dict_cuadrillas_por_casa.get(key_busqueda, []))

    for c_nombre in lista_cuadrillas_casa:
        if c_nombre in info_cuadrillas_js:
            monto_cuadrilla_en_esta_casa = 0
            desglose_partidas = []
            for (m_t, c_t, p_t), cuad_asig in cuadrillas_tratos.items():
                m_t_limp = str(m_t).upper().replace("MZ", "").strip()
                if m_t_limp == letra_mz_limpia and int(c_t) == num and str(cuad_asig).strip().upper() == c_nombre:
                    if p_t in precios_tratos:
                        valor_p = precios_tratos[p_t].get(tipo_v, 0.0)
                        monto_cuadrilla_en_esta_casa += valor_p
                        desglose_partidas.append({"nombre": p_t, "precio": valor_p})

            if monto_cuadrilla_en_esta_casa > 0:
                info_cuadrillas_js[c_nombre]["total_pagado"] += monto_cuadrilla_en_esta_casa
                info_cuadrillas_js[c_nombre]["tratos_realizados"].append({
                    "casa": f"Mz {mz} - Casa {num}",
                    "mzn_sort": str(mz).upper().strip(),
                    "num_sort": int(num),
                    "monto_total": monto_cuadrilla_en_esta_casa,
                    "partidas": desglose_partidas
                })

    folium.GeoJson(
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [geo]},
         "properties": {
             "manzana": mz, "numero": num, "tipo": tipo_v,
             "cuadrillas_list": lista_cuadrillas_casa,
             "color_base": color_tratos_val,
             "etiqueta": f"""<div style="font-size:12px;font-weight:bold;color:#27ae60;">Gastado: {formatear_plata(plata_g)}</div><div style="font-size:10px;color:#7f8c8d;">Presupuestado: {formatear_plata(plata_t)}</div>"""
         }},
        style_function=lambda x, c=color_tratos_val: {"fillColor": c, "fillOpacity": 0.7, "weight": 1.2, "color": "black"},
        # NOTA: Quitamos highlight_function de aquí para manejarlo por JS y que no haya conflicto
        tooltip=folium.GeoJsonTooltip(fields=["manzana", "numero", "tipo", "etiqueta"], aliases=["Manzana:", "Casa Nº:", "Tipo:", "Trato:"], style="background-color: white; border: 1px solid black; border-radius: 6px; font-family: Arial; font-size: 12px;")
    ).add_child(folium.Popup(popup_html_tratos, max_width=680)).add_to(fg_tratos)

# --- 6. OVERLAY HTML ---
overlay_html = r'''
{% macro html(this, kwargs) %}
<style>
.leaflet-control-layers { display: none !important; }
#panel-detalle-cuadrilla { position: fixed; top: 0; right: -400px; width: 350px; height: 100%; background: white; z-index: 10000; box-shadow: -5px 0 15px rgba(0,0,0,0.1); transition: right 0.3s ease; font-family: 'Segoe UI', Arial; display: flex; flex-direction: column; }
#panel-detalle-cuadrilla.active { right: 0; }
.item-cuadrilla:hover { background: #f9f9f9; }
.casa-header:hover { background: #f5f5f5; }
</style>

<a href="https://maximilianoazar.github.io/obra-campos-del-sur-ii/index.html" style="position: fixed; top: 20px; left: 60px; z-index: 9999; background: white; color: #2c3e50; text-decoration: none; padding: 10px 18px; border-radius: 50px; font-family: 'Segoe UI', Arial; font-size: 14px; font-weight: 600; box-shadow: 0 4px 12px rgba(0,0,0,0.15); border: 1px solid #eee; display: flex; align-items: center; gap: 8px;">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>
    <span>Volver al Inicio</span>
</a>

<div id="btn-toggle-view" onclick="toggleVista()" style="position: fixed; top: 780px; right: 20px; z-index: 9999; cursor: pointer; background: linear-gradient(135deg, #34495e, #2c3e50); color: white; padding: 12px 24px; border-radius: 8px; font-family: 'Segoe UI', Arial; font-size: 14px; font-weight: bold; border: 1px solid #1abc9c; display: flex; align-items: center; gap: 10px;">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1abc9c" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 2v6h-6"></path><path d="M3 12a9 9 0 0 1 15-6.7L21 8"></path><path d="M3 22v-6h6"></path><path d="M21 12a9 9 0 0 1-15 6.7L3 16"></path></svg>
    <span id="texto-toggle">Cambiar Pestaña a Tratos</span>
</div>

<div id="tarjeta-fisico" style="position: fixed; top: 20px; right: 20px; z-index: 9999; background: white; padding: 16px; border-radius: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.25); font-family: 'Segoe UI', Arial; width: 220px; transition: opacity 0.3s;">
    <div style="font-weight: bold; font-size: 13px; color: #555; margin-bottom: 8px;">Avance Físico Obra</div>
    <div style="font-size: 26px; font-weight: bold; color: #2c7be5; text-align: center; margin-bottom: 8px;">''' + str(avance_total_obra) + r'''%</div>
    <div style="background: #e0e0e0; border-radius: 8px; height: 12px; overflow: hidden;"><div style="width: ''' + str(avance_total_obra) + r'''%; height: 100%; background: linear-gradient(90deg, #27ae60, #2ecc71);"></div></div>
</div>

<div id="tarjeta-tratos" style="position: fixed; top: 20px; right: 20px; z-index: 9999; background: white; padding: 16px; border-radius: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.25); font-family: 'Segoe UI', Arial; width: 220px; opacity: 0; pointer-events: none; transition: opacity 0.3s;">
    <div style="font-weight: bold; font-size: 13px; color: #555; margin-bottom: 8px;">Plata Pagada (Tratos)</div>
    <div style="font-size: 19px; font-weight: bold; color: #1abc9c; text-align: center; margin-bottom: 8px;">''' + str(formatear_plata(total_plata_obra)) + r'''</div>
    <div style="font-size: 10px; color: #7f8c8d; text-align: center;">De ''' + str(formatear_plata(total_posible_obra)) + r''' presupuestados</div>
</div>

<div id="leyenda-fisico" style="position: fixed; bottom: 70px; left: 20px; z-index: 9999; background: rgba(255, 255, 255, 0.9); padding: 12px 18px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15); font-family: 'Segoe UI', Arial; border: 1px solid rgba(0,0,0,0.05); backdrop-filter: blur(8px);">
    <div style="font-weight: bold; font-size: 13px; margin-bottom: 8px; color: #333; text-transform: uppercase; letter-spacing: 0.5px;">Referencia de Avance</div>
    <div style="display: grid; grid-template-columns: repeat(4, auto); gap: 10px 20px; align-items: center;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 14px; height: 14px; background: #F2FF0D; border-radius: 50%; border: 1px solid #d4d400; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: #333;">!</div>
            <span style="font-size: 11px; color: #444; white-space: nowrap;">Observaciones</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 14px; height: 14px; background: #00FF19; border-radius: 3px; border: 1px solid rgba(0,0,0,0.1);"></div>
            <span style="font-size: 11px; color: #444; white-space: nowrap;">100% Finalizado</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 14px; height: 14px; background: #00F2FF; border-radius: 3px; border: 1px solid rgba(0,0,0,0.1);"></div>
            <span style="font-size: 11px; color: #444; white-space: nowrap;">85% - 99%</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 14px; height: 14px; background: #000DFF; border-radius: 3px; border: 1px solid rgba(0,0,0,0.1);"></div>
            <span style="font-size: 11px; color: #444; white-space: nowrap;">70% - 85%</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 14px; height: 14px; background: #C300FF; border-radius: 3px; border: 1px solid rgba(0,0,0,0.1);"></div>
            <span style="font-size: 11px; color: #444; white-space: nowrap;">50% - 70%</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 14px; height: 14px; background: #FF00AA; border-radius: 3px; border: 1px solid rgba(0,0,0,0.1);"></div>
            <span style="font-size: 11px; color: #444; white-space: nowrap;">30% - 50%</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 14px; height: 14px; background: #FF8400; border-radius: 3px; border: 1px solid rgba(0,0,0,0.1);"></div>
            <span style="font-size: 11px; color: #444; white-space: nowrap;">10% - 30%</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 14px; height: 14px; background: #D10000; border-radius: 3px; border: 1px solid rgba(0,0,0,0.1);"></div>
            <span style="font-size: 11px; color: #444; white-space: nowrap;">0% - 10%</span>
        </div>
    </div>
</div>

<div id="cont-cuadrillas" style="position: fixed; bottom: 85px; left: 60px; z-index: 9999; display:none; background:white; border-radius:12px; box-shadow:0 4px 15px rgba(0,0,0,0.2); width:230px; border:1px solid #1abc9c; overflow:hidden;">
    <div onclick="toggleMenuCuadrillas()" style="background:#1abc9c; color:white; padding:10px; font-weight:bold; cursor:pointer; display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:13px;">FILTRAR CUADRILLA</span><span id="flecha-menu">▼</span>
    </div>
    <div id="lista-cuadrillas-scroll" style="max-height:0px; overflow-y:auto; transition: max-height 0.3s ease-out;">
        ''' + html_opciones_cuadrillas + r'''
    </div>
</div>

<div id="panel-detalle-cuadrilla">
    <div style="background:#2c3e50; color:white; padding:20px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="display:flex; align-items:baseline; gap:8px;">
                <span id="det-id" style="background:#1abc9c; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:bold;"></span>
                <h3 id="det-nombre" style="margin:0; font-size:18px;"></h3>
            </div>
            <button onclick="cerrarDetalle()" style="background:none; border:none; color:white; font-size:24px; cursor:pointer;">×</button>
        </div>
        <p id="det-rep" style="margin:8px 0 0 0; opacity:0.9; font-size:14px; font-weight:500;"></p>
    </div>
    <div style="background:#1abc9c; color:white; padding:15px; text-align:center;">
        <div style="font-size:10px; text-transform:uppercase; letter-spacing:1px;">Total Acumulado Ganado</div>
        <div id="det-monto" style="font-size:24px; font-weight:bold;"></div>
    </div>
    <div id="det-lista" style="padding:20px; overflow-y:auto; flex-grow:1;"></div>
</div>

<script>
var vista_actual = 'fisico';
var cuadrilla_abierta = null;
var nombre_cuadrilla_filtro = 'TODAS';
var datosC = ''' + json.dumps(info_cuadrillas_js) + r''';

function toggleDetalleCuadrilla(n) {
    if (cuadrilla_abierta === n) { cerrarDetalle(); return; }
    let d = datosC[n];
    document.getElementById('det-id').innerText = "ID: " + d.id;
    document.getElementById('det-nombre').innerText = n;
    document.getElementById('det-rep').innerText = "👤 Representante: " + d.representante;
    document.getElementById('det-monto').innerText = new Intl.NumberFormat('es-CL', {style:'currency', currency:'CLP'}).format(d.total_pagado);

    let lista = document.getElementById('det-lista');
    lista.innerHTML = '<h4 style="font-size:12px; color:#999; border-bottom:1px solid #eee; padding-bottom:5px; margin-top:0;">CASAS Y DESGLOSE</h4>';

    if(d.tratos_realizados.length === 0) {
        lista.innerHTML += '<p style="font-size:11px; color:#ccc;">No hay registros.</p>';
    } else {
        d.tratos_realizados.sort((a, b) => {
            if (a.mzn_sort < b.mzn_sort) return -1;
            if (a.mzn_sort > b.mzn_sort) return 1;
            return a.num_sort - b.num_sort;
        });

        d.tratos_realizados.forEach((t, index) => {
            let idAcc = "acc-" + index;
            let partidasHtml = t.partidas.map(p => `
                <div style="display:flex; justify-content:space-between; font-size:10px; padding:4px 0; color:#7f8c8d; border-bottom:1px solid #f2f2f2;">
                    <span>${p.nombre}</span>
                    <span style="font-weight:bold;">$${new Intl.NumberFormat('es-CL').format(p.precio)}</span>
                </div>
            `).join('');

            lista.innerHTML += `
                <div style="border-bottom:1px solid #eee; margin-bottom:5px;">
                    <div class="casa-header" onclick="document.getElementById('${idAcc}').style.display = document.getElementById('${idAcc}').style.display === 'none' ? 'block' : 'none'"
                         style="display:flex; justify-content:space-between; padding:10px 0; cursor:pointer; font-size:12px; align-items:center;">
                        <span style="color:#34495e; font-weight:600;">🏠 ${t.casa}</span>
                        <div style="text-align:right;">
                            <span style="font-weight:bold; color:#27ae60;">$${new Intl.NumberFormat('es-CL').format(t.monto_total)}</span>
                            <span style="font-size:10px; color:#1abc9c; margin-left:5px;">▼</span>
                        </div>
                    </div>
                    <div id="${idAcc}" style="display:none; background:#f9f9f9; padding:5px 12px 10px 12px; border-radius:4px; margin-bottom:8px;">
                        ${partidasHtml}
                    </div>
                </div>`;
        });
    }
    document.getElementById('panel-detalle-cuadrilla').classList.add('active');
    cuadrilla_abierta = n;
    filtrarC(n);
}

function cerrarDetalle() {
    document.getElementById('panel-detalle-cuadrilla').classList.remove('active');
    cuadrilla_abierta = null;
}

function toggleMenuCuadrillas() {
    let l = document.getElementById('lista-cuadrillas-scroll');
    let f = document.getElementById('flecha-menu');
    if (l.style.maxHeight === '0px' || l.style.maxHeight === '') {
        l.style.maxHeight = '350px'; f.innerText = '▲';
    } else {
        l.style.maxHeight = '0px'; f.innerText = '▼';
    }
}

function filtrarC(nombre) {
    nombre_cuadrilla_filtro = nombre.toUpperCase().trim();
    var map = null;
    for (var i in window) { if (i.startsWith('map_')) map = window[i]; }

    // --- NUEVO: Lógica de resaltado en Popups ---
    var estiloPrevio = document.getElementById('estilo-resaltado-cuadrilla');
    if (estiloPrevio) estiloPrevio.remove();

    if (nombre !== 'TODAS') {
        var style = document.createElement('style');
        style.id = 'estilo-resaltado-cuadrilla';
        // Esto hará que en el popup, las filas que no son de la cuadrilla se vean tenues
        // y la fila de la cuadrilla seleccionada brille en amarillo con un borde lateral.
        style.innerHTML = `
            .fila-trato { opacity: 0.3; transition: all 0.3s; }
            .fila-trato[data-cuadrilla="${nombre}"] {
                opacity: 1 !important;
                background-color: #fff3cd !important;
                border-left: 5px solid #f1c40f !important;
                font-weight: bold !important;
                transform: scale(1.02);
            }`;
        document.head.appendChild(style);
    }
    // -------------------------------------------

    if (map) {
        map.eachLayer(function(layer) {
            if (layer.feature && layer.feature.properties.cuadrillas_list) {
                aplicarEstiloCapa(layer);

                layer.off('mouseover mouseout');
                layer.on('mouseover', function(e) {
                   if (nombre_cuadrilla_filtro !== 'TODAS' && layer.feature.properties.cuadrillas_list.includes(nombre_cuadrilla_filtro)) {
                       this.setStyle({ weight: 6, color: 'white' });
                   } else {
                       this.setStyle({ weight: 3, color: 'white' });
                   }
                   this.bringToFront();
                });
                layer.on('mouseout', function(e) {
                   aplicarEstiloCapa(this);
                });
            }
        });
    }
}

function aplicarEstiloCapa(layer) {
    var props = layer.feature.properties;
    var lista = props.cuadrillas_list || [];

    if (nombre_cuadrilla_filtro === 'TODAS') {
        layer.setStyle({ fillOpacity: 0.7, weight: 1.2, color: 'black', fillColor: props.color_base });
    } else if (lista.includes(nombre_cuadrilla_filtro)) {
        layer.setStyle({ fillOpacity: 0.9, weight: 4, color: '#f1c40f', fillColor: props.color_base });
        layer.bringToFront();
    } else {
        layer.setStyle({ fillOpacity: 0.05, weight: 1, color: '#ddd', fillColor: '#cccccc' });
    }
}

function toggleVista() {
    let inputs = document.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]');
    inputs.forEach(i => i.click());
    if(vista_actual === 'fisico') {
        vista_actual = 'tratos';
        document.getElementById('texto-toggle').innerText = 'Cambiar a Avance Físico';
        document.getElementById('tarjeta-fisico').style.opacity = '0';
        document.getElementById('tarjeta-tratos').style.opacity = '1';
        document.getElementById('tarjeta-tratos').style.pointerEvents = 'auto';
        document.getElementById('cont-cuadrillas').style.display = 'block';
        if(document.getElementById('leyenda-fisico')) document.getElementById('leyenda-fisico').style.display = 'none';
    } else {
        vista_actual = 'fisico';
        document.getElementById('texto-toggle').innerText = 'Cambiar Pestaña a Tratos';
        document.getElementById('tarjeta-fisico').style.opacity = '1';
        document.getElementById('tarjeta-tratos').style.opacity = '0';
        document.getElementById('cont-cuadrillas').style.display = 'none';
        if(document.getElementById('leyenda-fisico')) document.getElementById('leyenda-fisico').style.display = 'block';
        cerrarDetalle();
        filtrarC('TODAS');
    }
}
</script>
{% endmacro %}
'''

macro = MacroElement()
macro._template = Template(overlay_html)
m.get_root().add_child(macro)
m.fit_bounds(limites)

# FINALMENTE, GUARDAR
print("Guardando mapa_generado.html...")
m.save("mapa_generado.html")
print("¡Proceso completado!")
