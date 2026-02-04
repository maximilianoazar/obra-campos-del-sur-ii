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
from branca.element import Template, MacroElement

# ==========================================
# 1. CONFIGURACIÓN Y AUTENTICACIÓN (NUEVO)
# ==========================================

# Intenta obtener credenciales desde Variable de Entorno (GitHub Actions)
# O usa un archivo local 'credentials.json'
try:
    if "GCP_CREDENTIALS" in os.environ:
        print("🔑 Cargando credenciales desde Variable de Entorno...")
        creds_dict = json.loads(os.environ["GCP_CREDENTIALS"])
        gc = gspread.service_account_from_dict(creds_dict)
    else:
        print("🔑 Cargando credenciales desde archivo local 'credentials.json'...")
        gc = gspread.service_account(filename="credentials.json")
except Exception as e:
    print("❌ Error de autenticación. Asegúrate de tener el JSON de la cuenta de servicio.")
    print(f"Detalle: {e}")
    exit()

# Cargar imagen (Asume que está en la raíz del repo)
archivo_plano = 'plano.png'
if not os.path.exists(archivo_plano):
    print(f"❌ No se encontró {archivo_plano}. Sube el archivo al repositorio.")
    exit()

img = cv2.imread(archivo_plano)
h, w, _ = img.shape
print(f"✅ Imagen cargada: {w}x{h} píxeles")

# ==========================================
# 2. PROCESAMIENTO DE IMAGEN (OPENCV)
# ==========================================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

casas_geometria = []
centroides = [] # Lista de diccionarios con info pixel
centroides_casas_raw = [] # Solo tuplas (cx, cy)

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
                centroides_casas_raw.append((cx, cy))

                # Lógica de conversión a Folium
                coords = []
                for pt in approx:
                    px_x = float(pt[0][0])
                    px_y = float(pt[0][1])
                    coords.append(pixel_to_folium((px_x, px_y), h))
                coords.append(coords[0]) # Cerrar polígono
                casas_geometria.append(coords)
                
                # Guardar centroide para lógica de manzanas (usando coord pixel original)
                # OJO: Tu lógica original invertía cy para manzanas, aquí lo mantenemos coherente
                # a como estaba en tu código: cy_pixel = h - cy_mapa, 
                # Pero cx, cy de momentos ya son coordenadas de imagen (arriba-izquierda 0,0)
                # Tu código original hacía una conversión extraña, aquí usamos la directa del momento
                
                centroides.append({
                    "idx": index_geo,
                    "cx": cx,
                    "cy": cy  # Coordenada Y de imagen (0 arriba)
                })
                index_geo += 1

print(f"🏠 Viviendas detectadas: {len(casas_geometria)}")

# ==========================================
# 3. LÓGICA DE MANZANAS (ESTÁTICA)
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
mapa_manzanas = {} # index -> Letra

for casa in centroides:
    x, y = casa["cx"], casa["cy"]
    asignada = False
    for letra, regla in MANZANAS.items():
        if regla(x, y):
            casas_por_manzana[letra].append(casa)
            mapa_manzanas[casa["idx"]] = letra
            asignada = True
            break
    if not asignada:
        casas_por_manzana["SIN_MANZANA"].append(casa)
        mapa_manzanas[casa["idx"]] = "SIN"

# ==========================================
# 4. ORDENAMIENTO DE CASAS (NUMERACIÓN)
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
    min_x = min(c['cx'] for c in casas); max_x = max(c['cx'] for c in casas)
    min_y = min(c['cy'] for c in casas); max_y = max(c['cy'] for c in casas)
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

print("✅ Lógica de Manzanas y Numeración completada.")

# ==========================================
# 5. CARGA DE DATOS GOOGLE SHEETS
# ==========================================

# A. Cargar Maestro de Partidas
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
    print(f"✅ Maestro cargado: {len(lista_maestra_llaves)} partidas.")
except Exception as e:
    print(f"⚠️ Error cargando 'Partidas': {e}. Se usará lista vacía.")
    lista_maestra_llaves = set()

# B. Cargar Observaciones
try:
    sh_obs = gc.open('Pre F1')
    dict_observaciones = {}
    print("🔍 Escaneando Observaciones...")
    for hoja in sh_obs.worksheets():
        nombre_hoja = hoja.title.strip().upper()
        if "MZ" in nombre_hoja:
            letra_mz = nombre_hoja.replace("MZ", "").strip()
            filas = hoja.get_all_values()
            if len(filas) < 2: continue
            for fila in filas[1:]:
                if len(fila) < 3: continue
                try: num_casa = int(float(str(fila[0]).strip()))
                except: continue
                partida = str(fila[1]).strip()
                estado = str(fila[2]).strip()
                comentario = str(fila[3]).strip() if len(fila) > 3 else "Sin detalle"
                
                if estado.lower() == "en proceso":
                    key_partida = (letra_mz, num_casa, partida)
                    dict_observaciones[key_partida] = comentario
    print(f"✅ Total observaciones cargadas: {len(dict_observaciones)}")
except Exception as e:
    print(f"⚠️ Error leyendo observaciones 'Pre F1': {e}")
    dict_observaciones = {}

# C. Cargar Avances Principales
try:
    spreadsheet_name = '135-CR-CAMPOS DEL SUR 2 (VIVIENDAS_SEDE SOCIAL.1)(1)'
    sh = gc.open(spreadsheet_name)
    dict_detalles_casas = {}
    dict_avances = {} # Bruto

    print("🔍 Procesando Avances por Manzana...")
    for worksheet in sh.worksheets():
        sheet_name = worksheet.title
        if "MANZ" in sheet_name.upper():
            datos = worksheet.get_all_values()
            if not datos: continue
            letra_mz = sheet_name.replace("MANZ.", "").replace("MANZ", "").strip().upper()
            
            fila_item_idx = next((i for i, f in enumerate(datos[:50]) if f and str(f[0]).strip().upper() == "ITEM"), None)
            if fila_item_idx is None: continue

            columnas_casas = []
            # Buscar columnas de casas en el rango del encabezado
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
                    if "." not in item_val: titulo_act = desc_val; sub_act = ""
                    else: sub_act = desc_val
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

except Exception as e:
    print(f"❌ Error fatal leyendo Spreadsheet principal: {e}")
    exit()

# ==========================================
# 6. CLASIFICACIÓN Y FILTRADO
# ==========================================

tipos_ref = {
    "Tipo B": ["D1", "F1", "F8", "I1", "I8", "K1", "K8", "L1", "L7"],
    "Tipo C": ["B3", "B4", "B5", "B6", "B7", "G1"],
    "Tipo D": ["B1", "B2"],
    "Tipo A2": ["E16", "E17"],
    "Tipo A1-N": ["E1", "D11", "F11", "I12", "J21"]
}

dict_tipos_vivienda = {}
for i, geo in enumerate(casas_geometria):
    mz_val = str(mapa_manzanas.get(i, "SIN")).strip()
    num_val = str(mapa_numeros.get(i, 0)).strip()
    id_busqueda = f"{mz_val}{num_val}".replace("MANZ.", "").replace(" ", "")
    
    v_tipo = "Tipo A1"
    for t_nombre, lista in tipos_ref.items():
        if id_busqueda in lista: v_tipo = t_nombre; break
    dict_tipos_vivienda[(mz_val, int(num_val))] = v_tipo

# Recálculo de Avances con Filtros de Tipo
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

def extraer_codigo_partida(partida_raw):
    if not partida_raw: return None
    match = re.search(r'\[([A-Z0-9\.]+)\]', partida_raw)
    return match.group(1) if match else None

def partida_aplica(partida_raw, tipo_vivienda, manzana, casa):
    codigo = extraer_codigo_partida(partida_raw)
    if not codigo or codigo not in REGLAS_PARTIDAS: return True
    return tipo_vivienda in REGLAS_PARTIDAS[codigo].get("tipos", set())

dict_avances_filtrado = {}

for key, lista_partidas in dict_detalles_casas.items():
    mz, casa = key
    tipo_v = dict_tipos_vivienda.get(key, "Tipo A1")
    filtradas = []
    
    for p in lista_partidas:
        if partida_aplica(p.get("partida"), tipo_v, mz, casa):
            filtradas.append(p)
            
    if filtradas:
        total_p = len(filtradas)
        hechas = sum(1 for p in filtradas if p['estado'] == "✅")
        dict_avances_filtrado[key] = round((hechas / total_p) * 100, 1)

# ==========================================
# 7. GENERACIÓN DEL MAPA (HTML)
# ==========================================

def generar_html_popup(manzana, casa_num, detalles, tipo_vivienda, avance):
    # Filtrar
    detalles_visibles = [d for d in detalles if partida_aplica(d.get("partida"), tipo_vivienda, manzana, casa_num)]
    
    # Inyectar Observaciones
    letra_mz_limpia = manzana.replace("MZ", "").strip().upper()
    for d in detalles_visibles:
        nombre_excel = d.get('partida', '').strip().upper()
        for (mz_obs, casa_obs, partida_obs) in dict_observaciones.keys():
            if mz_obs == letra_mz_limpia and int(casa_obs) == int(casa_num):
                if partida_obs.upper() in nombre_excel:
                    d['tiene_obs'] = True
                    d['comentario'] = dict_observaciones[(mz_obs, casa_obs, partida_obs)]
                    break

    # Resumen
    resumen = {}
    for d in detalles_visibles:
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

    # HTML String Construction (Simplified for brevity but functional)
    html = f"""
    <div style="font-family: 'Segoe UI', Arial; width: 520px; background: white;">
        <div style="background: #2c3e50; color: white; padding: 10px;">
            <b>MZ {manzana} - Casa {casa_num} ({tipo_vivienda})</b> - {avance}%
        </div>
        <div style="height: 300px; overflow-y: auto; padding: 10px;">
            <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
    """
    
    current_tit = None
    for item in detalles_visibles:
        if item['titulo'] != current_tit:
            current_tit = item['titulo']
            html += f'<tr style="background:#eee;"><td colspan="2"><b>{current_tit}</b></td></tr>'
            
        icon = "⚠️" if item.get('tiene_obs') else item['estado']
        color = "#d4a017" if item.get('tiene_obs') else ("green" if item['estado']=="✅" else "red")
        
        texto_partida = item['partida']
        if item.get('tiene_obs'):
            texto_partida += f"<br><i style='color:#b8860b'>{item['comentario']}</i>"
            
        html += f"<tr><td style='border-bottom:1px solid #ddd;'>{texto_partida}</td><td style='color:{color}; text-align:center;'>{icon}</td></tr>"

    html += "</table></div></div>"
    return html

m = folium.Map(location=[h/2, w/2], zoom_start=0, crs='Simple', tiles=None)
folium.raster_layers.ImageOverlay(image=archivo_plano, bounds=[[0, 0], [h, w]], zindex=1).add_to(m)

for i, geo in enumerate(casas_geometria):
    mz = str(mapa_manzanas.get(i, "SIN"))
    try: num = int(float(mapa_numeros.get(i, 0)))
    except: num = 0
    key = (mz, num)
    
    avance_val = dict_avances_filtrado.get(key, 0)
    detalles = dict_detalles_casas.get(key, [])
    tipo_v = dict_tipos_vivienda.get(key, "Tipo A1")

    # Determinar si tiene observación para el color
    tiene_obs_mapa = False
    letra_mz_limpia = mz.replace("MZ", "").strip().upper()
    
    # Revisión rápida de obs
    for d in detalles:
         nombre_excel = d.get('partida', '').strip().upper()
         for (mz_o, c_o, p_o) in dict_observaciones.keys():
             if mz_o == letra_mz_limpia and int(c_o) == num and p_o.upper() in nombre_excel:
                 tiene_obs_mapa = True; break
         if tiene_obs_mapa: break

    # Color
    color_casa = "#d65548" # Rojo
    if tiene_obs_mapa: color_casa = "#f2ca27" # Amarillo Obs
    elif avance_val > 80: color_casa = "#36d278" # Verde
    elif avance_val >= 30: color_casa = "#409ad5" # Azul

    popup_html = generar_html_popup(mz, num, detalles, tipo_v, avance_val)
    
    folium.GeoJson(
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [geo]}, "properties": {"mz": mz, "num": num}},
        style_function=lambda x, c=color_casa: {"fillColor": c, "color": "black", "weight": 1, "fillOpacity": 0.5},
        highlight_function=lambda x: {"weight": 3, "fillOpacity": 0.8},
        tooltip=f"MZ {mz} - #{num}: {avance_val}%"
    ).add_child(folium.Popup(popup_html, max_width=520)).add_to(m)

# Calcular avance total
avance_total_obra = 0
if dict_avances_filtrado:
    avance_total_obra = round(sum(dict_avances_filtrado.values()) / len(dict_avances_filtrado), 1)

# Overlay simple
macro = MacroElement()
macro._template = Template(f"""
{{% macro html(this, kwargs) %}}
<div style="position:fixed; top:10px; right:10px; background:white; padding:10px; border:2px solid black; z-index:9999;">
    <b>Avance Total: {avance_total_obra}%</b>
</div>
{{% endmacro %}}
""")
m.get_root().add_child(macro)
m.fit_bounds([[0, 0], [h, w]])

output_file = "index.html"
m.save(output_file)
print(f"✅ Mapa generado exitosamente: {output_file}")
