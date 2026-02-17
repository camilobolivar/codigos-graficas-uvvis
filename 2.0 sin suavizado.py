from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- 1) Carpeta y archivos ---
CARPETA = Path(r"C:\Users\camil\Desktop\Proyecto de grado\Resutados UV VIS\Ensayo eucalipto")  # <-- cambia si hace falta
archivos = sorted(CARPETA.glob("*.dta"))

# --- 2) Lector UV-Vis .DTA (texto) ---
def leer_uvvis_dta(ruta):
    with open(ruta, "r", encoding="latin-1", errors="replace") as f:
        lineas = f.readlines()

    inicio = None
    for i, linea in enumerate(lineas):
        if linea.strip() == "nm":
            inicio = i + 1
            break
    if inicio is None:
        raise ValueError(f"No encontré 'nm' en {ruta.name}")

    datos = []
    for linea in lineas[inicio:]:
        if not linea.strip():
            continue
        partes = linea.split()
        if len(partes) >= 2:
            datos.append(partes[:2])

    df = pd.DataFrame(datos, columns=["Wavelength", "Absorbance"])
    df["Wavelength"] = df["Wavelength"].str.replace(",", ".").astype(float)
    df["Absorbance"] = df["Absorbance"].str.replace(",", ".").astype(float)
    df = df.sort_values("Wavelength").reset_index(drop=True)
    return df

# --- 3) Despike (vecinos + MAD) y Savitzky ---
def despike_mad_interpol(y, window=25, k=3.5):
    if window % 2 == 0:
        raise ValueError("window debe ser impar")
    r = window // 2
    y = y.astype(float)
    mask_out = np.zeros_like(y, dtype=bool)

    def mad(a):
        med = np.median(a)
        return np.median(np.abs(a - med))

    for i in range(len(y)):
        i0 = max(0, i - r)
        i1 = min(len(y), i + r + 1)
        neigh = y[i0:i1]

        med = np.median(neigh)
        s = mad(neigh)
        sigma = 1.4826 * s if s > 0 else 0.0

        if sigma > 0 and abs(y[i] - med) > k * sigma:
            mask_out[i] = True

    idx = np.arange(len(y))
    good = ~mask_out
    y_clean = y.copy()
    if good.sum() >= 2:
        y_clean[mask_out] = np.interp(idx[mask_out], idx[good], y[good])

    return y_clean, mask_out

def suavizar_resultado(y, despike_window=25, despike_k=3.5, sg_window=31, sg_poly=3):
    y1, _ = despike_mad_interpol(y, window=despike_window, k=despike_k)

    sg_window = min(sg_window, len(y1) if len(y1) % 2 == 1 else len(y1) - 1)
    if sg_window < 5:
        sg_window = 5 if len(y1) >= 5 else len(y1)

    y2 = savgol_filter(y1, window_length=sg_window, polyorder=sg_poly)
    return y2

# --- 4) Separar blanco y medidas ---
blancos = [a for a in archivos if "BLANCO" in a.name.upper()]
medidas = [a for a in archivos if a not in blancos]

if len(blancos) == 0:
    raise ValueError("No encontré archivo BLANCO en los nombres.")
blanco_path = blancos[0]

print("BLANCO:", blanco_path.name)
print("Medidas:", len(medidas))

# --- 5) Leer blanco crudo ---
df_blanco = leer_uvvis_dta(blanco_path)
x_b = df_blanco["Wavelength"].to_numpy()
y_b = df_blanco["Absorbance"].to_numpy()

# --- 6) Restar blanco y luego suavizar el resultado ---
curvas = []

for p in medidas:
    df_m = leer_uvvis_dta(p)
    x_m = df_m["Wavelength"].to_numpy()
    y_m = df_m["Absorbance"].to_numpy()

    # blanco interpolado a la malla de la medida
    y_b_interp = np.interp(x_m, x_b, y_b)

    # resultado crudo corregido
    y_corr_raw = y_m - y_b_interp

    # ✅ suavizar DESPUÉS de la resta
    y_corr_smooth = suavizar_resultado(y_corr_raw, despike_window=25, despike_k=3.5, sg_window=31, sg_poly=3)

    curvas.append({
        "nombre": p.stem,
        "x": x_m,
        "y": y_corr_smooth,
        "orden": y_corr_smooth[-1]  # para ordenar la leyenda según altura
    })

# ordenar por altura final (opcional)
curvas = sorted(curvas, key=lambda d: d["orden"], reverse=True)

# --- 7) Graficar ---
plt.figure(figsize=(11,7))
for c in curvas:
    plt.plot(c["x"], c["y"], linewidth=2, label=c["nombre"])

plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance corregida (suavizada)")
plt.title("Medidas (medida - blanco) y luego suavizado (vecinos+MAD+Savitzky)")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()



