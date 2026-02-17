from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

CARPETA = Path(r"C:\Users\camil\Desktop\Proyecto de grado\Resutados UV VIS\Ensayo eucalipto")  # <-- cambia esto
SALIDA  = CARPETA / "procesado"
SALIDA.mkdir(exist_ok=True)

CARPETA, SALIDA




def leer_uvvis_dta(ruta: Path) -> pd.DataFrame:
    """
    Lee archivos .DTA tipo UV-Vis (texto con metadata y tabla WaveLength/Absorbance).
    Retorna DataFrame con columnas: Wavelength (float), Absorbance (float)
    """
    with open(ruta, "r", encoding="latin-1", errors="replace") as f:
        lineas = f.readlines()

    # buscar "nm" (la tabla empieza después)
    inicio = None
    for i, linea in enumerate(lineas):
        if linea.strip() == "nm":
            inicio = i + 1
            break
    if inicio is None:
        raise ValueError(f"No encontré el marcador 'nm' en {ruta.name}")

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




def despike_mad_interpol(y: np.ndarray, window=25, k=3.5):
    """Marca outliers con vecinos (MAD) y los reemplaza por interpolación."""
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


def limpiar_espectro(df: pd.DataFrame,
                     despike_window=25, despike_k=3.5,
                     sg_window=31, sg_poly=3) -> pd.DataFrame:
    """
    Devuelve df con columnas:
    Wavelength, Absorbance_raw, Absorbance_clean, outlier
    """
    x = df["Wavelength"].to_numpy()
    y = df["Absorbance"].to_numpy()

    y_despike, out_mask = despike_mad_interpol(y, window=despike_window, k=despike_k)

    # sg_window debe ser impar y <= len(y)
    sg_window = min(sg_window, len(y) if len(y)%2==1 else len(y)-1)
    if sg_window < 5:
        sg_window = 5 if len(y) >= 5 else len(y)

    y_smooth = savgol_filter(y_despike, window_length=sg_window, polyorder=sg_poly)

    out = pd.DataFrame({
        "Wavelength": x,
        "Absorbance_raw": y,
        "Absorbance_clean": y_smooth,
        "outlier": out_mask
    })
    return out






archivos = sorted(list(CARPETA.glob("*.DTA")) + list(CARPETA.glob("*.dta")))
print("Archivos encontrados:", len(archivos))
for a in archivos[:10]:
    print(a.name)

# Identificar blanco(s) por nombre
blancos = [a for a in archivos if "BLANCO" in a.name.upper()]
medidas = [a for a in archivos if a not in blancos]

print("\nBlancos:", [b.name for b in blancos])
print("Medidas:", len(medidas))








if len(blancos) == 0:
    raise ValueError("No encontré ningún archivo con 'BLANCO' en el nombre. Renómbralo o ajusto el filtro.")

blanco_path = blancos[0]
df_blanco = leer_uvvis_dta(blanco_path)
blanco_clean = limpiar_espectro(df_blanco)

blanco_path.name, blanco_clean.head()










def alinear_y_restar(med_clean: pd.DataFrame, blanco_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Alinea por Wavelength (merge exacto). Si alguna longitud no coincide, se interpola el blanco.
    """
    x_m = med_clean["Wavelength"].to_numpy()
    y_m = med_clean["Absorbance_clean"].to_numpy()

    x_b = blanco_clean["Wavelength"].to_numpy()
    y_b = blanco_clean["Absorbance_clean"].to_numpy()

    # Interpolamos blanco a la malla del espectro de la medida
    y_b_on_m = np.interp(x_m, x_b, y_b)

    corr = y_m - y_b_on_m

    out = pd.DataFrame({
        "Wavelength": x_m,
        "Absorbance_clean": y_m,
        "Blank_clean_interp": y_b_on_m,
        "Absorbance_corrected": corr
    })
    return out


todos = []
for p in medidas:
    df_med = leer_uvvis_dta(p)
    med_clean = limpiar_espectro(df_med)

    corr = alinear_y_restar(med_clean, blanco_clean)
    corr["source_file"] = p.name
    corr["blank_file"] = blanco_path.name

    # Guardar por archivo
    out_name = p.stem + "_corr.csv"
    corr.to_csv(SALIDA / out_name, index=False, encoding="utf-8")
    todos.append(corr)

# Guardar “base grande”
df_todos = pd.concat(todos, ignore_index=True)
df_todos.to_csv(SALIDA / "todas_corr.csv", index=False, encoding="utf-8")

print("✅ Guardado en:", SALIDA)
print("Archivos corregidos:", len(medidas))
df_todos.head()






import matplotlib.pyplot as plt

# elige una medida cualquiera
ejemplo = df_todos["source_file"].iloc[0]
sub = df_todos[df_todos["source_file"] == ejemplo]

plt.figure(figsize=(10,6))
plt.plot(sub["Wavelength"], sub["Absorbance_corrected"])
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance (medida - blanco)")
plt.title(f"Corregida: {ejemplo}  (blanco: {blanco_path.name})")
plt.grid()
plt.show()
