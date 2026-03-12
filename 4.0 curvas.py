from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# =========================================================
# 1) CONFIGURACIÓN GENERAL
# =========================================================
CARPETA = Path(r"C:\Users\camil\Desktop\Proyecto de grado\Resutados UV VIS\Ensayo eucalipto\ensayo25 02 2026")
archivos = sorted(CARPETA.glob("*.dta"))

SALIDA = CARPETA / "procesado_resta_luego_suaviza"
SALIDA.mkdir(exist_ok=True)

print("Carpeta de entrada :", CARPETA)
print("Carpeta de salida  :", SALIDA)
print("Archivos .dta encontrados:", len(archivos))

if len(archivos) == 0:
    raise ValueError("No se encontraron archivos .dta en la carpeta indicada.")

# =========================================================
# 2) LECTOR ROBUSTO PARA ARCHIVOS UV-VIS .DTA
# =========================================================
def leer_uvvis_dta(ruta):
    """
    Lee un archivo .dta buscando la línea 'nm' como inicio de datos.
    Devuelve un DataFrame con columnas:
        - Wavelength
        - Absorbance
    """
    with open(ruta, "r", encoding="latin-1", errors="replace") as f:
        lineas = f.readlines()

    inicio = None
    for i, linea in enumerate(lineas):
        if linea.strip() == "nm":
            inicio = i + 1
            break

    if inicio is None:
        raise ValueError(f"No encontré la línea 'nm' en el archivo: {ruta.name}")

    datos = []
    for linea in lineas[inicio:]:
        if not linea.strip():
            continue

        partes = linea.split()
        if len(partes) >= 2:
            datos.append(partes[:2])

    if len(datos) == 0:
        raise ValueError(f"No se encontraron datos numéricos después de 'nm' en {ruta.name}")

    df = pd.DataFrame(datos, columns=["Wavelength", "Absorbance"])

    # Conversión robusta
    df["Wavelength"] = pd.to_numeric(
        df["Wavelength"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    df["Absorbance"] = pd.to_numeric(
        df["Absorbance"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

    # Eliminar filas no numéricas
    df = df.dropna(subset=["Wavelength", "Absorbance"])

    if df.empty:
        raise ValueError(f"Todos los datos quedaron inválidos al convertir a número en {ruta.name}")

    # Ordenar por longitud de onda
    df = df.sort_values("Wavelength").reset_index(drop=True)

    # Si hay longitudes de onda repetidas, promediar absorbancia
    if df["Wavelength"].duplicated().any():
        print(f"Advertencia: {ruta.name} tiene longitudes de onda repetidas. Se promediarán.")
        df = df.groupby("Wavelength", as_index=False)["Absorbance"].mean()

    return df

# =========================================================
# 3) DESPIKE + SUAVIZADO ROBUSTO
# =========================================================
def despike_mad_interpol(y, window=25, k=3.5):
    """
    Detecta spikes usando mediana local + MAD y los reemplaza por interpolación.
    """
    y = np.asarray(y, dtype=float)

    if len(y) == 0:
        raise ValueError("La señal y está vacía.")
    if window < 3:
        raise ValueError("window debe ser >= 3.")
    if window % 2 == 0:
        raise ValueError("window debe ser impar.")

    r = window // 2
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

    if good.sum() >= 2 and mask_out.any():
        y_clean[mask_out] = np.interp(idx[mask_out], idx[good], y[good])

    return y_clean, mask_out

def suavizar_resultado(y, despike_window=25, despike_k=3.5, sg_window=31, sg_poly=3):
    """
    Aplica despike y luego Savitzky-Golay de forma robusta.
    Si la señal es demasiado corta o no cumple condiciones, devuelve la señal despikeada.
    """
    y1, mask = despike_mad_interpol(y, window=despike_window, k=despike_k)
    n = len(y1)

    # Si la señal es muy corta, no forzamos Savitzky-Golay
    if n < 5:
        return y1.copy(), mask

    # Ajustar ventana Savitzky-Golay para que sea válida
    sg_window = min(sg_window, n)

    # Debe ser impar
    if sg_window % 2 == 0:
        sg_window -= 1

    # Debe ser mayor que polyorder
    if sg_window <= sg_poly:
        candidato = sg_poly + 2
        if candidato % 2 == 0:
            candidato += 1
        sg_window = candidato

    # Si tras ajustar se pasó del tamaño de la señal
    if sg_window > n:
        sg_window = n if n % 2 == 1 else n - 1

    # Validación final
    if sg_window < 3 or sg_window <= sg_poly:
        return y1.copy(), mask

    y2 = savgol_filter(y1, window_length=sg_window, polyorder=sg_poly)
    return y2, mask

# =========================================================
# 4) SEPARAR BLANCOS Y MEDIDAS
# =========================================================
blancos = [a for a in archivos if "BLANCO" in a.name.upper()]
medidas = [a for a in archivos if a not in blancos]

print("\nBlancos encontrados:")
for b in blancos:
    print(" -", b.name)

print("\nCantidad de medidas encontradas:", len(medidas))

if len(blancos) == 0:
    raise ValueError("No encontré ningún archivo BLANCO en los nombres.")
if len(medidas) == 0:
    raise ValueError("No encontré archivos de medida distintos del blanco.")

# Si hay varios blancos, toma el primero pero avisa
if len(blancos) > 1:
    print("\nAdvertencia: se encontraron varios blancos.")
    print("Se usará el primero en orden alfabético:", blancos[0].name)

blanco_path = blancos[0]

# =========================================================
# 5) LEER EL BLANCO
# =========================================================
df_blanco = leer_uvvis_dta(blanco_path)
x_b = df_blanco["Wavelength"].to_numpy()
y_b = df_blanco["Absorbance"].to_numpy()

print("\nBLANCO seleccionado:", blanco_path.name)
print(f"Rango blanco: [{x_b.min():.3f}, {x_b.max():.3f}] nm")
print(f"Número de puntos del blanco: {len(x_b)}")

# =========================================================
# 6) PROCESAR MEDIDAS
# =========================================================
curvas = []

for p in medidas:
    print("\n" + "=" * 70)
    print("Procesando:", p.name)

    df_m = leer_uvvis_dta(p)
    x_m = df_m["Wavelength"].to_numpy()
    y_m = df_m["Absorbance"].to_numpy()

    print(f"Rango medida: [{x_m.min():.3f}, {x_m.max():.3f}] nm")
    print(f"Número de puntos de la medida: {len(x_m)}")

    # Validar si la medida sale del rango del blanco
    if x_m.min() < x_b.min() or x_m.max() > x_b.max():
        print(
            "Advertencia: la medida tiene un rango fuera del blanco. "
            "np.interp usará valores de borde en los extremos."
        )
        print(
            f"  Rango medida : [{x_m.min():.3f}, {x_m.max():.3f}] nm\n"
            f"  Rango blanco : [{x_b.min():.3f}, {x_b.max():.3f}] nm"
        )

    # Interpolar blanco sobre la malla de la medida
    y_b_interp = np.interp(x_m, x_b, y_b)

    # Corrección cruda
    y_corr_raw = y_m - y_b_interp

    # Suavizado posterior
    y_corr_smooth, mask_out = suavizar_resultado(
        y_corr_raw,
        despike_window=25,
        despike_k=3.5,
        sg_window=31,
        sg_poly=3
    )

    n_out = int(mask_out.sum())
    print(f"Puntos corregidos por despike: {n_out}")

    # Guardar archivo procesado
    df_out = pd.DataFrame({
        "Wavelength": x_m,
        "Absorbance_raw": y_m,
        "Absorbance_blank_interp": y_b_interp,
        "Absorbance_corrected_raw": y_corr_raw,
        "Absorbance_corrected_smooth": y_corr_smooth
    })

    out_path = SALIDA / f"{p.stem}_corr_smooth.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print("Archivo guardado:", out_path.name)

    # Guardar información para ordenamiento y graficado
    curvas.append({
        "nombre": p.stem,
        "x": x_m,
        "y_raw": y_corr_raw,
        "y_smooth": y_corr_smooth,
        "orden": np.max(y_corr_smooth),   # criterio de ordenamiento
        "maximo": np.max(y_corr_smooth),
        "ultimo": y_corr_smooth[-1]
    })

# =========================================================
# 7) RESUMEN DE ORDENAMIENTO
# =========================================================
curvas = sorted(curvas, key=lambda d: d["orden"], reverse=True)

print("\n" + "=" * 70)
print("ORDEN FINAL DE CURVAS (por máximo de absorbancia suavizada):")
for i, c in enumerate(curvas, start=1):
    print(
        f"{i:2d}. {c['nombre']} | "
        f"máx = {c['maximo']:.6f} | "
        f"último = {c['ultimo']:.6f}"
    )

# =========================================================
# 8) GRÁFICA GENERAL DE TODAS LAS CURVAS SUAVIZADAS
# =========================================================
plt.figure(figsize=(11, 7))
for c in curvas:
    plt.plot(c["x"], c["y_smooth"], linewidth=2, label=c["nombre"])

plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance corregida (suavizada)")
plt.title("Medidas corregidas (medida - blanco) y luego suavizadas")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# =========================================================
# 9) GRÁFICAS INDIVIDUALES DE CONTROL
# =========================================================
# Esto sirve para verificar si el problema viene de la resta o del suavizado
for c in curvas:
    plt.figure(figsize=(10, 5))
    plt.plot(c["x"], c["y_raw"], label="Corregida cruda", alpha=0.8)
    plt.plot(c["x"], c["y_smooth"], linewidth=2, label="Corregida suavizada")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance corregida")
    plt.title(f"Control de procesamiento: {c['nombre']}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

print("\n✅ Proceso terminado correctamente.")
print("✅ Todos los archivos procesados fueron guardados en:")
print(SALIDA)