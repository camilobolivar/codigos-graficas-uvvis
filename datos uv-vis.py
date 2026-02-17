
from pathlib import Path

carpeta = Path(r"C:\Users\camil\Desktop\Proyecto de grado\Resutados UV VIS\Ensayo eucalipto")
print("Existe carpeta?", carpeta.exists())

# Lista los .DTA/.dta
archivos = sorted(list(carpeta.glob("*.DTA")) + list(carpeta.glob("*.dta")))
print("Cantidad:", len(archivos))
for a in archivos[:20]:
    print(a.name)




from pathlib import Path

ruta = Path(r"C:\Users\camil\Desktop\Proyecto de grado\Resutados UV VIS\Ensayo eucalipto\12 02 2026 BLANCO.DTA")

with open(ruta, "r", encoding="latin-1") as f:
    for i in range(30):
        print(f.readline())




from pathlib import Path
import pandas as pd

ruta = Path(r"C:\Users\camil\Desktop\Proyecto de grado\Resutados UV VIS\Ensayo eucalipto\12 02 2026 BLANCO.DTA")

# Leer todo como texto
with open(ruta, "r", encoding="latin-1") as f:
    lineas = f.readlines()

# Encontrar donde empieza la tabla (después de "nm")
inicio = None
for i, linea in enumerate(lineas):
    if linea.strip() == "nm":
        inicio = i + 1
        break

# Extraer solo las líneas de datos
datos = []
for linea in lineas[inicio:]:
    if linea.strip() == "":
        continue
    partes = linea.split()
    if len(partes) >= 2:
        datos.append(partes[:2])

# Crear DataFrame
df = pd.DataFrame(datos, columns=["Wavelength", "Absorbance"])

# Convertir coma decimal a punto y pasar a float
df["Wavelength"] = df["Wavelength"].str.replace(",", ".").astype(float)
df["Absorbance"] = df["Absorbance"].str.replace(",", ".").astype(float)

print(df.head())
print(df.shape)


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(df["Wavelength"], df["Absorbance"])
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.title("UV-Vis Spectrum")
plt.grid()
plt.show()





from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Aplicar filtro
abs_suavizada = savgol_filter(df["Absorbance"], window_length=21, polyorder=3)

# Graficar comparación
plt.figure(figsize=(8,5))
plt.plot(df["Wavelength"], df["Absorbance"], alpha=0.4, label="Original")
plt.plot(df["Wavelength"], abs_suavizada, linewidth=2, label="Suavizado (Savitzky-Golay)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.legend()
plt.grid()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

x = df["Wavelength"].to_numpy()
y = df["Absorbance"].to_numpy()

# s controla suavizado: más grande = más suave
spl = UnivariateSpline(x, y, s=5.0)
y_spline = spl(x)

plt.figure(figsize=(8,5))
plt.plot(x, y, alpha=0.35, label="Original")
plt.plot(x, y_spline, linewidth=2, label="Spline suavizado")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.legend(); plt.grid(); plt.show()



import numpy as np
import matplotlib.pyplot as plt

x = df["Wavelength"].to_numpy()
y = df["Absorbance"].to_numpy()

def despike_knn(y, window=11, k=4.0, replace="median"):
    """
    window: tamaño de la ventana de vecinos (impar).
    k: umbral en sigmas robustas (MAD). Más pequeño = más agresivo.
    replace: "median" o "interp"
    """
    if window % 2 == 0:
        raise ValueError("window debe ser impar (ej: 9, 11, 15, 21).")
    r = window // 2
    y = y.astype(float).copy()
    y_clean = y.copy()
    mask_out = np.zeros_like(y, dtype=bool)

    # helper MAD robusta
    def mad(a):
        med = np.median(a)
        return np.median(np.abs(a - med))

    for i in range(len(y)):
        i0 = max(0, i - r)
        i1 = min(len(y), i + r + 1)

        neigh = y[i0:i1]
        med = np.median(neigh)
        s = mad(neigh)

        # escala robusta ~ sigma (1.4826*MAD)
        sigma = 1.4826 * s if s > 0 else 0.0

        if sigma > 0 and np.abs(y[i] - med) > k * sigma:
            mask_out[i] = True
            if replace == "median":
                y_clean[i] = med

    if replace == "interp":
        idx = np.arange(len(y))
        good = ~mask_out
        # si hay suficientes puntos buenos, interpolamos
        if good.sum() >= 2:
            y_clean[mask_out] = np.interp(idx[mask_out], idx[good], y[good])

    return y_clean, mask_out

# --- AJUSTES ---
y_clean, out_mask = despike_knn(y, window=21, k=4.0, replace="interp")

print("Puntos marcados como erróneos:", int(out_mask.sum()))

plt.figure(figsize=(9,5))
plt.plot(x, y, alpha=0.35, label="Original")
plt.plot(x, y_clean, linewidth=2, label="Corregido (vecinos + MAD)")
plt.scatter(x[out_mask], y[out_mask], s=12, label="Outliers", alpha=0.8)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.legend()
plt.grid()
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

x = df["Wavelength"].to_numpy()
y = df["Absorbance"].to_numpy()

# -------------------------
# 1) ELIMINAR OUTLIERS (vecinos + MAD)
# -------------------------

def despike_knn(y, window=21, k=4.0):
    if window % 2 == 0:
        raise ValueError("window debe ser impar")
    r = window // 2
    y = y.astype(float)
    y_clean = y.copy()
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

    # Interpolar puntos erróneos
    idx = np.arange(len(y))
    good = ~mask_out
    y_clean[mask_out] = np.interp(idx[mask_out], idx[good], y[good])

    return y_clean, mask_out


y_despike, out_mask = despike_knn(y, window=25, k=3.5)

# -------------------------
# 2) SUAVIZADO SAVITZKY-GOLAY
# -------------------------

y_smooth = savgol_filter(y_despike, window_length=31, polyorder=3)

# -------------------------
# 3) GRAFICAR
# -------------------------

plt.figure(figsize=(10,6))

plt.plot(x, y, alpha=0.3, label="Original")
plt.plot(x, y_despike, alpha=0.6, label="Sin outliers")
plt.plot(x, y_smooth, linewidth=2.5, label="Final (Suavizado)")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.legend()
plt.grid()
plt.show()

print("Puntos corregidos:", out_mask.sum())
