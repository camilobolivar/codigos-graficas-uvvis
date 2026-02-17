from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# --- 3) Separar blanco y medidas ---
blancos = [a for a in archivos if "BLANCO" in a.name.upper()]
medidas = [a for a in archivos if a not in blancos]

if len(blancos) == 0:
    raise ValueError("No encontré archivo BLANCO en los nombres.")
blanco_path = blancos[0]

print("BLANCO:", blanco_path.name)
print("Medidas:", len(medidas))

# --- 4) Leer blanco crudo ---
df_blanco = leer_uvvis_dta(blanco_path)
x_b = df_blanco["Wavelength"].to_numpy()
y_b = df_blanco["Absorbance"].to_numpy()

# --- 5) Restar blanco crudo a cada medida cruda ---
curvas = []

for p in medidas:
    df_m = leer_uvvis_dta(p)
    x_m = df_m["Wavelength"].to_numpy()
    y_m = df_m["Absorbance"].to_numpy()

    # Interpolar blanco a la malla de esta medida
    y_b_interp = np.interp(x_m, x_b, y_b)

    # Corregida (sin suavizar)
    y_corr = y_m - y_b_interp

    curvas.append({
        "nombre": p.stem,
        "x": x_m,
        "y": y_corr,
        "orden": y_corr[-1]  # para ordenar la leyenda según altura (valor final)
    })

# --- 6) Ordenar curvas según cómo quedan arriba/abajo (opcional) ---
curvas = sorted(curvas, key=lambda d: d["orden"], reverse=True)

# --- 7) Graficar ---
plt.figure(figsize=(11,7))
for c in curvas:
    plt.plot(c["x"], c["y"], linewidth=1.8, label=c["nombre"])

plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance (medida - blanco) [crudo]")
plt.title("Medidas corregidas restando BLANCO (sin suavizado)")
plt.grid(True)

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
