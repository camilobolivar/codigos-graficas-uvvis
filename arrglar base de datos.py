from pathlib import Path

CARPETA = Path(r"C:\Users\camil\Desktop\Proyecto de grado\Resutados UV VIS\Ensayo eucalipto")  # <-- cambia si hace falta

#archivos = sorted(list(CARPETA.glob("*.DTA")) + list(CARPETA.glob("*.dta")))

# En Windows, esto ya captura .DTA y .dta sin duplicar
# archivos = sorted(CARPETA.glob("*.dta"))

archivos = sorted({p.resolve() for p in CARPETA.glob("*.dta")})
archivos = sorted(archivos, key=lambda p: p.name.lower())

print("Carpeta:", CARPETA)
print("Existe carpeta?:", CARPETA.exists())
print("Cantidad de archivos .DTA:", len(archivos))

for a in archivos[:20]:
    print("-", a.name)






# Separar blanco(s) y medidas
blancos = [a for a in archivos if "BLANCO" in a.name.upper()]
medidas = [a for a in archivos if a not in blancos]

print("Cantidad de blancos:", len(blancos))
for b in blancos:
    print("BLANCO ->", b.name)

print("\nCantidad de medidas:", len(medidas))
for m in medidas[:10]:
    print("MEDIDA ->", m.name)





import pandas as pd

def leer_uvvis_dta(ruta):
    with open(ruta, "r", encoding="latin-1", errors="replace") as f:
        lineas = f.readlines()

    # Buscar donde empieza la tabla (después de "nm")
    inicio = None
    for i, linea in enumerate(lineas):
        if linea.strip() == "nm":
            inicio = i + 1
            break

    if inicio is None:
        raise ValueError("No encontré el marcador 'nm' en el archivo.")

    datos = []
    for linea in lineas[inicio:]:
        if not linea.strip():
            continue
        partes = linea.split()
        if len(partes) >= 2:
            datos.append(partes[:2])

    df = pd.DataFrame(datos, columns=["Wavelength", "Absorbance"])

    # Convertir coma decimal a punto
    df["Wavelength"] = df["Wavelength"].str.replace(",", ".").astype(float)
    df["Absorbance"] = df["Absorbance"].str.replace(",", ".").astype(float)

    return df


# Probar con el BLANCO
df_blanco = leer_uvvis_dta(blancos[0])

print(df_blanco.head())
print("Shape:", df_blanco.shape)




import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(df_blanco["Wavelength"], df_blanco["Absorbance"])
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.title(f"BLANCO (crudo): {blancos[0].name}")
plt.grid(True)
plt.show()




import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

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


def limpiar_y(y, despike_window=25, despike_k=3.5, sg_window=31, sg_poly=3):
    y1, out_mask = despike_mad_interpol(y, window=despike_window, k=despike_k)

    # asegurar sg_window impar y <= len(y)
    sg_window = min(sg_window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if sg_window < 5:
        sg_window = 5 if len(y) >= 5 else len(y)

    y2 = savgol_filter(y1, window_length=sg_window, polyorder=sg_poly)
    return y2, out_mask


# Aplicar al blanco
x_b = df_blanco["Wavelength"].to_numpy()
y_b = df_blanco["Absorbance"].to_numpy()

y_blanco_limpio, mask_out_b = limpiar_y(y_b, despike_window=25, despike_k=3.5, sg_window=31, sg_poly=3)

print("Outliers corregidos (blanco):", int(mask_out_b.sum()))

plt.figure(figsize=(10,6))
plt.plot(x_b, y_b, alpha=0.3, label="Blanco crudo")
plt.plot(x_b, y_blanco_limpio, linewidth=2.5, label="Blanco limpio")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.title("BLANCO: crudo vs limpio")
plt.grid(True)
plt.legend()
plt.show()





import matplotlib.pyplot as plt
import numpy as np

# 1) Elegir la primera medida
m0 = medidas[0]
df_m0 = leer_uvvis_dta(m0)

# 2) Limpiar la medida
x_m = df_m0["Wavelength"].to_numpy()
y_m = df_m0["Absorbance"].to_numpy()
y_med_limpia, mask_out_m = limpiar_y(y_m, despike_window=25, despike_k=3.5, sg_window=31, sg_poly=3)

print("Medida:", m0.name)
print("Outliers corregidos (medida):", int(mask_out_m.sum()))

# 3) Alinear el blanco a la malla de la medida (por si no coinciden exactamente)
y_blanco_interp = np.interp(x_m, x_b, y_blanco_limpio)

# 4) Restar blanco
y_corr = y_med_limpia - y_blanco_interp

# 5) Graficar
plt.figure(figsize=(10,6))
plt.plot(x_m, y_m, alpha=0.25, label="Medida cruda")
plt.plot(x_m, y_med_limpia, alpha=0.8, label="Medida limpia")
plt.plot(x_m, y_corr, linewidth=2.5, label="Medida corregida (medida - blanco)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.title(f"Corrección con blanco: {m0.name}")
plt.grid(True)
plt.legend()
plt.show()

# 5) Graficar SOLO la corregida
plt.figure(figsize=(10,6))
plt.plot(x_m, y_corr, linewidth=2.5)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance (corregida)")
plt.title(f"Espectro corregido: {m0.name}")
plt.grid(True)
plt.show()


import pandas as pd
from pathlib import Path
import numpy as np

# Carpeta de salida
SALIDA = CARPETA / "procesado"
SALIDA.mkdir(exist_ok=True)

todos = []

for p in medidas:
    # 1) leer medida
    df_med = leer_uvvis_dta(p)
    x = df_med["Wavelength"].to_numpy()
    y = df_med["Absorbance"].to_numpy()

    # 2) limpiar medida
    y_med_limpia, _ = limpiar_y(
        y,
        despike_window=25,
        despike_k=3.5,
        sg_window=31,
        sg_poly=3
    )

    # 3) interpolar blanco limpio a la malla de esta medida
    y_b_interp = np.interp(x, x_b, y_blanco_limpio)

    # 4) restar
    y_corr = y_med_limpia - y_b_interp

    # 5) dataframe final por archivo
    df_out = pd.DataFrame({
        "Wavelength": x,
        "Absorbance_corrected": y_corr
    })
    df_out["source_file"] = p.name
    df_out["blank_file"] = blancos[0].name

    # guardar por archivo
    out_name = p.stem + "_CORREGIDA.csv"
    df_out.to_csv(SALIDA / out_name, index=False, encoding="utf-8")

    todos.append(df_out)

# 6) guardar base total
df_todos = pd.concat(todos, ignore_index=True)
df_todos.to_csv(SALIDA / "TODAS_CORREGIDAS.csv", index=False, encoding="utf-8")

print("✅ Guardado listo")
print("Carpeta salida:", SALIDA)
print("Archivos corregidos:", len(medidas))
print("Base total filas:", df_todos.shape[0])







import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SALIDA = CARPETA / "procesado"
SALIDA.mkdir(exist_ok=True)

todos = []

for p in medidas:
    df_med = leer_uvvis_dta(p)
    x = df_med["Wavelength"].to_numpy()
    y = df_med["Absorbance"].to_numpy()

    y_med_limpia, _ = limpiar_y(y, despike_window=25, despike_k=3.5, sg_window=31, sg_poly=3)
    y_b_interp = np.interp(x, x_b, y_blanco_limpio)
    y_corr = y_med_limpia - y_b_interp

    df_out = pd.DataFrame({
        "Wavelength": x,
        "Absorbance_corrected": y_corr
    })
    df_out["source_file"] = p.name
    df_out["blank_file"] = blancos[0].name

    out_name = p.stem + "_CORREGIDA.csv"
    df_out.to_csv(SALIDA / out_name, index=False, encoding="utf-8")

    todos.append(df_out)

    # ✅ Graficar ESTA medida corregida
    plt.figure(figsize=(10,5))
    plt.plot(x, y_corr, linewidth=2.2)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance (corregida)")
    plt.title(f"Corregida: {p.name}")
    plt.grid(True)
    plt.show()

df_todos = pd.concat(todos, ignore_index=True)
df_todos.to_csv(SALIDA / "TODAS_CORREGIDAS.csv", index=False, encoding="utf-8")

print("✅ Guardado listo")
print("Carpeta salida:", SALIDA)
print("Archivos corregidos:", len(medidas))
print("Base total filas:", df_todos.shape[0])







import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SALIDA = CARPETA / "procesado"
SALIDA.mkdir(exist_ok=True)

todos = []

plt.figure(figsize=(10,6))

for p in medidas:
    df_med = leer_uvvis_dta(p)
    x = df_med["Wavelength"].to_numpy()
    y = df_med["Absorbance"].to_numpy()

    y_med_limpia, _ = limpiar_y(y, despike_window=25, despike_k=3.5, sg_window=31, sg_poly=3)
    y_b_interp = np.interp(x, x_b, y_blanco_limpio)
    y_corr = y_med_limpia - y_b_interp

    df_out = pd.DataFrame({
        "Wavelength": x,
        "Absorbance_corrected": y_corr
    })
    df_out["source_file"] = p.name
    df_out["blank_file"] = blancos[0].name

    out_name = p.stem + "_CORREGIDA.csv"
    df_out.to_csv(SALIDA / out_name, index=False, encoding="utf-8")

    todos.append(df_out)

    # ✅ añade la curva a la figura general
    plt.plot(x, y_corr, alpha=0.35)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance (corregida)")
plt.title("Todas las medidas corregidas (medida - blanco)")
plt.grid(True)
plt.show()

df_todos = pd.concat(todos, ignore_index=True)
df_todos.to_csv(SALIDA / "TODAS_CORREGIDAS.csv", index=False, encoding="utf-8")

print("✅ Guardado listo")
print("Carpeta salida:", SALIDA)
print("Archivos corregidos:", len(medidas))
print("Base total filas:", df_todos.shape[0])
















curvas = []

for i, p in enumerate(medidas):
    df_med = leer_uvvis_dta(p)
    x = df_med["Wavelength"].to_numpy()
    y = df_med["Absorbance"].to_numpy()

    y_med_limpia, _ = limpiar_y(y, 25, 3.5, 31, 3)
    y_b_interp = np.interp(x, x_b, y_blanco_limpio)
    y_corr = y_med_limpia - y_b_interp

    curvas.append({
        "nombre": p.stem,
        "x": x,
        "y": y_corr,
        "orden": y_corr[-1]  # valor final para ordenar
    })

# 🔥 Ordenar según valor final (de mayor a menor)
curvas = sorted(curvas, key=lambda d: d["orden"], reverse=True)

plt.figure(figsize=(11,7))

for curva in curvas:
    plt.plot(curva["x"], curva["y"], linewidth=2, label=curva["nombre"])

plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance (corregida)")
plt.title("Todas las medidas corregidas")
plt.grid(True)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.tight_layout()
plt.show()






































import matplotlib.pyplot as plt

# Escribe aquí una parte del nombre para buscar (ej: "PRIMERA", "SEGUNDA", "DECIMA")
BUSCAR = "blanco"

coinciden = sorted(df_todos["source_file"].unique())
coinciden = [n for n in coinciden if BUSCAR.upper() in n.upper()]

print("Coincidencias:", len(coinciden))
for n in coinciden[:20]:
    print("-", n)

# Tomamos la primera coincidencia
archivo = coinciden[0]
sub = df_todos[df_todos["source_file"] == archivo]

plt.figure(figsize=(10,6))
plt.plot(sub["Wavelength"], sub["Absorbance_corrected"], linewidth=2.5)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance (corregida)")
plt.title(f"Corregida: {archivo}")
plt.grid(True)
plt.show()




import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Ejemplo: cambia el nombre exacto del CSV si quieres
csv_path = SALIDA / (medidas[0].stem + "_CORREGIDA.csv")

df_plot = pd.read_csv(csv_path)

plt.figure(figsize=(10,6))
plt.plot(df_plot["Wavelength"], df_plot["Absorbance_corrected"], linewidth=2.5)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance (corregida)")
plt.title(f"Desde CSV: {csv_path.name}")
plt.grid(True)
plt.show()
