"""
Traductor de PDF  (Inglés → Español)  — V2
✔ Imágenes preservadas en su posición original
✔ Ecuaciones LaTeX/matemáticas no se traducen
✔ Layout original respetado

Requisitos — instala una sola vez:
    pip install deep-translator PyMuPDF

Luego ejecuta:
    python traducir_pdf.py
"""

# ── Verificar e instalar dependencias ────────────────────────────────────────
import importlib
import importlib.util
import sys, subprocess

DEPENDENCIAS = {
    "deep_translator": "deep-translator",
    "fitz":            "PyMuPDF",
}

faltantes = [pip for mod, pip in DEPENDENCIAS.items() if not importlib.util.find_spec(mod)]
if faltantes:
    print("Instalando dependencias faltantes:", faltantes)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + faltantes)
    print("Listo.\n")

# ── Imports ───────────────────────────────────────────────────────────────────
import fitz                          # PyMuPDF
import re, os, time, threading
from deep_translator import GoogleTranslator
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ── Detección de ecuaciones ───────────────────────────────────────────────────

_EC_PATRONES = [
    r'\$[^$]+\$',
    r'\\\[[\s\S]+?\\\]',
    r'\\\([\s\S]+?\\\)',
    r'\\(?:frac|sqrt|sum|int|prod|lim|infty|alpha|beta|gamma|delta|theta|lambda'
    r'|mu|sigma|pi|omega|nabla|partial|cdot|times|div|leq|geq|neq|approx|equiv'
    r'|in|subset|cup|cap|forall|exists)\b',
    r'\b(?:dx|dy|dz|dt)\b',
    r'[∑∏∫∂∇√±∞≤≥≠≈∈⊂∪∩∀∃]',
]
_EC_RE = re.compile('|'.join(_EC_PATRONES), re.DOTALL)

def es_ecuacion(texto):
    return bool(_EC_RE.search(texto))

def es_texto_valido(texto):
    t = texto.strip()
    if len(t) < 3:
        return False
    if re.match(r'^[\d\s\.\-\,\|\:\/\\()\[\]{}=+*<>_^~`\'"#@%&!?;]+$', t):
        return False
    return True

# ── Traducción con reintentos ─────────────────────────────────────────────────

def traducir_texto(texto, traductor, reintentos=3):
    if not es_texto_valido(texto) or es_ecuacion(texto):
        return texto
    MAX = 4500
    if len(texto) > MAX:
        partes = [texto[i:i+MAX] for i in range(0, len(texto), MAX)]
        return " ".join(traducir_texto(p, traductor) for p in partes)
    for intento in range(reintentos):
        try:
            r = traductor.translate(texto)
            return r if r else texto
        except Exception:
            if intento < reintentos - 1:
                time.sleep(1.5)
            else:
                return texto
    return texto

# ── Helpers ───────────────────────────────────────────────────────────────────

def _detectar_fontsize(pagina, x0, y0, x1, y1):
    """Tamaño de fuente promedio del bloque."""
    try:
        zona = fitz.Rect(x0, y0, x1, y1)
        spans_data = pagina.get_text("dict", clip=zona)
        tamanios = []
        for blk in spans_data.get("blocks", []):
            for linea in blk.get("lines", []):
                for span in linea.get("spans", []):
                    tamanios.append(span.get("size", 11))
        if tamanios:
            return round(sum(tamanios) / len(tamanios), 1)
    except Exception:
        pass
    return 11.0

def _color_fondo(pagina, rect):
    """Detecta el color de fondo de la zona (para borrar el texto original)."""
    try:
        clip = fitz.Rect(rect.x0, rect.y0, rect.x0 + 2, rect.y0 + 2)
        pix = pagina.get_pixmap(clip=clip, matrix=fitz.Matrix(1, 1))
        r, g, b = pix.pixel(0, 0)[:3]
        return (r/255, g/255, b/255)
    except Exception:
        return (1, 1, 1)

def _insertar_texto(pagina, rect, texto, fontsize):
    """Inserta texto traducido en el rectángulo, reduciendo fuente si no cabe."""
    fs = fontsize
    MIN_FS = 6.0
    while fs >= MIN_FS:
        resultado = pagina.insert_textbox(
            rect, texto,
            fontsize=fs,
            fontname="helv",
            color=(0, 0, 0),
            align=fitz.TEXT_ALIGN_LEFT,
            overlay=True,
        )
        if resultado >= 0:
            break
        fs -= 0.5
    if fs < MIN_FS:
        pagina.insert_textbox(
            rect, texto, fontsize=MIN_FS,
            fontname="helv", color=(0, 0, 0),
            align=fitz.TEXT_ALIGN_LEFT, overlay=True,
        )

# ── Núcleo: traducir PDF preservando imágenes y ecuaciones ───────────────────

def traducir_pdf(ruta_entrada, ruta_salida, cb_progreso=None, cb_log=None):
    """
    Estrategia:
      1. Abrir el PDF original — las imágenes viajan con él intactas.
      2. Por cada página, iterar bloques de texto solamente (tipo 0).
      3. Bloques de imagen (tipo 1) → no se tocan en absoluto.
      4. Bloques de texto con ecuaciones → no se tocan.
      5. Bloques de texto normal → borrar con rect del color de fondo
         y reinsertar el texto traducido en la misma posición.
      6. Guardar. Las imágenes siguen embebidas en el PDF porque nunca
         se eliminaron — solo se editó la capa de texto.
    """
    if cb_log: cb_log("Abriendo PDF…")

    # Abrir en modo lectura/escritura — las imágenes embebidas permanecen intactas
    doc = fitz.open(ruta_entrada)
    traductor = GoogleTranslator(source='en', target='es')
    total = doc.page_count

    for n in range(total):
        pagina = doc.load_page(n)
        if cb_log:      cb_log(f"Traduciendo página {n+1} / {total}…")
        if cb_progreso: cb_progreso(n+1, total)

        # get_text("blocks") devuelve:
        #   (x0, y0, x1, y1, texto, block_no, block_type)
        #   block_type = 0 → texto
        #   block_type = 1 → imagen   ← NO tocamos estos
        bloques = pagina.get_text("blocks", sort=True)

        for bloque in bloques:
            block_type = bloque[6]

            # ── IMAGEN: saltar completamente, queda donde está ──
            if block_type == 1:
                continue

            # ── TEXTO ──
            x0, y0, x1, y1 = bloque[0], bloque[1], bloque[2], bloque[3]
            texto_original  = bloque[4].strip()

            if not texto_original:
                continue

            # Ecuación → no traducir, dejar intacta
            if es_ecuacion(texto_original):
                continue

            texto_traducido = traducir_texto(texto_original, traductor)

            # Si no cambió (error o texto intraducible) → dejar como está
            if texto_traducido == texto_original:
                continue

            rect = fitz.Rect(x0, y0, x1, y1)
            fontsize = _detectar_fontsize(pagina, x0, y0, x1, y1)
            color_bg = _color_fondo(pagina, rect)

            # Borrar texto original
            pagina.draw_rect(rect, color=color_bg, fill=color_bg, overlay=True)

            # Escribir texto traducido
            _insertar_texto(pagina, rect, texto_traducido, fontsize)

    if cb_log: cb_log("Guardando PDF…")

    # garbage=4 limpia objetos huérfanos pero NO elimina imágenes referenciadas
    # deflate=True comprime para reducir tamaño
    doc.save(ruta_salida, garbage=4, deflate=True)
    doc.close()

    if cb_log:      cb_log(f"✅  Guardado: {os.path.basename(ruta_salida)}")
    if cb_progreso: cb_progreso(total, total)


# ── Interfaz gráfica ──────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Traductor PDF  EN → ES  |  V2")
        self.resizable(False, False)
        self.ruta_pdf = None
        self._build_ui()
        self._centrar()

    def _centrar(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth()  - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"+{x}+{y}")

    def _build_ui(self):
        BG   = "#1a1a2e"
        CARD = "#16213e"
        ACC  = "#e94560"
        TXT  = "#eaeaea"
        MUTE = "#8892a4"

        self.configure(bg=BG)

        tk.Label(self, text="Traductor PDF", font=("Georgia", 17, "bold"),
                 bg=BG, fg=TXT).pack(padx=36, pady=(20, 2))
        tk.Label(self,
                 text="inglés → español  ·  imágenes y ecuaciones preservadas",
                 font=("Helvetica", 9), bg=BG, fg=MUTE).pack()
        tk.Frame(self, height=2, bg=ACC).pack(fill="x", padx=18, pady=10)

        # Selector de archivo
        frame = tk.Frame(self, bg=CARD)
        frame.pack(fill="x", padx=18, pady=(0, 16))
        self.lbl_archivo = tk.Label(
            frame, text="  Ningún archivo seleccionado",
            font=("Consolas", 10), bg=CARD, fg=MUTE,
            anchor="w", width=46, height=2)
        self.lbl_archivo.pack(side="left", padx=8, pady=8)
        tk.Button(
            frame, text="Abrir PDF",
            font=("Helvetica", 11, "bold"),
            bg=ACC, fg="white", activebackground="#c73652",
            activeforeground="white", relief="flat", cursor="hand2",
            padx=12, command=self._seleccionar
        ).pack(side="right", padx=8, pady=8)

        # Barra de progreso
        sty = ttk.Style(self)
        sty.theme_use("default")
        sty.configure("P.Horizontal.TProgressbar",
                      troughcolor=CARD, background=ACC, thickness=10)
        self.progreso = ttk.Progressbar(
            self, style="P.Horizontal.TProgressbar",
            orient="horizontal", length=440, mode="determinate")
        self.progreso.pack(padx=18, pady=(0, 4))

        # Log de estado
        self.lbl_log = tk.Label(self, text="",
                                font=("Consolas", 10),
                                bg=BG, fg=MUTE, anchor="w")
        self.lbl_log.pack(fill="x", padx=18, pady=(0, 8))

        # Botón traducir
        self.btn = tk.Button(
            self, text="▶  Traducir",
            font=("Helvetica", 11, "bold"),
            bg="#0f3460", fg=TXT,
            activebackground="#1a5276", activeforeground=TXT,
            relief="flat", cursor="hand2",
            padx=20, pady=10, state="disabled",
            command=self._iniciar)
        self.btn.pack(pady=(0, 20))

    def _seleccionar(self):
        ruta = filedialog.askopenfilename(
            title="Selecciona el PDF en inglés",
            filetypes=[("PDF", "*.pdf")])
        if ruta:
            self.ruta_pdf = ruta
            self.lbl_archivo.config(
                text=f"  {os.path.basename(ruta)}", fg="#eaeaea")
            self.btn.config(state="normal")
            self.progreso["value"] = 0
            self.lbl_log.config(text="")

    def _iniciar(self):
        self.btn.config(state="disabled", text="Traduciendo…")
        threading.Thread(target=self._hilo, daemon=True).start()

    def _hilo(self):
        try:
            base, ext = os.path.splitext(self.ruta_pdf)
            salida = f"{base}_traducido{ext}"
            traducir_pdf(
                self.ruta_pdf, salida,
                cb_progreso=self._prog,
                cb_log=self._log)
            self.after(0, lambda: messagebox.showinfo(
                "¡Listo!", f"Archivo guardado en:\n{salida}"))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.after(0, lambda: self.btn.config(
                state="normal", text="▶  Traducir"))

    def _prog(self, actual, total):
        self.after(0, lambda: self.progreso.config(
            value=int(actual / total * 100)))

    def _log(self, msg):
        self.after(0, lambda: self.lbl_log.config(text=f"  {msg}"))


if __name__ == "__main__":
    App().mainloop()