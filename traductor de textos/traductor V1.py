"""
Traductor de PDF  (Inglés → Español)
Corre directamente en tu PC con interfaz gráfica.

Requisitos — instala una sola vez en tu terminal:
    pip install deep-translator PyMuPDF reportlab

Luego ejecuta:
    python traducir_pdf.py
"""

# ── Verificar dependencias antes de arrancar ──────────────────────────────────
import importlib, sys, subprocess
import importlib
import importlib.util
import sys, subprocess

DEPENDENCIAS = {
    "deep_translator": "deep-translator",
    "fitz":            "PyMuPDF",
    "reportlab":       "reportlab",
}

faltantes = [pip for mod, pip in DEPENDENCIAS.items() if not importlib.util.find_spec(mod)]
if faltantes:
    print("Instalando dependencias faltantes:", faltantes)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + faltantes)
    print("Listo. Reinicia el script si hay errores de importación.\n")

# ── Imports ───────────────────────────────────────────────────────────────────
import fitz
import re, os, time, threading
from deep_translator import GoogleTranslator
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ── Lógica de traducción ──────────────────────────────────────────────────────

def es_ecuacion(texto):
    patrones = [
        r'\$.*?\$',
        r'\\\[.*?\\\]',
        r'\\\(.*?\\\)',
        r'\\[a-zA-Z]+\{',
        r'^\s*[\d\W]{1,5}\s*$',
    ]
    return any(re.search(p, texto, re.DOTALL) for p in patrones)


def es_texto_valido(texto):
    texto = texto.strip()
    if len(texto) < 3:
        return False
    if re.match(r'^[\d\s\.\-\,\|\:\/\\]+$', texto):
        return False
    return True


def traducir_texto(texto, traductor, reintentos=3):
    if not es_texto_valido(texto) or es_ecuacion(texto):
        return texto
    MAX_CHARS = 4500
    if len(texto) > MAX_CHARS:
        partes = [texto[i:i + MAX_CHARS] for i in range(0, len(texto), MAX_CHARS)]
        return " ".join(traducir_texto(p, traductor) for p in partes)
    for intento in range(reintentos):
        try:
            resultado = traductor.translate(texto)
            return resultado if resultado else texto
        except Exception:
            if intento < reintentos - 1:
                time.sleep(1.5)
            else:
                return texto
    return texto


def traducir_pdf(ruta_entrada, ruta_salida, cb_progreso=None, cb_log=None):
    doc = fitz.open(ruta_entrada)
    traductor = GoogleTranslator(source='en', target='es')
    total = doc.page_count
    paginas = []

    for n in range(total):
        pagina = doc.load_page(n)
        bloques = pagina.get_text("blocks", sort=True)
        if cb_log:      cb_log(f"Traduciendo página {n + 1} / {total}…")
        if cb_progreso: cb_progreso(n + 1, total)

        textos = []
        for bloque in bloques:
            if bloque[6] == 1: continue          # imagen → saltar
            texto = bloque[4].strip()
            if not texto: continue
            textos.append(traducir_texto(texto, traductor))
        paginas.append(textos)

    doc.close()
    if cb_log: cb_log("Generando PDF…")

    # ── ReportLab ──
    estilos = getSampleStyleSheet()
    e_cuerpo = ParagraphStyle('Cuerpo', parent=estilos['Normal'],
                               fontSize=10, leading=14, spaceAfter=6,
                               fontName='Helvetica')
    e_titulo = ParagraphStyle('Titulo', parent=estilos['Heading2'],
                               fontSize=12, leading=16, spaceAfter=8,
                               fontName='Helvetica-Bold')

    doc_out = SimpleDocTemplate(ruta_salida, pagesize=letter,
                                rightMargin=0.75*inch, leftMargin=0.75*inch,
                                topMargin=0.9*inch, bottomMargin=0.9*inch)
    historia = []

    for n, textos in enumerate(paginas):
        if n > 0:
            historia.append(Spacer(1, 14))
            historia.append(Paragraph(
                f"<font color='grey'>─── Página {n+1} ───</font>",
                estilos['Normal']))
            historia.append(Spacer(1, 6))
        for texto in textos:
            estilo = e_titulo if (len(texto) < 80 and not texto.endswith('.')) else e_cuerpo
            seguro = texto.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
            try:
                historia.append(Paragraph(seguro, estilo))
            except Exception:
                limpio = re.sub(r'[^\x20-\x7E\u00C0-\u024F]', '', seguro)
                historia.append(Paragraph(limpio, e_cuerpo))

    doc_out.build(historia)
    if cb_log:      cb_log(f"✅  Guardado: {os.path.basename(ruta_salida)}")
    if cb_progreso: cb_progreso(total, total)


# ── Interfaz gráfica ──────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Traductor PDF  EN → ES")
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
        FONT_T = ("Georgia", 17, "bold")
        FONT_B = ("Consolas", 10)
        FONT_N = ("Helvetica", 11, "bold")

        self.configure(bg=BG)

        # Título
        tk.Label(self, text="Traductor PDF", font=FONT_T,
                 bg=BG, fg=TXT).pack(padx=36, pady=(20, 2))
        tk.Label(self, text="inglés  →  español", font=("Helvetica", 10),
                 bg=BG, fg=MUTE).pack()
        tk.Frame(self, height=2, bg=ACC).pack(fill="x", padx=18, pady=10)

        # Selector de archivo
        frame = tk.Frame(self, bg=CARD)
        frame.pack(fill="x", padx=18, pady=(0, 16))

        self.lbl_archivo = tk.Label(frame, text="  Ningún archivo seleccionado",
                                    font=FONT_B, bg=CARD, fg=MUTE,
                                    anchor="w", width=46, height=2)
        self.lbl_archivo.pack(side="left", padx=8, pady=8)

        tk.Button(frame, text="Abrir PDF", font=FONT_N,
                  bg=ACC, fg="white", activebackground="#c73652",
                  activeforeground="white", relief="flat",
                  cursor="hand2", padx=12,
                  command=self._seleccionar).pack(side="right", padx=8, pady=8)

        # Barra de progreso
        sty = ttk.Style(self)
        sty.theme_use("default")
        sty.configure("P.Horizontal.TProgressbar",
                      troughcolor=CARD, background=ACC, thickness=10)
        self.progreso = ttk.Progressbar(self, style="P.Horizontal.TProgressbar",
                                        orient="horizontal", length=420,
                                        mode="determinate")
        self.progreso.pack(padx=18, pady=(0, 4))

        # Log
        self.lbl_log = tk.Label(self, text="", font=FONT_B,
                                bg=BG, fg=MUTE, anchor="w")
        self.lbl_log.pack(fill="x", padx=18, pady=(0, 8))

        # Botón traducir
        self.btn = tk.Button(self, text="▶  Traducir", font=FONT_N,
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
            self.lbl_archivo.config(text=f"  {os.path.basename(ruta)}", fg="#eaeaea")
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
            traducir_pdf(self.ruta_pdf, salida,
                         cb_progreso=self._prog,
                         cb_log=self._log)
            self.after(0, lambda: messagebox.showinfo(
                "¡Listo!", f"Archivo guardado en:\n{salida}"))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.after(0, lambda: self.btn.config(state="normal", text="▶  Traducir"))

    def _prog(self, actual, total):
        self.after(0, lambda: self.progreso.config(value=int(actual/total*100)))

    def _log(self, msg):
        self.after(0, lambda: self.lbl_log.config(text=f"  {msg}"))


if __name__ == "__main__":
    App().mainloop()
