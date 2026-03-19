"""
emulsion_sim.py
===============
Simulación 2D de emulsión agua-aceite con Tween 80 (surfactante).
Gotas modeladas como esferas 3D proyectadas en un plano.

Correcciones aplicadas respecto a versión original:
  1. Conservación de VOLUMEN (3D): R_new = (R_i³ + R_j³)^(1/3)
  2. Barrera de coalescencia física: P = exp(-E0*(θ_i+θ_j))
     con freno adicional usando min() (no max())
  3. Energía interfacial con área esférica: γ · 4πR²
  4. Visualización robusta: gotas periódicas bien dibujadas,
     fondo de aceite fijo, colormap por cobertura θ,
     tamaño de surfactante proporcional a R, animación mejorada.

Uso:
    python emulsion_sim.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

plt.rcParams["figure.dpi"] = 140
rng = np.random.default_rng(3)

# Carpeta de salida: misma carpeta donde está este script
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
def out(name): return os.path.join(OUT_DIR, name)

# ───────────────────────────────────────────────
# PARÁMETROS
# ───────────────────────────────────────────────
L = 1.0          # dominio adimensional

dt = 2e-3
n_steps = 3000
save_every = 50

D0 = 2e-3        # coef. difusión base (Stokes-Einstein simplificado D~D0/R)

# Goteo
drop_interval = 25
R_in  = 0.018
x_in_mean  = 0.50
x_in_sigma = 0.10
y_in = 0.90

# Coalescencia (barrera de energía por surfactante)
# Con θ≈0.8 y E0=5 → P~exp(-8)≈0.0003: coalescencia casi nula (emulsión demasiado estable).
# E0=1.5 da comportamiento observable: a θ bajo se fusionan, a θ alto se frena.
E0        = 1.5  # barrera en kT
min_merge = 0.50  # freno multiplicativo suave a θ→1

# Adsorción (Langmuir)
c_bulk = 1.0
k_ads  = 2.0
theta0 = 0.0

# Tensión interfacial (modelo lineal con θ)
gamma_ow    = 1.0    # agua-aceite sin surfactante (u.a.)
delta_gamma = 0.75   # reducción máxima a θ=1

max_drops = 500

# ───────────────────────────────────────────────
# FUNCIONES FÍSICAS
# ───────────────────────────────────────────────

def gamma_drop(theta):
    """Tensión interfacial efectiva γ(θ) = γ_ow − Δγ·θ."""
    return gamma_ow - delta_gamma * theta


def interfacial_energy(drops):
    """
    Energía interfacial total (esferas 3D):
        E_int = Σ γ_i · 4πR_i²
    """
    if len(drops) == 0:
        return 0.0
    return sum(gamma_drop(th) * 4.0 * np.pi * R**2
               for _, _, R, th in drops)


def add_drop(drops):
    """Inyecta una gota nueva por la parte superior del dominio."""
    x = rng.normal(x_in_mean, x_in_sigma) % L
    drops.append([float(x), float(y_in), float(R_in), float(theta0)])


def step_brownian(drops):
    """Movimiento browniano: D ~ D0/R (Stokes-Einstein)."""
    for d in drops:
        x, y, R, th = d
        D  = D0 / max(R, 1e-6)
        sq = np.sqrt(2 * D * dt)
        x  = (x + sq * rng.standard_normal()) % L
        y  = (y + sq * rng.standard_normal()) % L
        d[0], d[1] = float(x), float(y)


def step_adsorption(drops):
    """Adsorción tipo Langmuir: dθ/dt = k_ads·c_bulk·(1−θ)."""
    for d in drops:
        th   = d[3] + k_ads * c_bulk * (1.0 - d[3]) * dt
        d[3] = float(np.clip(th, 0.0, 1.0))


def try_merge(drops):
    """
    Fusiona gotas solapadas con probabilidad:
        P_merge = exp(-E0·(θ_i + θ_j))          [barrera Arrhenius]
        P_merge *= max(0, 1 − min_merge·(θ_i+θ_j)/2)  [freno por saturación]

    Conservación de VOLUMEN 3D:
        R_new = (R_i³ + R_j³)^(1/3)
    """
    if len(drops) < 2:
        return

    X  = np.array([d[0] for d in drops], dtype=float)
    Y  = np.array([d[1] for d in drops], dtype=float)
    R  = np.array([d[2] for d in drops], dtype=float)
    TH = np.array([d[3] for d in drops], dtype=float)

    alive = np.ones(len(drops), dtype=bool)

    for i in range(len(drops)):
        if not alive[i]:
            continue
        for j in range(i + 1, len(drops)):
            if not alive[j]:
                continue

            # distancia con imagen mínima (PBC)
            dx = X[i] - X[j];  dx -= np.round(dx / L) * L
            dy = Y[i] - Y[j];  dy -= np.round(dy / L) * L
            dist = np.hypot(dx, dy)

            if dist < (R[i] + R[j]):
                # ── CORRECCIÓN 1: barrera física con min() ──
                P = np.exp(-E0 * (TH[i] + TH[j]))
                # freno adicional cuando θ → 1 (reduce P, nunca la sube)
                P *= max(0.0, 1.0 - min_merge * (TH[i] + TH[j]) / 2.0)
                P  = float(np.clip(P, 0.0, 1.0))

                if rng.random() < P:
                    Ri, Rj = R[i], R[j]

                    # ── CORRECCIÓN 2: conservación de volumen 3D ──
                    Rnew = (Ri**3 + Rj**3) ** (1.0 / 3.0)

                    # posición pesada por volumen (masa ∝ R³)
                    Wi, Wj = Ri**3, Rj**3
                    xj = X[j] - np.round((X[j] - X[i]) / L) * L
                    yj = Y[j] - np.round((Y[j] - Y[i]) / L) * L
                    xnew = ((Wi * X[i] + Wj * xj) / (Wi + Wj)) % L
                    ynew = ((Wi * Y[i] + Wj * yj) / (Wi + Wj)) % L

                    # cobertura θ conservando área superficial
                    thnew = (Wi * TH[i] + Wj * TH[j]) / (Wi + Wj)

                    alive[j] = False
                    X[i], Y[i], R[i], TH[i] = xnew, ynew, Rnew, thnew

    drops[:] = [
        [float(X[i]), float(Y[i]), float(R[i]), float(TH[i])]
        for i in range(len(drops)) if alive[i]
    ]


# ───────────────────────────────────────────────
# LOOP PRINCIPAL
# ───────────────────────────────────────────────

drops  = []
frames = []
stats  = {k: [] for k in ("t", "N", "R_mean", "R_max",
                           "theta_mean", "gamma_eff", "Eint")}

for step in range(n_steps + 1):

    if step % drop_interval == 0 and len(drops) < max_drops:
        add_drop(drops)

    step_brownian(drops)
    step_adsorption(drops)
    try_merge(drops)

    stats["t"].append(step * dt)
    stats["N"].append(len(drops))

    if drops:
        Rarr  = np.array([d[2] for d in drops])
        THarr = np.array([d[3] for d in drops])
        th_m  = float(THarr.mean())
        stats["R_mean"].append(float(Rarr.mean()))
        stats["R_max"].append(float(Rarr.max()))
        stats["theta_mean"].append(th_m)
        stats["gamma_eff"].append(gamma_ow - delta_gamma * th_m)
        stats["Eint"].append(interfacial_energy(drops))
    else:
        stats["R_mean"].append(0.0);  stats["R_max"].append(0.0)
        stats["theta_mean"].append(0.0)
        stats["gamma_eff"].append(float(gamma_ow))
        stats["Eint"].append(0.0)

    if step % save_every == 0:
        frames.append([d.copy() for d in drops])
        print(f"step {step:4d}/{n_steps} | N={len(drops):3d} | "
              f"Rmean={stats['R_mean'][-1]:.4f} | "
              f"θ={stats['theta_mean'][-1]:.3f} | "
              f"Eint={stats['Eint'][-1]:.5f}")

print("\nSimulación terminada.")
print(f"E_int inicial = {stats['Eint'][0]:.5f}")
print(f"E_int final   = {stats['Eint'][-1]:.5f}")
print(f"Reducción     = {stats['Eint'][0] - stats['Eint'][-1]:.5f}")


# ───────────────────────────────────────────────
# GRÁFICAS DE ESTADÍSTICAS
# ───────────────────────────────────────────────

t = np.array(stats["t"])
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

axes[0].plot(t, stats["N"], color="#2563eb")
axes[0].set(xlabel="t (u.a.)", ylabel="Número de gotas",
            title="Conteo de gotas"); axes[0].grid(True, alpha=0.4)

axes[1].plot(t, stats["R_mean"], label=r"$\langle R \rangle$", color="#16a34a")
axes[1].plot(t, stats["R_max"],  label=r"$R_{\max}$",          color="#dc2626", ls="--")
axes[1].set(xlabel="t (u.a.)", ylabel="Radio (u.a.)",
            title="Crecimiento por coalescencia")
axes[1].legend(); axes[1].grid(True, alpha=0.4)

axes[2].plot(t, stats["theta_mean"], label=r"$\langle\theta\rangle$", color="#7c3aed")
axes[2].plot(t, stats["gamma_eff"],  label=r"$\gamma_{\rm eff}$",     color="#ea580c", ls="--")
axes[2].set(xlabel="t (u.a.)", title="Adsorción y tensión efectiva")
axes[2].legend(); axes[2].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(out("stats.png"), dpi=150)
plt.show()

# Energía interfacial
fig2, ax2 = plt.subplots(figsize=(6, 3.5))
ax2.plot(t, stats["Eint"], color="#0891b2")
ax2.set(xlabel="t (u.a.)",
        ylabel=r"$E_{\rm int}=\sum\gamma_i\,4\pi R_i^2$  (u.a.)",
        title="Energía interfacial total")
ax2.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(out("eint.png"), dpi=150)
plt.show()


# ───────────────────────────────────────────────
# VISUALIZACIÓN MEJORADA DE LA ESCENA
# ───────────────────────────────────────────────

# Fondo de aceite FIJO (semilla propia, no afecta rng global)
_bg_rng = np.random.default_rng(42)
_bg_x   = _bg_rng.random(500) * L
_bg_y   = _bg_rng.random(500) * L

cmap = plt.get_cmap("plasma")
norm = Normalize(vmin=0, vmax=1)


def draw_scene(drops, title="Gotas + Tween 80", n_surf=22, ax=None):
    """
    Dibuja las gotas con:
      • Color de relleno según cobertura θ (cmap plasma).
      • Surfactantes (cabeza+cola) proporcionales a θ y R.
      • Condiciones periódicas: gotas que cruzan el borde se dibujan
        también en el lado opuesto.
      • Fondo de aceite reproducible (semilla fija).
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # fondo aceite
    ax.scatter(_bg_x, _bg_y, s=3, color="#d97706", alpha=0.12, zorder=0)

    for x, y, R, th in drops:
        # gotas que tocan el borde → imagen espejo para PBC visual
        images = [(x, y)]
        if x - R < 0:   images.append((x + L, y))
        if x + R > L:   images.append((x - L, y))
        if y - R < 0:   images.append((x, y + L))
        if y + R > L:   images.append((x, y - L))

        color = cmap(norm(th))

        for xi, yi in images:
            # disco de agua (relleno semitransparente)
            circle = plt.Circle((xi, yi), R,
                                 facecolor=(*color[:3], 0.25),
                                 edgecolor=color, linewidth=1.8, zorder=2)
            ax.add_patch(circle)

            # surfactantes en la interfaz
            m = int(n_surf * th)
            if m <= 0:
                continue
            angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
            for ang in angles:
                ux, uy = np.cos(ang), np.sin(ang)
                xs, ys = xi + R * ux, yi + R * uy
                # cabeza hidrofílica (sobre la interfaz)
                ax.scatter([xs], [ys], s=14, color=color,
                            zorder=4, edgecolors="none")
                # cola hidrofóbica (hacia el aceite, ~40% R)
                tail = 0.40 * R
                ax.plot([xs, xs + tail * ux], [ys, ys + tail * uy],
                         color=color, linewidth=1.0, alpha=0.75, zorder=3)

    # colorbar θ
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Cobertura θ", fraction=0.04, pad=0.02)

    ax.set_title(title)
    if standalone:
        plt.tight_layout()
        plt.savefig(out("scene_final.png"), dpi=150)
        plt.show()


# Escena inicial y final en paralelo
fig, (ax_i, ax_f) = plt.subplots(1, 2, figsize=(11, 5.5))
draw_scene(frames[0],  title="t = inicial", ax=ax_i)
draw_scene(frames[-1], title="t = final",   ax=ax_f)
plt.suptitle("Evolución de la emulsión", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(out("scene_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()

print("\nArchivos guardados:")
print("  stats.png  — estadísticas temporales")
print("  eint.png   — energía interfacial")
print("  scene_comparison.png — escena inicial vs final")
