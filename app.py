# app.py — S2P Analyzer Pro : Cp / tanδ / ESR / ESL + métriques + rapport PDF
# Auteur : imane — usage académique/indus

import re, math, cmath, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import streamlit as st
from scipy.signal import savgol_filter

st.set_page_config(page_title="S2P Analyzer Pro — Cp / tanδ / ESR / ESL", layout="wide")

# =========================
# Parsing Touchstone .s2p
# =========================
def parse_touchstone_s2p(file_like):
    freq_unit = "Hz"
    data_format = "RI"  # RI, MA, DB
    z0 = 50.0
    freqs, S11_list, S21_list, S12_list, S22_list = [], [], [], [], []

    text = file_like.read().decode("utf-8", errors="ignore")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue

        if line.startswith("#"):
            tokens = line[1:].strip().split()
            tokens_upper = [t.upper() for t in tokens]
            for u in ["HZ", "KHZ", "MHZ", "GHZ"]:
                if u in tokens_upper:
                    freq_unit = u.capitalize()
                    break
            for fmt in ["RI", "MA", "DB"]:
                if fmt in tokens_upper:
                    data_format = fmt
                    break
            if "R" in tokens_upper:
                idx = tokens_upper.index("R")
                if idx + 1 < len(tokens):
                    try: z0 = float(tokens[idx + 1])
                    except: pass
            continue

        parts = re.split(r"\s+", line)
        if len(parts) < 9:
            continue
        try:
            fval = float(parts[0])
        except:
            continue

        mult = {"Hz":1.0, "KHz":1e3, "kHz":1e3, "MHz":1e6, "GHz":1e9}
        f_Hz = fval * mult.get(freq_unit, 1.0)
        vals = list(map(float, parts[1:9]))

        def to_complex(a,b,fmt):
            if fmt == "RI":
                return complex(a,b)
            elif fmt == "MA":
                return a * cmath.exp(1j * math.radians(b))
            elif fmt == "DB":
                mag = 10**(a/20.0)
                return mag * cmath.exp(1j * math.radians(b))
            return complex(a,b)

        S11 = to_complex(vals[0], vals[1], data_format)
        S21 = to_complex(vals[2], vals[3], data_format)
        S12 = to_complex(vals[4], vals[5], data_format)
        S22 = to_complex(vals[6], vals[7], data_format)

        freqs.append(f_Hz)
        S11_list.append(S11); S21_list.append(S21)
        S12_list.append(S12); S22_list.append(S22)

    if not freqs:
        raise ValueError("Aucune donnée valide dans le .s2p")

    return (np.asarray(freqs,float),
            np.asarray(S11_list,complex),
            np.asarray(S21_list,complex),
            np.asarray(S12_list,complex),
            np.asarray(S22_list,complex),
            float(z0), data_format, freq_unit)

# =========================
# Calculs de base
# =========================
def s11_to_zin(S11, Z0):  # Z_in = Z0(1+Γ)/(1-Γ)
    return Z0 * (1 + S11) / (1 - S11)

def compute_params_from_s11(freqs, S11, Z0):
    omega = 2*np.pi*freqs
    Zin = s11_to_zin(S11, Z0)
    Yin = 1.0 / Zin
    G, B = np.real(Yin), np.imag(Yin)
    with np.errstate(divide='ignore', invalid='ignore'):
        Cp = np.where(omega!=0, B/omega, np.nan)
        tanD = np.where(B!=0, G/np.abs(B), np.nan)
    Rs, Xs = np.real(Zin), np.imag(Zin)
    with np.errstate(divide='ignore', invalid='ignore'):
        Cs = np.where(Xs<0, -1.0/(omega*Xs), np.nan)
        Ls = np.where(Xs>0,  Xs/omega, np.nan)  # ESL quand X>0
    return pd.DataFrame({
        "freq_Hz": freqs, "Cp_F": Cp, "tanD": tanD,
        "ESR_Ohm": Rs, "ESL_H": Ls, "Cs_F": Cs, "Ls_H": Ls
    }), Zin, Yin

# =========================
# Outils “prof de carac”
# =========================
def apply_band_and_decimate(df, fmin, fmax, decim):
    d = df[(df["freq_Hz"]>=fmin) & (df["freq_Hz"]<=fmax)].copy()
    if decim > 1:
        d = d.iloc[::decim, :].reset_index(drop=True)
    return d

def apply_savgol(y, win, poly):
    if win is None or win < 3 or poly is None:
        return y
    win = int(win) + (int(win)%2==0)  # fenêtre impaire
    if win < 3 or win > len(y):
        return y
    try:
        return savgol_filter(y, window_length=win, polyorder=int(poly))
    except Exception:
        return y

def estimate_metrics(df, target_freq, low_cap_max, high_L_min):
    f = df["freq_Hz"].values
    Rs = df["ESR_Ohm"].values.astype(float)
    Xs = (2*np.pi*f*df["ESL_H"].fillna(0).values).astype(float)  # juste pour signe au-dessus SRF
    Cp = df["Cp_F"].values.astype(float)
    tanD = df["tanD"].values.astype(float)
    Ls = df["ESL_H"].values.astype(float)

    # SRF = zéro de Xs (imag(Z)=0) ≈ zéro de 1/(ωCp) + ωL -> recherche changement de signe de imag(Z)
    Zin_re = Rs
    Zin_im = np.imag(s11_to_zin((0*f).astype(complex)+0j, 50)) # placeholder non utilisé
    # meilleur: recalculer directement imag(Z) depuis série:
    omega = 2*np.pi*f
    # On recompose Xs depuis Cs/Ls approchés: préfèrons imag(Zin) depuis Rs + j*Xs
    # Ici, on le recalcule proprement via Cp & Ls n’est pas fiable autour de SRF, donc on refait à partir d’origine :
    # (plus simple: considérer signe de Cs/Ls: SRF ~ freq où Cs devient NaN et Ls apparait)
    srf = np.nan
    sign = np.sign(df["Cs_F"].fillna(0).values*(-1))  # Cs existe => capacitif
    trans = np.where(np.diff(sign) != 0)[0]
    if trans.size > 0:
        srf = f[trans[0]+1]

    # Cp basse fréquence (médiane sous low_cap_max)
    mask_lowC = f <= low_cap_max
    Cp_lf = np.nanmedian(Cp[mask_lowC]) if np.any(mask_lowC) else np.nan

    # ESL haute fréquence (médiane au-dessus high_L_min)
    mask_highL = f >= max(high_L_min, srf if np.isfinite(srf) else 0)
    ESL_hf = np.nanmedian(Ls[mask_highL]) if np.any(mask_highL) else np.nan

    # ESR et tanD à une fréquence cible
    def interp_at(x, y, x0):
        if np.all(~np.isfinite(y)): return np.nan
        try: return float(np.interp(x0, x[np.isfinite(y)], y[np.isfinite(y)]))
        except: return np.nan

    ESR_f0 = interp_at(f, Rs, target_freq)
    tanD_f0 = interp_at(f, tanD, target_freq)
    Q_f0 = 1.0/tanD_f0 if np.isfinite(tanD_f0) and tanD_f0>0 else np.nan

    return {
        "SRF_Hz": srf,
        "Cp_lowF_F": Cp_lf,
        "ESL_highF_H": ESL_hf,
        "ESR_at_f0_Ohm": ESR_f0,
        "tanD_at_f0": tanD_f0,
        "Q_at_f0": Q_f0
    }

def figure_loglog(x, y, xlabel, ylabel, title, mask_pos=True):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y) & (x>0)
    if mask_pos: mask &= (y>0)
    fig, ax = plt.subplots()
    if np.any(mask): ax.loglog(x[mask], y[mask])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, which="both")
    return fig

def figure_semilogx(x, y, xlabel, ylabel, title):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y) & (x>0)
    fig, ax = plt.subplots()
    if np.any(mask): ax.semilogx(x[mask], y[mask])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, which="both")
    return fig

def make_pdf_report(df, meta_text, metrics):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1 — Titre + méta + métriques
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
        ax.axis("off")
        lines = [
            "S2P Analyzer Pro — Rapport de caractérisation",
            "",
            meta_text,
            "",
            "Métriques clés :",
            f"• SRF (Self-Resonant Frequency) : {metrics['SRF_Hz']:.6g} Hz" if np.isfinite(metrics['SRF_Hz']) else "• SRF : N/A",
            f"• Cp (basse fréquence) : {metrics['Cp_lowF_F']:.6g} F" if np.isfinite(metrics['Cp_lowF_F']) else "• Cp (basse fréquence) : N/A",
            f"• ESL (haute fréquence) : {metrics['ESL_highF_H']:.6g} H" if np.isfinite(metrics['ESL_highF_H']) else "• ESL (haute fréquence) : N/A",
            f"• ESR(f0) : {metrics['ESR_at_f0_Ohm']:.6g} Ω" if np.isfinite(metrics['ESR_at_f0_Ohm']) else "• ESR(f0) : N/A",
            f"• tanδ(f0) : {metrics['tanD_at_f0']:.6g}" if np.isfinite(metrics['tanD_at_f0']) else "• tanδ(f0) : N/A",
            f"• Q(f0)=1/tanδ : {metrics['Q_at_f0']:.6g}" if np.isfinite(metrics['Q_at_f0']) else "• Q(f0) : N/A",
        ]
        ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=11)
        pdf.savefig(fig); plt.close(fig)

        f = df["freq_Hz"].values
        pdf.savefig(figure_loglog(f, np.abs(df["Cp_F"].values), "Fréquence (Hz)", "|Cp| (F)", "Capacitance parallèle Cp")); plt.close()
        pdf.savefig(figure_semilogx(f, df["tanD"].values, "Fréquence (Hz)", "tanδ", "Facteur de pertes tanδ")); plt.close()
        pdf.savefig(figure_loglog(f, np.clip(df["ESR_Ohm"].values.astype(float),1e-15,None), "Fréquence (Hz)", "ESR (Ω)", "Résistance série équivalente (ESR)")); plt.close()
        pdf.savefig(figure_loglog(f, np.abs(df["ESL_H"].values.astype(float)), "Fréquence (Hz)", "ESL (H)", "Inductance série équivalente (ESL)")); plt.close()
    buf.seek(0)
    return buf

# =========================
# UI — Sidebar
# =========================
st.title("Analyse .s2p – Pro")
uploaded = st.file_uploader("Dépose ton fichier .s2p", type=["s2p"])

with st.sidebar:
    st.header("Prétraitement & Options")
    fmin = st.number_input("fmin (Hz)", value=1.0, min_value=0.0, step=1.0, format="%.6f")
    fmax = st.number_input("fmax (Hz)", value=1e11, min_value=0.0, step=1.0, format="%.6f")
    decim = st.number_input("Décimation (1=no)", min_value=1, value=1, step=1)
    st.caption("Décimer = garder 1 point sur N pour accélérer.")

    st.markdown("---")
    st.subheader("Lissage (Savitzky–Golay)")
    use_sg = st.checkbox("Activer le lissage", value=False)
    sg_win = st.number_input("Fenêtre (impair)", min_value=3, value=9, step=2)
    sg_poly = st.number_input("Ordre polynôme", min_value=1, value=2, step=1)

    st.markdown("---")
    st.subheader("Métriques")
    target_freq = st.number_input("Fréquence f0 (Hz)", value=1e9, format="%.6f")
    low_cap_max = st.number_input("Bande Cp basse fréquence max (Hz)", value=1e6, format="%.6f")
    high_L_min = st.number_input("Bande ESL min (Hz)", value=1e9, format="%.6f")

# =========================
# Main logic
# =========================
if uploaded:
    try:
        freqs, S11, S21, S12, S22, Z0, fmt, funit = parse_touchstone_s2p(uploaded)
        df, Zin, Yin = compute_params_from_s11(freqs, S11, Z0)
        meta = f"Z0={Z0:.2f} Ω | Format={fmt} | Unité={funit} | N={len(df)}"
        st.success(meta)

        # Bande + décimation
        df = apply_band_and_decimate(df, fmin, fmax, int(decim))

        # Lissage optionnel
        if use_sg:
            for col in ["Cp_F", "tanD", "ESR_Ohm", "ESL_H"]:
                y = df[col].values.astype(float)
                df[col] = apply_savgol(y, sg_win, sg_poly)

        # Tableau
        st.dataframe(df.head(500), use_container_width=True)

        # Graphes (2 colonnes)
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(figure_loglog(df["freq_Hz"], np.abs(df["Cp_F"]), "Fréquence (Hz)", "|Cp| (F)", "Capacitance parallèle Cp"))
            st.pyplot(figure_loglog(df["freq_Hz"], np.clip(df["ESR_Ohm"].astype(float),1e-15,None), "Fréquence (Hz)", "ESR (Ω)", "Résistance série équivalente (ESR)"))
        with c2:
            st.pyplot(figure_semilogx(df["freq_Hz"], df["tanD"], "Fréquence (Hz)", "tanδ", "Facteur de pertes tanδ"))
            st.pyplot(figure_loglog(df["freq_Hz"], np.abs(df["ESL_H"].astype(float)), "Fréquence (Hz)", "ESL (H)", "Inductance série équivalente (ESL)"))

        # Métriques comme un prof de carac
        metrics = estimate_metrics(df, target_freq, low_cap_max, high_L_min)
        colm = st.columns(3)
        colm[0].metric("SRF (Hz)", f"{metrics['SRF_Hz']:.6g}" if np.isfinite(metrics['SRF_Hz']) else "N/A")
        colm[1].metric("Cp basse fréquence (F)", f"{metrics['Cp_lowF_F']:.6g}" if np.isfinite(metrics['Cp_lowF_F']) else "N/A")
        colm[2].metric("ESL haute fréquence (H)", f"{metrics['ESL_highF_H']:.6g}" if np.isfinite(metrics['ESL_highF_H']) else "N/A")
        colm2 = st.columns(3)
        colm2[0].metric("ESR @ f0 (Ω)", f"{metrics['ESR_at_f0_Ohm']:.6g}" if np.isfinite(metrics['ESR_at_f0_Ohm']) else "N/A")
        colm2[1].metric("tanδ @ f0", f"{metrics['tanD_at_f0']:.6g}" if np.isfinite(metrics['tanD_at_f0']) else "N/A")
        colm2[2].metric("Q @ f0", f"{metrics['Q_at_f0']:.6g}" if np.isfinite(metrics['Q_at_f0']) else "N/A")

        # Exports
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger CSV", csv, "analysis_s2p_pro.csv", "text/csv")

        pdf_buf = make_pdf_report(df, meta, metrics)
        st.download_button("Télécharger rapport PDF", data=pdf_buf, file_name="rapport_s2p_pro.pdf", mime="application/pdf")

    except Exception as e:
        st.error(f"Erreur : {e}")
else:
    st.info("Charge un fichier .s2p pour commencer.")
