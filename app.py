# app.py  — Analyse .s2p : Cp / tanδ / ESR / ESL (Streamlit)
import re, math, cmath, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Analyse .s2p – Cp / tanδ / ESR / ESL", layout="wide")

# ---------- Parsing ----------
def parse_touchstone_s2p(file_like):
    freq_unit = "Hz"
    data_format = "RI"  # RI, MA, DB
    z0 = 50.0
    freqs, S11_list, S21_list, S12_list, S22_list = [], [], [], [], []

    for raw in file_like.read().decode("utf-8", errors="ignore").splitlines():
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
                    try:
                        z0 = float(tokens[idx + 1])
                    except Exception:
                        pass
            continue

        parts = re.split(r"\s+", line)
        if len(parts) < 9:
            continue
        try:
            fval = float(parts[0])
        except Exception:
            continue

        mult = {"Hz":1.0, "KHz":1e3, "kHz":1e3, "MHz":1e6, "GHz":1e9}
        freq_Hz = fval * mult.get(freq_unit, 1.0)
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

        freqs.append(freq_Hz); S11_list.append(S11); S21_list.append(S21)
        S12_list.append(S12);  S22_list.append(S22)

    if not freqs:
        raise ValueError("Aucune donnée valide dans le .s2p")
    return (np.asarray(freqs,float),
            np.asarray(S11_list,complex),
            np.asarray(S21_list,complex),
            np.asarray(S12_list,complex),
            np.asarray(S22_list,complex),
            float(z0), data_format, freq_unit)

# ---------- Calculs ----------
def s11_to_zin(S11, Z0): return Z0 * (1 + S11) / (1 - S11)

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
    })

def plot_loglog(x, y, xlabel, ylabel, title):
    mask = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0)
    fig, ax = plt.subplots()
    if np.any(mask): ax.loglog(x[mask], y[mask])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, which="both")
    return fig

def plot_semilogx(x, y, xlabel, ylabel, title):
    mask = np.isfinite(x) & np.isfinite(y) & (x>0)
    fig, ax = plt.subplots()
    if np.any(mask): ax.semilogx(x[mask], y[mask])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, which="both")
    return fig

# ---------- UI ----------
st.title("Analyse .s2p – Cp / tanδ / ESR / ESL")
uploaded = st.file_uploader("Dépose ton fichier .s2p", type=["s2p"])

if uploaded:
    try:
        freqs, S11, S21, S12, S22, Z0, fmt, funit = parse_touchstone_s2p(uploaded)
        df = compute_params_from_s11(freqs, S11, Z0)

        st.success(f"Z0={Z0:.2f} Ω | Format={fmt} | Unité={funit}")
        st.dataframe(df.head(200))

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_loglog(df["freq_Hz"].values, np.abs(df["Cp_F"].values),
                                  "Fréquence (Hz)", "|Cp| (F)", "Capacitance parallèle Cp"))
            st.pyplot(plot_loglog(df["freq_Hz"].values, 
                                  np.clip(df["ESR_Ohm"].values.astype(float),1e-15,None),
                                  "Fréquence (Hz)", "ESR (Ω)", "Résistance série équivalente (ESR)"))
        with c2:
            st.pyplot(plot_semilogx(df["freq_Hz"].values, df["tanD"].values,
                                    "Fréquence (Hz)", "tanδ", "Facteur de pertes tanδ"))
            esl = df["ESL_H"].values.astype(float)
            esl[~np.isfinite(esl)] = np.nan
            st.pyplot(plot_loglog(df["freq_Hz"].values, np.abs(esl),
                                  "Fréquence (Hz)", "ESL (H)", "Inductance série équivalente (ESL)"))

        # bouton de téléchargement CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger CSV", csv, "analysis_s2p_cp_tand_esr_esl.csv", "text/csv")
    except Exception as e:
        st.error(f"Erreur : {e}")
else:
    st.info("Charge un fichier .s2p pour commencer.")
