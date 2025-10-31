# app.py — Analyse .s2p : Cp / tanδ / ESR / ESL (UI deluxe: Plotly + badges unités + annotations)
import re, math, cmath, io
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# ----------------- Page & thème (CSS léger) -----------------
st.set_page_config(page_title="Analyse .s2p — Cp / tanδ / ESR / ESL", layout="wide")
st.markdown("""
<style>
.main .block-container {max-width: 1250px; padding-top: 1rem;}
.badge {display:inline-block; padding:2px 8px; border-radius:999px; font-size:.75rem;
        background:#eef2ff; color:#3730a3; border:1px solid #c7d2fe; margin-left:6px;}
.card {background:#fff; border:1px solid #ececec; border-radius:16px; padding:14px 16px;
       box-shadow:0 6px 18px rgba(0,0,0,.05);}
.kpi-title {font-size:.9rem; color:#6b7280; margin-bottom:4px;}
.kpi-value {font-size:1.55rem; font-weight:800; color:#111827; line-height:1.1;}
.kpi-sub {font-size:.8rem; color:#6b7280;}
.section-title {font-weight:800; font-size:1.15rem; letter-spacing:.2px;}
hr {border:none; border-top:1px solid #eee; margin:16px 0;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------- Conversions d’unités -----------------
SI_FREQ = {"Hz":1.0, "kHz":1e3, "MHz":1e6, "GHz":1e9}
SI_CAP  = {"F":1.0, "mF":1e-3, "µF":1e-6, "nF":1e-9, "pF":1e-12, "fF":1e-15}
SI_IND  = {"H":1.0, "mH":1e-3, "µH":1e-6, "nH":1e-9, "pH":1e-12}
SI_RES  = {"Ω":1.0, "mΩ":1e-3, "kΩ":1e3}
def fmt(x, g=6): 
    return "N/A" if x is None or not np.isfinite(x) else f"{x:.{g}g}"
def interp_at(x, y, x0):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m): return np.nan
    return float(np.interp(x0, x[m], y[m]))

# ----------------- Parsing Touchstone -----------------
def parse_touchstone_s2p(file_like):
    freq_unit = "Hz"; fmt = "RI"; z0 = 50.0
    F, s11, s21, s12, s22 = [], [], [], [], []
    text = file_like.read().decode("utf-8", errors="ignore")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("!"): continue
        if line.startswith("#"):
            t = line[1:].strip().split(); T = [x.upper() for x in t]
            for u in ["HZ","KHZ","MHZ","GHZ"]:
                if u in T: freq_unit = u.capitalize(); break
            for f in ["RI","MA","DB"]:
                if f in T: fmt = f; break
            if "R" in T:
                i = T.index("R")
                if i+1 < len(t):
                    try: z0 = float(t[i+1])
                    except: pass
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 9: continue
        try: fval = float(parts[0])
        except: continue
        mult = {"Hz":1.0,"KHz":1e3,"kHz":1e3,"MHz":1e6,"GHz":1e9}
        f_Hz = fval*mult.get(freq_unit,1.0)
        a = list(map(float, parts[1:9]))
        def cplx(x,y):
            if fmt=="RI": return complex(x,y)
            if fmt=="MA": return x*cmath.exp(1j*math.radians(y))
            if fmt=="DB": return (10**(x/20.0))*cmath.exp(1j*math.radians(y))
            return complex(x,y)
        S11=cplx(a[0],a[1]); S21=cplx(a[2],a[3]); S12=cplx(a[4],a[5]); S22=cplx(a[6],a[7])
        F.append(f_Hz); s11.append(S11); s21.append(S21); s12.append(S12); s22.append(S22)
    if not F: raise ValueError("Aucune donnée valide dans le .s2p")
    return np.array(F,float), np.array(s11,complex), np.array(s21,complex), np.array(s12,complex), np.array(s22,complex), z0, fmt, freq_unit

# ----------------- Calculs -----------------
def s11_to_zin(S11, Z0): return Z0*(1+S11)/(1-S11)
def compute_params_from_s11(freqs, S11, Z0):
    w = 2*np.pi*freqs
    Zin = s11_to_zin(S11, Z0); Yin = 1.0/Zin
    G, B = np.real(Yin), np.imag(Yin)
    with np.errstate(divide='ignore', invalid='ignore'):
        Cp = np.where(w!=0, B/w, np.nan)
        tanD = np.where(B!=0, G/np.abs(B), np.nan)
    Rs, Xs = np.real(Zin), np.imag(Zin)
    with np.errstate(divide='ignore', invalid='ignore'):
        Cs = np.where(Xs<0, -1.0/(w*Xs), np.nan)
        Ls = np.where(Xs>0,  Xs/w, np.nan)
    df = pd.DataFrame({"freq_Hz":freqs,"Cp_F":Cp,"tanD":tanD,"ESR_Ohm":Rs,"ESL_H":Ls,"Cs_F":Cs,"Ls_H":Ls})
    return df, Zin, Yin

def band_decimate(df, fmin, fmax, decim):
    d = df[(df["freq_Hz"]>=fmin) & (df["freq_Hz"]<=fmax)].copy()
    if decim>1: d = d.iloc[::decim,:].reset_index(drop=True)
    return d

def savgol_opt(y, win, poly):
    if not win or not poly: return y
    win = int(win); 
    if win%2==0: win += 1
    if win<3 or win>len(y): return y
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, win, int(poly))
    except Exception:
        return y

# ----------------- UI -----------------
st.title("Analyse .s2p — Cp / tanδ / ESR / ESL")
uploaded = st.file_uploader("Dépose ton fichier .s2p", type=["s2p"])

with st.sidebar:
    st.header("Paramètres d’analyse")
    colf = st.columns(2)
    f0_val = colf[0].number_input("f₀", value=1.0, min_value=0.0, format="%.6f")
    f0_u   = colf[1].selectbox("Unité", list(SI_FREQ.keys()), index=3)  # GHz
    f0 = f0_val*SI_FREQ[f0_u]

    colb = st.columns(4)
    fmin = colb[0].number_input("fmin (Hz)", value=1.0, min_value=0.0)
    fmax = colb[1].number_input("fmax (Hz)", value=1e11, min_value=0.0)
    deci = colb[2].number_input("Décimation", min_value=1, value=1, step=1)
    use_sg = colb[3].checkbox("Lissage SG", value=False)

    if use_sg:
        colsg = st.columns(2)
        sg_win  = int(colsg[0].number_input("Fenêtre (impair)", min_value=3, value=9, step=2))
        sg_poly = int(colsg[1].number_input("Ordre", min_value=1, value=2, step=1))
    else:
        sg_win = sg_poly = None

    st.markdown("---")
    st.subheader("Unités KPI")
    colu = st.columns(4)
    uC = colu[0].selectbox("Cp", list(SI_CAP.keys()), index=4)   # pF
    uL = colu[1].selectbox("ESL", list(SI_IND.keys()), index=3)  # nH
    uR = colu[2].selectbox("ESR", list(SI_RES.keys()), index=0)  # Ω
    uF = colu[3].selectbox("Fréquence", list(SI_FREQ.keys()), index=3)  # GHz

if uploaded:
    try:
        F, S11, *_rest, Z0, fmt, funit = parse_touchstone_s2p(uploaded)
        df, Zin, Yin = compute_params_from_s11(F, S11, Z0)
        df = band_decimate(df, fmin, fmax, int(deci))
        if use_sg:
            for c in ["Cp_F","tanD","ESR_Ohm","ESL_H"]:
                df[c] = savgol_opt(df[c].values.astype(float), sg_win, sg_poly)

        st.success(f"Z0={Z0:.2f} Ω | Format={fmt} | Unité entête={funit} | Points={len(df)}")

        # --- Interpolation @ f0 ---
        Cp_f0   = interp_at(df["freq_Hz"], df["Cp_F"], f0)
        ESR_f0  = interp_at(df["freq_Hz"], df["ESR_Ohm"], f0)
        ESL_f0  = interp_at(df["freq_Hz"], df["ESL_H"], f0)
        tanD_f0 = interp_at(df["freq_Hz"], df["tanD"], f0)
        Q_f0    = (1.0/tanD_f0) if np.isfinite(tanD_f0) and tanD_f0>0 else np.nan
        if np.isfinite(tanD_f0) and tanD_f0>1:
            st.warning("tanδ@f₀ > 1 → Q très faible. Tu es peut-être hors régime capacitif (proche SRF) ou sans dé-embedding.")

        # --- Détection SRF (transition C→L) ---
        Cs = df["Cs_F"].values
        srf = np.nan
        sign = np.sign(np.nan_to_num(Cs)*-1)  # capacitif => signe opposé
        tr = np.where(np.diff(sign)!=0)[0]
        if tr.size>0: srf = df.loc[tr[0]+1, "freq_Hz"]

        # --- KPIs (cartes) ---
        f0_disp  = f0/SI_FREQ[uF]
        ESR_disp = ESR_f0/SI_RES[uR] if np.isfinite(ESR_f0) else None
        Cp_disp  = Cp_f0/SI_CAP[uC]  if np.isfinite(Cp_f0)  else None
        ESL_disp = ESL_f0/SI_IND[uL] if np.isfinite(ESL_f0) else None

        k1,k2,k3 = st.columns(3)
        with k1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-title">Fréquence f₀ <span class="badge">{uF}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{fmt(f0_disp)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-sub">SRF ≈ {fmt(srf/SI_FREQ[uF] if np.isfinite(srf) else None)} {uF}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with k2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-title">Cp @ f₀ <span class="badge">{uC}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{fmt(Cp_disp)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-sub">tanδ @ f₀ = {fmt(tanD_f0)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with k3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-title">ESR / ESL @ f₀ <span class="badge">{uR}</span> / <span class="badge">{uL}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{fmt(ESR_disp)} / {fmt(ESL_disp)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-sub">Q @ f₀ = {fmt(Q_f0)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ----------------- Onglets -----------------
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Courbes", "🧮 Tableau", "📝 Rapport PDF", "⚙️ Détails"])
        f = df["freq_Hz"].values

        # ---- Courbes (Plotly interactif) ----
        with tab1:
            # Cp & tanδ
            row1 = st.columns(2)
            with row1[0]:
                y = np.abs(df["Cp_F"].values.astype(float))
                mask = np.isfinite(f) & np.isfinite(y) & (f>0) & (y>0)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=f[mask], y=y[mask], mode="lines", name="|Cp| (F)"))
                if np.isfinite(f0): fig.add_vline(x=f0, line_dash="dot", annotation_text="f₀", annotation_position="top right")
                if np.isfinite(srf): fig.add_vline(x=srf, line_dash="dot", line_color="orange", annotation_text="SRF", annotation_position="top right")
                fig.update_layout(title="Capacitance parallèle Cp", xaxis_type="log", yaxis_type="log",
                                  xaxis_title="Fréquence (Hz)", yaxis_title="|Cp| (F)", height=420)
                st.plotly_chart(fig, use_container_width=True)
            with row1[1]:
                y = df["tanD"].values.astype(float)
                mask = np.isfinite(f) & np.isfinite(y) & (f>0)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=f[mask], y=y[mask], mode="lines", name="tanδ"))
                if np.isfinite(f0): fig.add_vline(x=f0, line_dash="dot", annotation_text="f₀", annotation_position="top right")
                if np.isfinite(srf): fig.add_vline(x=srf, line_dash="dot", line_color="orange", annotation_text="SRF", annotation_position="top right")
                fig.update_layout(title="Facteur de pertes tanδ", xaxis_type="log",
                                  xaxis_title="Fréquence (Hz)", yaxis_title="tanδ (—)", height=420)
                st.plotly_chart(fig, use_container_width=True)

            # ESR & ESL
            row2 = st.columns(2)
            with row2[0]:
                y = np.clip(df["ESR_Ohm"].values.astype(float), 1e-15, None)
                mask = np.isfinite(f) & np.isfinite(y) & (f>0) & (y>0)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=f[mask], y=y[mask], mode="lines", name="ESR (Ω)"))
                if np.isfinite(f0): fig.add_vline(x=f0, line_dash="dot", annotation_text="f₀", annotation_position="top right")
                fig.update_layout(title="Résistance série équivalente (ESR)", xaxis_type="log", yaxis_type="log",
                                  xaxis_title="Fréquence (Hz)", yaxis_title="ESR (Ω)", height=420)
                st.plotly_chart(fig, use_container_width=True)
            with row2[1]:
                y = np.abs(df["ESL_H"].values.astype(float))
                mask = np.isfinite(f) & np.isfinite(y) & (f>0) & (y>0)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=f[mask], y=y[mask], mode="lines", name="ESL (H)"))
                if np.isfinite(f0): fig.add_vline(x=f0, line_dash="dot", annotation_text="f₀", annotation_position="top right")
                if np.isfinite(srf): fig.add_vline(x=srf, line_dash="dot", line_color="orange", annotation_text="SRF", annotation_position="top right")
                fig.update_layout(title="Inductance série équivalente (ESL)", xaxis_type="log", yaxis_type="log",
                                  xaxis_title="Fréquence (Hz)", yaxis_title="ESL (H)", height=420)
                st.plotly_chart(fig, use_container_width=True)

        # ---- Tableau ----
        with tab2:
            st.write("Astuce : utilise la loupe en haut à droite du tableau pour rechercher.")
            st.dataframe(df, use_container_width=True)

        # ---- Rapport PDF (mêmes unités que KPI) ----
        with tab3:
            def make_pdf(df):
                buf = io.BytesIO()
                with PdfPages(buf) as pdf:
                    # Page 1 — KPIs
                    fig, ax = plt.subplots(figsize=(8.27, 11.69))
                    ax.axis("off")
                    lines = [
                        "Rapport d'analyse .s2p",
                        "",
                        f"f₀ = {fmt(f0/SI_FREQ[uF])} {uF}",
                        f"Cp(f₀) = {fmt(Cp_f0/SI_CAP[uC] if np.isfinite(Cp_f0) else None)} {uC}",
                        f"ESR(f₀) = {fmt(ESR_f0/SI_RES[uR] if np.isfinite(ESR_f0) else None)} {uR}",
                        f"ESL(f₀) = {fmt(ESL_f0/SI_IND[uL] if np.isfinite(ESL_f0) else None)} {uL}",
                        f"tanδ(f₀) = {fmt(tanD_f0)} (—)",
                        f"Q(f₀) = {fmt(Q_f0)} (—)",
                        f"SRF ≈ {fmt(srf/SI_FREQ[uF] if np.isfinite(srf) else None)} {uF}",
                    ]
                    ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=11)
                    pdf.savefig(fig); plt.close(fig)
                buf.seek(0); return buf
            pdf_buf = make_pdf(df)
            st.download_button("Télécharger rapport PDF", data=pdf_buf, file_name="rapport_s2p.pdf", mime="application/pdf")

        # ---- Détails bruts ----
        with tab4:
            st.json({
                "Z0": Z0, "format": fmt, "unite_entete": funit,
                "f0_Hz": f0, "SRF_Hz": srf,
                "Cp_f0_F": float(Cp_f0) if np.isfinite(Cp_f0) else None,
                "ESR_f0_Ohm": float(ESR_f0) if np.isfinite(ESR_f0) else None,
                "ESL_f0_H": float(ESL_f0) if np.isfinite(ESL_f0) else None,
                "tanD_f0": float(tanD_f0) if np.isfinite(tanD_f0) else None,
                "Q_f0": float(Q_f0) if np.isfinite(Q_f0) else None
            })

        # Export CSV global
        st.download_button("Télécharger CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="analysis_s2p.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Erreur : {e}")
else:
    st.info("Charge un fichier .s2p pour commencer.")
