# app.py — Analyse .s2p : Cp / tanδ / ESR / ESL (UI premium)
import re, math, cmath, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import streamlit as st
from scipy.signal import savgol_filter
import plotly.graph_objects as go

# ================== PAGE & STYLE ==================
st.set_page_config(page_title="Analyse .s2p — Cp / tanδ / ESR / ESL", layout="wide", page_icon="📈")
st.markdown("""
<style>
:root{
  --card-bg: rgba(255,255,255,0.75);
  --glass: blur(10px) saturate(1.2);
}
.main .block-container {padding-top: 0.8rem; padding-bottom: 2rem; max-width: 1200px;}
/* Hero */
.hero{
  background: linear-gradient(135deg,#141e30 0%,#243b55 50%, #3b82f6 100%);
  border-radius: 18px;
  padding: 26px 28px;
  color: #fff;
  box-shadow: 0 16px 40px rgba(0,0,0,.25);
  position: relative; overflow: hidden;
}
.hero h1{margin: 0; letter-spacing:.3px;}
.hero p{opacity:.9; margin:.2rem 0 0;}
/* Badges */
.badges{display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.8rem}
.badge{
  background: rgba(255,255,255,.12);
  border: 1px solid rgba(255,255,255,.25);
  padding: 6px 10px; border-radius: 999px; font-size:.86rem;
  display:flex; align-items:center; gap:.45rem;
}
/* Cards */
.kpi-grid{display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; margin-top:14px;}
.kpi-card{
  backdrop-filter: var(--glass);
  background: var(--card-bg);
  border: 1px solid rgba(0,0,0,.06);
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 6px 20px rgba(0,0,0,.06);
}
.kpi-title{font-size:.88rem; color:#5b6574; margin:0 0 6px;}
.kpi-value{font-size:1.9rem; font-weight:800; color:#0f172a; letter-spacing:.2px;}
.kpi-sub{font-size:.82rem; color:#6b7280; margin-top:2px}
.sep{border:none; border-top:1px solid #eceff3; margin:18px 0}
/* Warnings */
.stAlert{border-radius:14px !important; border:1px solid #fef3c7 !important}
/* Tabs */
[data-baseweb="tab-list"]{gap:.4rem}
</style>
""", unsafe_allow_html=True)

# ================== UTILS UNITÉS ==================
SI_FREQ = {"Hz":1.0, "kHz":1e3, "MHz":1e6, "GHz":1e9}
SI_CAP  = {"F":1.0, "mF":1e-3, "µF":1e-6, "nF":1e-9, "pF":1e-12, "fF":1e-15}
SI_IND  = {"H":1.0, "mH":1e-3, "µH":1e-6, "nH":1e-9, "pH":1e-12}
SI_RES  = {"Ω":1.0, "mΩ":1e-3, "kΩ":1e3}

def to_unit(x, factor): 
    return None if (x is None or not np.isfinite(x)) else float(x)/factor

def fmt_num(x, digits=6):
    return "N/A" if (x is None or not np.isfinite(x)) else f"{x:.{digits}g}"

# ================== PARSING ==================
def parse_touchstone_s2p(file_like):
    freq_unit = "Hz"; data_format = "RI"; z0 = 50.0
    freqs, S11_list, S21_list, S12_list, S22_list = [], [], [], [], []
    text = file_like.read().decode("utf-8", errors="ignore")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("!"): 
            continue
        if line.startswith("#"):
            t = line[1:].strip().split(); T = [x.upper() for x in t]
            for u in ["HZ","KHZ","MHZ","GHZ"]:
                if u in T: freq_unit = u.capitalize(); break
            for f in ["RI","MA","DB"]:
                if f in T: data_format = f; break
            if "R" in T:
                i = T.index("R")
                if i+1 < len(t):
                    try: z0 = float(t[i+1])
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
        f_Hz = fval*mult.get(freq_unit,1.0)
        a = list(map(float, parts[1:9]))
        def cplx(x,y,fmtname):
            if fmtname=="RI": return complex(x,y)
            if fmtname=="MA": return x*cmath.exp(1j*math.radians(y))
            if fmtname=="DB": return (10**(x/20.0))*cmath.exp(1j*math.radians(y))
            return complex(x,y)
        S11=cplx(a[0],a[1],data_format); S21=cplx(a[2],a[3],data_format)
        S12=cplx(a[4],a[5],data_format); S22=cplx(a[6],a[7],data_format)
        freqs.append(f_Hz); S11_list.append(S11); S21_list.append(S21); S12_list.append(S12); S22_list.append(S22)
    if not freqs: 
        raise ValueError("Aucune donnée valide dans le .s2p")
    return (np.asarray(freqs,float),
            np.asarray(S11_list,complex),
            np.asarray(S21_list,complex),
            np.asarray(S12_list,complex),
            np.asarray(S22_list,complex),
            float(z0), data_format, freq_unit)

# ================== CALCULS ==================
def s11_to_zin(S11, Z0): 
    return Z0*(1+S11)/(1-S11)

def compute_params_from_s11(freqs, S11, Z0):
    w = 2*np.pi*freqs
    Zin = s11_to_zin(S11, Z0)
    Yin = 1.0/Zin
    G, B = np.real(Yin), np.imag(Yin)
    with np.errstate(divide='ignore', invalid='ignore'):
        Cp   = np.where(w!=0, B/w, np.nan)
        tanD = np.where(B!=0, G/np.abs(B), np.nan)
    Rs, Xs = np.real(Zin), np.imag(Zin)
    with np.errstate(divide='ignore', invalid='ignore'):
        Cs = np.where(Xs<0, -1.0/(w*Xs), np.nan)
        Ls = np.where(Xs>0,  Xs/w, np.nan)
    df = pd.DataFrame({"freq_Hz":freqs,"Cp_F":Cp,"tanD":tanD,"ESR_Ohm":Rs,"ESL_H":Ls,"Cs_F":Cs})
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
    try: return savgol_filter(y, win, int(poly))
    except: return y

def interp_at(f, y, f0):
    f = np.asarray(f,float); y = np.asarray(y,float)
    mask = np.isfinite(f) & np.isfinite(y)
    if not np.any(mask): return np.nan
    return float(np.interp(f0, f[mask], y[mask]))

# ================== HEADER ==================
st.markdown("""
<div class="hero">
  <h1>Analyse <i>.s2p</i> — Cp / tanδ / ESR / ESL</h1>
  <p>Lecture Touchstone, extraction paramétrique, KPIs à f₀, courbes & rapport.</p>
  <div class="badges">
    <div class="badge">🔧 Extraction série/parallèle</div>
    <div class="badge">📎 Export CSV & PDF</div>
    <div class="badge">📡 Lissage Savitzky–Golay</div>
    <div class="badge">⚡ SRF & régimes détectables</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.write("")

# ================== SIDEBAR ==================
uploaded = st.file_uploader("Dépose ton fichier .s2p", type=["s2p"])

with st.sidebar:
    st.header("Options d'analyse")
    c1,c2 = st.columns(2)
    f0_val = c1.number_input("Fréquence f₀", value=1.0, min_value=0.0, format="%.6f")
    f0_unit = c2.selectbox("Unité f₀", list(SI_FREQ.keys()), index=3)  # GHz

    s1,s2 = st.columns(2)
    fmin_val = s1.number_input("fmin", value=1.0, min_value=0.0, format="%.6f")
    fmin_unit = s2.selectbox("Unité fmin", list(SI_FREQ.keys()), index=0)
    t1,t2 = st.columns(2)
    fmax_val = t1.number_input("fmax", value=1e11, min_value=0.0, format="%.6f")
    fmax_unit = t2.selectbox("Unité fmax", list(SI_FREQ.keys()), index=3)

    decim = st.number_input("Décimation (1 = aucune)", min_value=1, value=1, step=1)

    st.markdown("---")
    st.subheader("Lissage (Savitzky–Golay)")
    use_sg = st.checkbox("Activer le lissage", value=False)
    sg_win = st.number_input("Fenêtre (impair)", min_value=3, value=9, step=2)
    sg_poly = st.number_input("Ordre polynôme", min_value=1, value=2, step=1)

    st.markdown("---")
    st.subheader("Unités d'affichage (KPIs)")
    uC  = st.selectbox("Unité Cp", list(SI_CAP.keys()), index=4)     # pF
    uL  = st.selectbox("Unité ESL", list(SI_IND.keys()), index=3)    # nH
    uR  = st.selectbox("Unité ESR", list(SI_RES.keys()), index=0)    # Ω
    uF0 = st.selectbox("Unité fréquence", list(SI_FREQ.keys()), index=3)  # GHz

# valeurs SI
f0   = f0_val*SI_FREQ[f0_unit]
fmin = fmin_val*SI_FREQ[fmin_unit]
fmax = fmax_val*SI_FREQ[fmax_unit]

# ================== MAIN ==================
if uploaded:
    try:
        freqs, S11, S21, S12, S22, Z0, data_fmt, funit = parse_touchstone_s2p(uploaded)
        df, Zin, Yin = compute_params_from_s11(freqs, S11, Z0)
        meta = f"Z0={Z0:.2f} Ω | Format={data_fmt} | Unité entête={funit} | Points={len(df)}"
        st.success(meta)

        # bande + décimation + lissage
        df = band_decimate(df, fmin, fmax, int(decim))
        if use_sg:
            for col in ["Cp_F","tanD","ESR_Ohm","ESL_H"]:
                df[col] = savgol_opt(df[col].values.astype(float), sg_win, sg_poly)

        # KPIs @ f0
        Cp_f0   = interp_at(df["freq_Hz"], df["Cp_F"],    f0)
        ESR_f0  = interp_at(df["freq_Hz"], df["ESR_Ohm"], f0)
        ESL_f0  = interp_at(df["freq_Hz"], df["ESL_H"],   f0)
        tanD_f0 = interp_at(df["freq_Hz"], df["tanD"],    f0)
        Q_f0    = (1.0/tanD_f0) if np.isfinite(tanD_f0) and tanD_f0>0 else np.nan

        if np.isfinite(tanD_f0) and tanD_f0>1:
            st.warning("tanδ@f₀ > 1 → Q très faible. Vérifie le régime (inductif / voisinage SRF) ou le de-embedding.")

        if (f0<df["freq_Hz"].min()) or (f0>df["freq_Hz"].max()):
            st.info("f₀ est hors de la bande affichée → interpolation impossible dans cette bande.")

        # Conversions pour affichage
        Cp_disp  = to_unit(Cp_f0,  SI_CAP[uC])
        ESL_disp = to_unit(ESL_f0, SI_IND[uL])
        ESR_disp = to_unit(ESR_f0, SI_RES[uR])
        f0_disp  = to_unit(f0,     SI_FREQ[uF0])

        # ================== KPIs CARDS ==================
        st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-title">Fréquence f₀ ({uF0})</div><div class="kpi-value">🛰️ {fmt_num(f0_disp)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-sub">Référence d’évaluation des métriques</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-title">ESR @ f₀ ({uR})</div><div class="kpi-value">⚙️ {fmt_num(ESR_disp)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-sub">Résistance série équivalente</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-title">Cp @ f₀ ({uC})</div><div class="kpi-value">📦 {fmt_num(Cp_disp)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-sub">Capacitance parallèle extraite de B/ω</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-title">tanδ @ f₀ (—)</div><div class="kpi-value">🔥 {fmt_num(tanD_f0)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-sub">tanδ = G/|B|</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-title">ESL @ f₀ ({uL})</div><div class="kpi-value">🧲 {fmt_num(ESL_disp)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-sub">Inductance série équivalente (X>0)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-title">Q @ f₀ (—)</div><div class="kpi-value">🎯 {fmt_num(Q_f0)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-sub">Q = 1/tanδ</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<hr class="sep">', unsafe_allow_html=True)

        # ================== JAUGES (Plotly) ==================
        def gauge(value, title, suffix="", vmax=None):
            if not np.isfinite(value): value, vmax = 0, 1
            if vmax is None:
                vmax = 10**np.ceil(np.log10(abs(value)+1e-30))
            return go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                number={'suffix': f" {suffix}"},
                gauge={'axis': {'range': [None, vmax]},
                       'bar': {'thickness': 0.3},
                       'borderwidth': 1,
                       'bgcolor': "white"},
                title={'text': title}
            ))

        g1,g2,g3 = st.columns(3)
        with g1:  st.plotly_chart(gauge(Cp_disp or 0,  f"Cp @ f₀ ({uC})",  suffix=f"{uC}"), use_container_width=True, config={"displayModeBar": False})
        with g2:  st.plotly_chart(gauge(ESR_disp or 0, f"ESR @ f₀ ({uR})", suffix=f"{uR}"), use_container_width=True, config={"displayModeBar": False})
        with g3:  st.plotly_chart(gauge(ESL_disp or 0, f"ESL @ f₀ ({uL})", suffix=f"{uL}"), use_container_width=True, config={"displayModeBar": False})

        st.markdown('<hr class="sep">', unsafe_allow_html=True)

        # ================== ONGLET COURBES / TABLE / EXPORT ==================
        tab1, tab2, tab3 = st.tabs(["📊 Courbes", "🧾 Données", "📎 Export"])

        with tab1:
            # Matplotlib (sobre + lisible)
            def fig_loglog(x, y, xlabel, ylabel, title, positive_y=True):
                x = np.asarray(x,float); y = np.asarray(y,float)
                m = np.isfinite(x) & np.isfinite(y) & (x>0)
                if positive_y: m &= (y>0)
                fig, ax = plt.subplots()
                if np.any(m): ax.loglog(x[m], y[m])
                ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, which="both")
                return fig
            def fig_semilogx(x, y, xlabel, ylabel, title):
                x = np.asarray(x,float); y = np.asarray(y,float)
                m = np.isfinite(x) & np.isfinite(y) & (x>0)
                fig, ax = plt.subplots()
                if np.any(m): ax.semilogx(x[m], y[m])
                ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, which="both")
                return fig

            c1,c2 = st.columns(2)
            with c1:
                st.pyplot(fig_loglog(df["freq_Hz"], np.abs(df["Cp_F"]), "Fréquence (Hz)", "|Cp| (F)", "Capacitance parallèle Cp"))
                st.pyplot(fig_loglog(df["freq_Hz"], np.clip(df["ESR_Ohm"].astype(float),1e-15,None), "Fréquence (Hz)", "ESR (Ω)", "Résistance série équivalente (ESR)"))
            with c2:
                st.pyplot(fig_semilogx(df["freq_Hz"], df["tanD"], "Fréquence (Hz)", "tanδ (—)", "Facteur de pertes tanδ"))
                st.pyplot(fig_loglog(df["freq_Hz"], np.abs(df["ESL_H"].astype(float)), "Fréquence (Hz)", "ESL (H)", "Inductance série équivalente (ESL)"))

        with tab2:
            st.dataframe(df, use_container_width=True)

        with tab3:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger CSV", csv, "analysis_s2p.csv", "text/csv")

            def make_pdf(df, meta):
                buf = io.BytesIO()
                with PdfPages(buf) as pdf:
                    # Page 1 — résumé
                    fig, ax = plt.subplots(figsize=(8.27, 11.69))
                    ax.axis("off")
                    lines = [
                        "Rapport d'analyse .s2p",
                        "",
                        meta,
                        "",
                        f"f₀ = {fmt_num(f0_disp)} {uF0}",
                        f"ESR(f₀) = {fmt_num(ESR_disp)} {uR}",
                        f"Cp(f₀) = {fmt_num(Cp_disp)} {uC}",
                        f"ESL(f₀) = {fmt_num(ESL_disp)} {uL}",
                        f"tanδ(f₀) = {fmt_num(tanD_f0)} (—)",
                        f"Q(f₀) = {fmt_num(Q_f0)} (—)",
                    ]
                    ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=11)
                    pdf.savefig(fig); plt.close(fig)
                    # Pages graphes
                    f = df["freq_Hz"].values
                    def fig_loglog_simple(x, y, xlabel, ylabel, title):
                        fig, ax = plt.subplots()
                        m = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0)
                        if np.any(m): ax.loglog(x[m], y[m])
                        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, which="both")
                        return fig
                    def fig_semilogx_simple(x, y, xlabel, ylabel, title):
                        fig, ax = plt.subplots()
                        m = np.isfinite(x) & np.isfinite(y) & (x>0)
                        if np.any(m): ax.semilogx(x[m], y[m])
                        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, which="both")
                        return fig
                    pdf.savefig(fig_loglog_simple(f, np.abs(df["Cp_F"].values), "Fréquence (Hz)", "|Cp| (F)", "Capacitance parallèle Cp")); plt.close()
                    pdf.savefig(fig_semilogx_simple(f, df["tanD"].values, "Fréquence (Hz)", "tanδ (—)", "Facteur de pertes tanδ")); plt.close()
                    pdf.savefig(fig_loglog_simple(f, np.clip(df["ESR_Ohm"].values.astype(float),1e-15,None), "Fréquence (Hz)", "ESR (Ω)", "Résistance série équivalente (ESR)")); plt.close()
                    pdf.savefig(fig_loglog_simple(f, np.abs(df["ESL_H"].values.astype(float)), "Fréquence (Hz)", "ESL (H)", "Inductance série équivalente (ESL)")); plt.close()
                buf.seek(0); 
                return buf

            pdf_buf = make_pdf(df, meta)
            st.download_button("Télécharger rapport PDF", data=pdf_buf, file_name="rapport_s2p.pdf", mime="application/pdf")

    except Exception as e:
        st.error(f"Erreur : {e}")
else:
    st.info("Charge un fichier .s2p pour commencer.")
