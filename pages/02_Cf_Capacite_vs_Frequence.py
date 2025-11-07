# pages/02_Cf_Capacite_vs_Frequence.py
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from utils.measure_parsers import parse_measure_file

st.set_page_config(page_title="C(f) — Capacité vs Fréquence", layout="wide")

st.markdown("""
<div class="hero">
  <h1>C(f) — Capacité vs Fréquence</h1>
  <p>Lecture 2509RE/CSV multi-fichiers. C(f) (semilog), C@10 kHz vs Vosc, Accordabilité vs Vosc, Accordabilité en fréquence.</p>
</div>
""", unsafe_allow_html=True)
st.write("")

uploads = st.file_uploader("Dépose tes fichiers 2509RE / CSV", type=None, accept_multiple_files=True)
if not uploads:
    st.info("Charge au moins un fichier pour commencer.")
    st.stop()

target_10k = 10_000
target_1k  = 1_000
max_freq   = st.number_input("Fréquence max pour tracés accordabilité(f)", value=1_000_000, step=1000)

cap_points = []    # (Vosc, C@10k pF)
per_file_plots = []

fig, ax = plt.subplots()
for f in uploads:
    parsed = parse_measure_file(f)
    df = parsed["df"].copy()
    if parsed["mode"] != "frequence":
        st.warning(f"« {parsed['meta']['filename']} » détecté comme mode tension — ignoré pour C(f).")
        continue

    df["Cap_pF"] = df["Cap_F"] * 1e12
    # C(f) semilog
    ax.semilogx(df["Freq_Hz"], df["Cap_pF"], marker='o', label=parsed["meta"]["filename"])

    # C@10kHz
    i10 = (df["Freq_Hz"] - target_10k).abs().idxmin()
    cap10 = float(df.loc[i10, "Cap_pF"])

    # Vosc depuis fichier ou nom
    vosc = parsed["meta"]["Vosc_file"]
    if vosc is None:
        m = re.search(r"_F\(var\)_(.*)V", parsed["meta"]["filename"])
        if m:
            try: vosc = float(m.group(1).replace(",", "."))
            except: vosc = None
    if vosc is None:
        st.info(f"Vosc non trouvé dans le nom de « {parsed['meta']['filename']} » — la courbe C@10kHz vs Vosc ignorera ce fichier.")
    else:
        cap_points.append((vosc, cap10))

ax.set_xlabel("Fréquence (Hz)"); ax.set_ylabel("Capacité (pF)"); ax.set_title("Capacité en fonction de la fréquence")
ax.grid(True, which="both")
if len(ax.lines)>0: ax.legend()
st.pyplot(fig)

# Tableau C@10kHz vs Vosc + Accordabilité vs Vosc
if cap_points:
    data = pd.DataFrame(cap_points, columns=["Vosc", "Cap10k_pF"]).sort_values("Vosc")
    # C0 = référence @ 0.4 V si dispo, sinon @ Vosc min
    if 0.4 in data["Vosc"].values:
        C0 = data.loc[data["Vosc"]==0.4, "Cap10k_pF"].values[0]
    else:
        C0 = data.iloc[0]["Cap10k_pF"]
        st.info(f"Référence 0.4 V absente → C0 pris à Vosc={data.iloc[0]['Vosc']} V.")

    data["Tune_%"] = (C0 - data["Cap10k_pF"])/C0*100.0

    fig2, ax2 = plt.subplots()
    ax2.plot(data["Vosc"], data["Tune_%"], marker='o')
    ax2.set_xlabel("Tension d'oscillation (V)"); ax2.set_ylabel("Accordabilité (%)")
    ax2.set_title("Accordabilité à 10 kHz vs Vosc"); ax2.grid(True)
    st.pyplot(fig2)

    st.dataframe(data, use_container_width=True)
    st.download_button("Exporter CSV (C@10kHz & Tune)", data.to_csv(index=False).encode("utf-8"),
                       "Cf_10kHz_Vosc.csv", "text/csv")

# Accordabilité en fréquence (réf = C@1kHz par fichier)
fig3, ax3 = plt.subplots()
for f in uploads:
    parsed = parse_measure_file(f)
    if parsed["mode"] != "frequence":
        continue
    df = parsed["df"].copy()
    df["Cap_pF"] = df["Cap_F"]*1e12

    i1k = (df["Freq_Hz"] - target_1k).abs().idxmin()
    C1k = float(df.loc[i1k, "Cap_pF"])
    df["TuneFreq_%"] = (C1k - df["Cap_pF"])/C1k*100.0

    df_plot = df[(df["Freq_Hz"]>=1_000) & (df["Freq_Hz"]<=max_freq)].copy()
    # filtre de pente simple (anti-artefacts)
    vals = df_plot["TuneFreq_%"].values
    keep = [True] + [abs(vals[i]-vals[i-1])<5 for i in range(1, len(vals))]
    df_plot = df_plot[np.array(keep, bool)]

    # Label avec Vosc si dispo
    lab = parsed["meta"]["filename"]
    if parsed["meta"]["Vosc_file"] is not None:
        lab = f"{parsed['meta']['Vosc_file']} V"

    ax3.semilogx(df_plot["Freq_Hz"], df_plot["TuneFreq_%"], marker='o', label=lab)

ax3.set_xlabel("Fréquence (Hz)"); ax3.set_ylabel("Accordabilité (%)")
ax3.set_title("Accordabilité vs Fréquence — Réf = C(1 kHz) du fichier")
ax3.grid(True, which="both")
if len(ax3.lines)>0: ax3.legend(title="Vosc")
st.pyplot(fig3)

