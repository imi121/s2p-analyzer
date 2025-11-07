# pages/01_CV_Accordabilite.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from utils.measure_parsers import parse_measure_file

st.set_page_config(page_title="C(V) — Accordabilité", layout="wide")

st.markdown("""
<div class="hero">
  <h1>C(V) — Accordabilité</h1>
  <p>Lecture 2509RE/CSV multi-fichiers. Accordabilité (%) vs V et Accordabilité/Volt.</p>
</div>
""", unsafe_allow_html=True)
st.write("")

uploads = st.file_uploader("Dépose tes fichiers 2509RE / CSV", type=None, accept_multiple_files=True)
if not uploads:
    st.info("Charge au moins un fichier pour commencer.")
    st.stop()

dfs = []
labels = []
for f in uploads:
    parsed = parse_measure_file(f)
    if parsed["mode"] != "tension":
        st.warning(f"« {parsed['meta']['filename']} » détecté comme mode fréquence — ignoré pour C(V).")
        continue
    df = parsed["df"].copy()
    df["Cap_pF"] = df["Cap_F"] * 1e12
    # C0 ≈ proche de 0 V
    i0 = df["Voltage_V"].abs().idxmin()
    C0 = df.loc[i0, "Cap_pF"]
    df["Tune_%"] = (C0 - df["Cap_pF"]) / C0 * 100.0
    dfs.append(df)
    labels.append(parsed["meta"]["filename"])

if not dfs:
    st.error("Aucun fichier en mode tension lisible.")
    st.stop()

# Courbes Accordabilité vs V
fig, ax = plt.subplots()
for df, lab in zip(dfs, labels):
    ax.plot(df["Voltage_V"], df["Tune_%"], marker='o', label=lab)
ax.set_xlabel("Tension (V)"); ax.set_ylabel("Accordabilité (%)"); ax.set_title("Accordabilité vs Tension")
ax.grid(True); ax.legend()
st.pyplot(fig)

# Accordabilité / Volt : moyenne ± écart-type sur tensions 1…7 V
target_V = st.multiselect("Tensions à évaluer (%/V)", [1,2,3,4,5,6,7], default=[1,2,3,4,5,6,7])
if target_V:
    mean_, std_ = [], []
    for V in target_V:
        vals = []
        for df in dfs:
            idx = (df["Voltage_V"] - V).abs().idxmin()
            vals.append(df.loc[idx, "Tune_%"] / V)
        mean_.append(np.mean(vals)); std_.append(np.std(vals))

    fig2, ax2 = plt.subplots()
    ax2.errorbar(target_V, mean_, yerr=std_, marker='o', capsize=5)
    ax2.set_xlabel("Tension (V)"); ax2.set_ylabel("Accordabilité par Volt (%/V)")
    ax2.set_title("Accordabilité normalisée — Moyenne ± Écart-type")
    ax2.grid(True)
    st.pyplot(fig2)

# Tableau concaténé exportable
big = pd.concat([d.assign(File=lab) for d, lab in zip(dfs, labels)], ignore_index=True)
st.dataframe(big, use_container_width=True)
st.download_button("Exporter CSV", big.to_csv(index=False).encode("utf-8"), "CV_accordabilite.csv", "text/csv")

