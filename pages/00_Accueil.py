# pages/00_Accueil.py
import streamlit as st

st.set_page_config(page_title="Accueil — Mesures MIM", layout="wide")

st.markdown("""
<style>
.hero{
  background: linear-gradient(135deg, rgba(63,63,214,0.7) 0%, rgba(25,28,56,0.9) 100%);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 20px; padding: 28px 32px; color: #fff;
  box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}
.grid {display:grid; grid-template-columns: repeat(3, 1fr); gap:18px; margin-top:18px;}
.card {
  background: rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12);
  border-radius: 16px; padding: 18px 20px; color:#fff;
}
.card h3{margin:.2rem 0 .6rem; font-weight:800}
.card p{opacity:.9; font-size:.95rem}
.btn{
  display:inline-block; margin-top:.6rem; background: linear-gradient(135deg,#4f46e5,#06b6d4);
  padding:9px 14px; border-radius:10px; color:#fff; text-decoration:none; font-weight:600;
}
</style>
<div class="hero">
  <h1>Mesures MIM — Accueil</h1>
  <p>Choisissez un module : Haute fréquence (.s2p), Accordabilité C(V), ou Capacité C(f).</p>
</div>
""", unsafe_allow_html=True)

st.write("")
st.markdown('<div class="grid">', unsafe_allow_html=True)

# Carte 1 : HF .s2p (lien vers app.py)
st.markdown("""
<div class="card">
  <h3>Haute fréquence (.s2p)</h3>
  <p>Extraction Cp / tanδ / ESR / ESL avec SRF, lissage et rapport PDF.</p>
  <a class="btn" href="./app" target="_self">Ouvrir</a>
</div>
""", unsafe_allow_html=True)

# Carte 2 : C(V)
st.markdown("""
<div class="card">
  <h3>C(V) — Accordabilité</h3>
  <p>Accordabilité (%) vs V, Accordabilité/Volt (moyenne ± écart-type). Lecture 2509RE.</p>
  <a class="btn" href="./C(V)_Accordabilite" target="_self">Ouvrir</a>
</div>
""", unsafe_allow_html=True)

# Carte 3 : C(f)
st.markdown("""
<div class="card">
  <h3>C(f) — Capacité vs Fréquence</h3>
  <p>Courbes C(f), C@10kHz vs Vosc, Accordabilité vs Vosc et Accordabilité en fréquence.</p>
  <a class="btn" href="./Cf_Capacite_vs_Frequence" target="_self">Ouvrir</a>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
