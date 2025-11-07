# utils/measure_parsers.py
import io, re, csv
import numpy as np
import pandas as pd

HEADER_MAP = {
    # variations possibles
    "freq": ["freq", "frequency", "freq (hz)", "f (hz)"],
    "voltage": ["dcb (v)", "bias (v)", "v", "voltage", "volt", "vdc"],
    "cap": ["cp (f)", "cap (f)", "c (f)", "capacitance (f)", "c"],
    "rp": ["rp (ohms)", "rp", "rpar", "r (ohms)"],
}

def _normalize_headers(cols):
    norm = []
    for c in cols:
        c0 = re.sub(r"\s+", " ", str(c or "")).strip().lower()
        norm.append(c0)
    return norm

def _guess_sep(sample: str):
    # ordre: tab, ;, , , espaces multiples
    if "\t" in sample: return "\t"
    if ";" in sample:  return ";"
    if "," in sample:  return ","
    # fallback: split sur espaces
    return r"\s+"

def _read_text_to_df(file_bytes: bytes):
    # saute les lignes commentaires débutant par "!"
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("!")]
    if not lines:
        raise ValueError("Fichier vide après retrait des commentaires.")
    # détecte séparateur
    sep = _guess_sep("\n".join(lines[:10]))
    # si la 1ère ligne ressemble à un header → header=0, sinon header=None
    header_line = lines[0].lower()
    has_header = any(k in header_line for k in ["freq", "hz", "cp", "rp", "v", "ohms"])
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, engine="python",
                     header=0 if has_header else None)
    return df

def _coerce_numeric(df: pd.DataFrame):
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def _match_col(norm_cols, keys):
    for i, c in enumerate(norm_cols):
        for k in keys:
            if c == k or c.startswith(k.split()[0]):  # tolère "cp (f)" ~ "cp"
                return i
    return None

def parse_measure_file(uploaded_file):
    """
    Retourne dict:
      {
        'mode': 'tension' | 'frequence',
        'df': DataFrame avec colonnes normalisées:
              - tension: ["Freq_Hz", "Voltage_V", "Cap_F", "Rp_Ohm"]
              - frequence: au minimum ["Freq_Hz","Cap_F"] (+ "Voltage_V" si présent)
        'meta': {...}
      }
    Supporte fichiers 2509RE, Agilent 4294A bruts, ou CSV classiques.
    """
    # lit tous les octets
    if hasattr(uploaded_file, "getvalue"):
        raw = uploaded_file.getvalue()
        name = uploaded_file.name
    else:
        raw = uploaded_file.read()
        name = getattr(uploaded_file, "name", "file")

    df = _read_text_to_df(raw)
    df = _coerce_numeric(df)

    # essaie de reconnaître automatiquement les colonnes
    if df.shape[1] >= 3:
        # header connu ?
        norm_cols = _normalize_headers(df.columns)
        iF = _match_col(norm_cols, HEADER_MAP["freq"]) if any(isinstance(x,str) for x in df.columns) else 0
        iV = _match_col(norm_cols, HEADER_MAP["voltage"])
        iC = _match_col(norm_cols, HEADER_MAP["cap"])
        iR = _match_col(norm_cols, HEADER_MAP["rp"])

        # fallback si pas d'entêtes
        if iF is None: iF = 0
        if iC is None:
            # cherche colonne avec grandeur ~1e-9..1e-14 (typique F)
            magnitudes = [float(df.iloc[:50, j].astype(float).abs().median()) if pd.api.types.is_numeric_dtype(df.iloc[:,j]) else np.nan
                          for j in range(df.shape[1])]
            iC = int(np.nanargmin(np.abs(np.log10(np.maximum(magnitudes, 1e-30)) - (-11))))  # proche du pF

        # si présence d'une vraie colonne de tension → mode "tension"
        mode = "tension" if (iV is not None) else "frequence"

        if mode == "tension":
            cols = {
                "Freq_Hz": df.iloc[:, iF].values,
                "Voltage_V": df.iloc[:, iV].values if iV is not None else np.nan,
                "Cap_F": df.iloc[:, iC].values,
                "Rp_Ohm": df.iloc[:, iR].values if (iR is not None) else np.nan
            }
            out = pd.DataFrame(cols)
        else:
            # au minimum F, C — tension éventuellement absente
            cols = {
                "Freq_Hz": df.iloc[:, iF].values,
                "Cap_F": df.iloc[:, iC].values
            }
            if iV is not None:
                cols["Voltage_V"] = df.iloc[:, iV].values
            out = pd.DataFrame(cols)

    elif df.shape[1] == 2:
        out = pd.DataFrame({"Freq_Hz": df.iloc[:,0].values, "Cap_F": df.iloc[:,1].values})
        mode = "frequence"
    else:
        raise ValueError("Format non reconnu (colonnes insuffisantes).")

    # extraction Vosc ~ depuis le nom de fichier si présent
    m = re.search(r"([0-9]+(?:[._][0-9]+)?)\s*V", name.replace(",", "."))
    vosc = float(m.group(1)) if m else None

    return {
        "mode": mode,
        "df": out,
        "meta": {"filename": name, "Vosc_file": vosc}
    }
