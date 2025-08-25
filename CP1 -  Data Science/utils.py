import pandas as pd
import numpy as np
from scipy import stats
from pandas.api.types import CategoricalDtype

def carregar_dados():
    df = pd.read_csv("ai_job_trends_dataset.csv", encoding="utf-8")

    if "Job Status" in df.columns:
        df["Job Status"] = df["Job Status"].fillna("Não Informado")

    for col in df.columns:
        if df[col].dtype == object and df[col].str.contains("%", na=False).any():
            df[col] = df[col].str.replace("%", "").astype(float)

    return df

def _aplicar_ordenacao_ordinal(df: pd.DataFrame) -> pd.DataFrame:

    status_order = ["Declining", "Stable", "Increasing"]
    impact_order = ["Low", "Moderate", "High"]
    edu_order = ["High School", "Associate Degree", "Bachelor’s Degree", "Master’s Degree", "PhD"]

    if "Job Status" in df:
        df["Job Status"] = pd.Categorical(df["Job Status"], categories=status_order, ordered=True)
    if "AI Impact Level" in df:
        df["AI Impact Level"] = pd.Categorical(df["AI Impact Level"], categories=impact_order, ordered=True)
    if "Required Education" in df:
        df["Required Education"] = pd.Categorical(df["Required Education"], categories=edu_order, ordered=True)
    return df

def classificar_variaveis(df: pd.DataFrame) -> pd.DataFrame:
    from pandas.api.types import (
        is_integer_dtype, is_float_dtype
    )
    linhas = []
    for col in df.columns:
        s = df[col]
        
        if isinstance(s.dtype, CategoricalDtype) and s.dtype.ordered:
            tipo = "Qualitativa"
            escala = "Ordinal"

        elif s.dtype == "object" or (isinstance(s.dtype, pd.CategoricalDtype) and not s.dtype.ordered):
            tipo = "Qualitativa"
            escala = "Nominal"
        
        elif is_integer_dtype(s):
            tipo = "Quantitativa"
            escala = "Discreta"
        
        elif is_float_dtype(s):
            tipo = "Quantitativa"
            escala = "Contínua"

        else:
            tipo = "Outro"
            escala = "-"

        linhas.append({
            "Variável": col,
            "Tipo": tipo,
            "Escala": escala,
            "N únicos": int(s.nunique(dropna=True)),
            "% Nulos": round(s.isna().mean()*100, 2),
        })
    return pd.DataFrame(linhas)

def medidas_descritivas(df: pd.DataFrame, colunas_num=None) -> pd.DataFrame:
    if colunas_num is None:
        colunas_num = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    rows = []
    for c in colunas_num:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty: 
            continue
        moda = s.mode()
        rows.append({
            "Variável": c,
            "n": int(s.size),
            "Média": float(s.mean()),
            "Mediana": float(s.median()),
            "Moda": float(moda.iloc[0]) if not moda.empty else np.nan,
            "Mín": float(s.min()),
            "Q1": float(s.quantile(0.25)),
            "Q3": float(s.quantile(0.75)),
            "Máx": float(s.max()),
            "Desvio Padrão": float(s.std(ddof=1)) if s.size > 1 else np.nan,
            "Variância": float(s.var(ddof=1)) if s.size > 1 else np.nan,
        })
    return pd.DataFrame(rows)

def ic_media(s: pd.Series, conf: float = 0.95):
    x = pd.to_numeric(s, errors="coerce").dropna().values
    n = x.size
    if n < 2:
        return (np.nan, np.nan, np.nan)
    media = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - (1-conf)/2, df=n-1)
    return float(media), float(media - tcrit*se), float(media + tcrit*se)

def correlacoes(x: pd.Series, y: pd.Series, conf: float = 0.95):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    df = pd.concat([x, y], axis=1).dropna()
    if df.shape[0] < 3:
        return None

    r_p, p_p = stats.pearsonr(df.iloc[:,0], df.iloc[:,1])
    r_s, p_s = stats.spearmanr(df.iloc[:,0], df.iloc[:,1])

    n = df.shape[0]
    if abs(r_p) >= 1:
        ic_p = (np.nan, np.nan)
    else:
        z = np.arctanh(r_p)
        se = 1/np.sqrt(n-3)
        zcrit = stats.norm.ppf(1 - (1-conf)/2)
        li_z, ls_z = z - zcrit*se, z + zcrit*se
        ic_p = (np.tanh(li_z), np.tanh(ls_z))
    return {
        "n": n,
        "pearson_r": r_p, "pearson_p": p_p, "pearson_ic": ic_p,
        "spearman_rho": r_s, "spearman_p": p_s
    }


def t_test_duas_amostras(s1: pd.Series, s2: pd.Series, alternative="two-sided"):
    x1 = pd.to_numeric(s1, errors="coerce").dropna().values
    x2 = pd.to_numeric(s2, errors="coerce").dropna().values
    if x1.size < 2 or x2.size < 2:
        return None
    tstat, pval = stats.ttest_ind(x1, x2, equal_var=False, alternative=alternative)

    nx, ny = x1.size, x2.size
    sx, sy = x1.std(ddof=1), x2.std(ddof=1)
    s_pooled = np.sqrt(((sx**2)/nx) + ((sy**2)/ny))
    d = (x1.mean() - x2.mean()) / s_pooled if s_pooled != 0 else np.nan
    return {
        "t": float(tstat),
        "p": float(pval),
        "mean1": float(x1.mean()),
        "mean2": float(x2.mean()),
        "cohens_d": float(d)
    }

def anova_oneway_por_industria(df: pd.DataFrame, col="Median Salary (USD)", min_n=100):
    if "Industry" not in df: 
        return None
    grupos = []
    labels = []
    for ind, sub in df.groupby("Industry"):
        vals = pd.to_numeric(sub[col], errors="coerce").dropna()
        if vals.size >= min_n:
            grupos.append(vals.values)
            labels.append(ind)
    if len(grupos) < 2:
        return None
    F, p = stats.f_oneway(*grupos)
    return {"F": float(F), "p": float(p), "labels": labels}

def media_ic_por_grupo(df: pd.DataFrame, group_col: str, value_col: str, conf=0.95) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group_col):
        s = pd.to_numeric(sub[value_col], errors="coerce").dropna()
        n = s.size
        if n < 2:
            continue
        mean = s.mean()
        se = s.std(ddof=1)/np.sqrt(n)
        tcrit = stats.t.ppf(1 - (1-conf)/2, df=n-1)
        ci = tcrit*se
        rows.append({"Grupo": g, "Média": mean, "IC_95": ci, "n": n})
    out = pd.DataFrame(rows).sort_values("Média", ascending=False)
    return out

def crescimento_vagas(df: pd.DataFrame) -> pd.DataFrame:
    req = ["Industry", "Job Openings (2024)", "Projected Openings (2030)"]
    if not all(c in df.columns for c in req):
        return pd.DataFrame()
    agg = df.groupby("Industry", as_index=False)[["Job Openings (2024)", "Projected Openings (2030)"]].sum()
    agg["Crescimento (%)"] = (agg["Projected Openings (2030)"] - agg["Job Openings (2024)"]) / agg["Job Openings (2024)"].replace(0, np.nan) * 100
    agg = agg.dropna(subset=["Crescimento (%)"]).sort_values("Crescimento (%)", ascending=False)
    return agg
