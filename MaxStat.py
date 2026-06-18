# -------------------------------------------------
#  MaxStat.py – core statistical utilities
# -------------------------------------------------
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

from scipy.stats import (
    fisher_exact,
    chi2_contingency,
    mannwhitneyu,
    MonteCarloMethod,
)

from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import Table2x2

import seaborn as sns
import matplotlib.pyplot as plt


## Helpers


def format_p_value(p: float) -> str:
    try:
        """Return a compact, journal‑style string for a p‑value."""
        if pd.isna(p):
            return "nan"
        if p < 0.001:
            return "< .001"
        if 0.045 <= p < 0.050:
            return f"{p:.3f}".lstrip('0')
        return f"{p:.2f}".lstrip('0')
    except:
        return p


def phi_coefficient(a: int, b: int, c: int, d: int) -> float:
    """Pearson’s φ for a 2 × 2 contingency table."""
    num = a * d - b * c
    den = ((a + b) * (c + d) * (a + c) * (b + d)) ** 0.5
    return num / den if den != 0 else 0.0


def _phi_ci_normal(phi: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Normal‑approximation CI for φ (Cohen 1988)."""
    if n <= 2:
        return np.nan, np.nan
    se = ((1 - phi ** 2) / (n - 2)) ** 0.5
    z = 1.96 if alpha == 0.05 else scipy.stats.norm.ppf(1 - alpha / 2)
    low = max(phi - z * se, -1.0)
    high = min(phi + z * se, 1.0)
    return low, high


def _or_ci_woolf(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact CI for the odds‑ratio (Haldane‑Anscombe correction)."""
    tbl = Table2x2([[a, b], [c, d]], shift_zeros=True)
    return tbl.oddsratio_confint(alpha=alpha)


def _fdr_adjust(p_series: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini–Hochberg FDR correction."""
    adj = multipletests(p_series, method="fdr_bh", alpha=alpha)
    return pd.DataFrame({"p_adj": adj[1], "significant": adj[0]}, index=p_series.index)


def _standardized_residuals(tbl: pd.DataFrame) -> pd.Series:
    """Return z‑scores (standardized residuals) for the “1” column of a 2‑col table."""
    chi2, _, _, exp = chi2_contingency(tbl.values, correction=False)
    # obs for the “1” column; if missing, treat as zeros
    obs_one = tbl.get(1, pd.Series(0, index=tbl.index)).values
    exp_one = exp[:, 1]
    return (obs_one - exp_one) / np.sqrt(exp_one)


def _cramers_v(chi2: float, n: int, k: int) -> float:
    """Cramér’s V for a χ² test (k = min(rows, cols))."""
    return np.sqrt(chi2 / (n * (k - 1))) if k > 1 else np.nan


def _chi2_subtable_phi(a: int, b: int, c: int, d: int) -> float:
    """φ for a 2 × 2 sub‑table derived from a larger χ² table."""
    chi2, _, _, _ = chi2_contingency([[a, b], [c, d]], correction=False)
    return np.sqrt(chi2 / (a + b + c + d))

def mwu_effect(x, y):
    """Return (U, p, Rosenthal‑r)."""
    u, p = mannwhitneyu(x.dropna(), y.dropna(), alternative="two-sided")
    n0, n1 = len(y), len(x)
    # Rosenthal’s r (standardized)
    r = (u - n0*n1/2) / np.sqrt(n0*n1*(n0+n1+1)/12) / np.sqrt(n0+n1)
    return u, p, r


    

### Analysis

def pairwise_mwu_ranking(
    df: pd.DataFrame,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> pd.Series:
    """
    Rank columns by the number of pair‑wise Mann‑Whitney U “wins”.
    A win = column i has a *significantly* larger median than column j.
    Returns a Series of ranks (1 = most wins).
    """
    cols = df.columns
    n = len(cols)

    wins = pd.DataFrame(0, index=cols, columns=cols, dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            x, y = df[cols[i]].dropna(), df[cols[j]].dropna()
            u, p = mannwhitneyu(x, y, alternative=alternative)

            if p < alpha:
                if x.median() > y.median():
                    wins.at[cols[i], cols[j]] = 1
                else:
                    wins.at[cols[j], cols[i]] = 1

    total_wins = wins.sum(axis=1)
    return total_wins.rank(method="dense", ascending=False).astype(int)


def country_code_association(
    df_bin: pd.DataFrame,
    country_col: str,
    code_cols: List[str],
    alpha: float = 0.05,
    add_ci: bool = True,
) -> pd.DataFrame:
    """
    For every (country, binary code) pair compute:

        • a,b,c,d  – raw cell counts
        • φ, odds‑ratio, raw p‑value (Fisher exact)
        • optional 95 % CIs for φ and OR
        • BH‑adjusted p‑value and a ``significant`` flag

    Returns a *long* DataFrame ready for pivoting or reporting.
    """
    records = []
    countries = df_bin[country_col].unique()

    for code in code_cols:
        # ----- global totals for this code (ignore NAs) -----
        global_one  = df_bin[code].sum(skipna=True)
        global_zero = df_bin[code].count() - global_one

        for ct in countries:
            sub = df_bin[df_bin[country_col] == ct]
            sub = sub[sub[code].notna()]

            a = sub[code].sum()
            b = len(sub) - a
            c = global_one  - a
            d = global_zero - b

            odds, p_raw = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            phi = phi_coefficient(a, b, c, d)

            row = {
                "country": ct,
                "code": code,
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "phi": phi,
                "odds": odds,
                "p_raw": p_raw,
            }

            if add_ci:
                row["phi_ci_low"], row["phi_ci_high"] = _phi_ci_normal(phi, a + b + c + d)
                row["or_ci_low"],  row["or_ci_high"]  = _or_ci_woolf(a, b, c, d)

            records.append(row)

    out = pd.DataFrame(records)
    out = out.join(_fdr_adjust(out["p_raw"], alpha=alpha))
    return out

def topic_country_association(
    df: pd.DataFrame,
    topic_col: List[str],
    country_col: str = "country",
    resid_thr: float = 1.96,
    phi_thr: float = 0.1,
    n_resamples: int = 20_000,
    alpha: float = 0.05) -> pd.DataFrame:
    
    """
    Monte‑Carlo χ² + Cramér’s V for each binary column in ``topic_col`` vs.
    ``country_col``.  Columns that are all‑0 or all‑1 are automatically
    handled with a Fisher exact test (exact φ, odds‑ratio, p‑value).
    """

    from scipy.stats import chi2_contingency, fisher_exact, MonteCarloMethod

    mc = MonteCarloMethod(n_resamples=n_resamples)
    rows = []

    for col in topic_col:
        tbl = pd.crosstab(df[country_col], df[col])

        try: 
            chi2, p_raw, _, exp = chi2_contingency(tbl.values,correction=False, method=mc)
           # chi2, p_raw, _, exp = chi2_contingency(tbl.values)
            
            n = exp.sum()
            k = min(tbl.shape)
            
            cramers_v = _cramers_v(chi2, n, k)
            test_used = "χ² (MC)"
            
        except ValueError:               # zero‑margin problem → Fisher exact
            

            # collapse everything into a single 2×2 table:
            a = tbl[1].sum() if 1 in tbl.columns else 0   # total “1”
            b = tbl[0].sum() if 0 in tbl.columns else 0   # total “0”
            # Fisher exact on the aggregated table
            odds, p_raw = fisher_exact([[a, b], [0, 0]], alternative="two-sided")
            # φ for a 2×2 table with one margin = 0 is simply 0
            phi = 0.0
            cramers_v = np.nan
            test_used = "Fisher (fallback)"
            # we still want to report a row, so fill the standard fields
            rows.append(
                {
                    "code": col,
                    "test": test_used,
                    "χ²": np.nan,
                    "p_raw": p_raw,
                    #"p_formatted": format_p_value(p_raw),
                    "CramérsV": cramers_v,
                    "over_countries [φ]": "",
                    "under_countries [φ]": "",
                }
            )
            
            continue   # skip the rest of the loop – no residuals to compute
        
        resid = _standardized_residuals(tbl)

        over, under = [], []
        
        pos_col = True if True in tbl.columns else tbl.columns[1]
        neg_col = False if False in tbl.columns else tbl.columns[0]
        
        for ct, r in zip(tbl.index, resid):
            
            if abs(r) <= resid_thr:
                continue

        
            o1 = tbl.at[ct, pos_col] if pos_col in tbl.columns else 0
            o0 = tbl.at[ct, neg_col] if neg_col in tbl.columns else 0
            other1 = tbl[pos_col].sum() - o1
            other0 = tbl[neg_col].sum() - o0
        
            phi = _chi2_subtable_phi(o1, o0, other1, other0)
            phi_signed = phi if r > 0 else -phi
            entry = f"{ct} ({phi_signed:.2f})"

            if abs(phi) <= phi_thr:
                continue
            
            if r > 0:
                over.append(entry)
            else:
                under.append(entry)

        rows.append(
            {
                "code": col,
                "test": test_used,
                "χ²": chi2.round(2),
                "p_raw": p_raw,
                "p_formatted": format_p_value(p_raw),
                "CramérsV": cramers_v.round(2),
                "over_countries [φ]": ", ".join(over),
                "under_countries [φ]": ", ".join(under),
            }
        )

    out = pd.DataFrame(rows).set_index("code")
    out = out.join(_fdr_adjust(out["p_raw"], alpha=alpha))
    out["significant"] = out["p_adj"] < alpha
    return out

def mwu_group_vs_rest(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    direction_labels: Tuple[str, str] = ("Higher", "Lower"),
    alpha: float = 0.05,
) -> Dict[Any, str]:
    """
    Two‑sided Mann‑Whitney U test of *value_col* inside each group
    (e.g. each topic) against the rest of the sample.

    Returns a dict {group → formatted result string}.  The string
    contains the direction of the effect and the raw p‑value; a
    post‑hoc BH correction is applied.
    """
    raw = {}
    for g, sub in df.groupby(group_col):
        rest = df[df[group_col] != g]
        x, y = sub[value_col].dropna(), rest[value_col].dropna()
        if len(x) == 0 or len(y) == 0:
            continue
        u, p, r = mwu_effect(x,y)

        direction = direction_labels[0] if x.mean() > y.mean() else direction_labels[1]
        raw[g] = (direction, p,r)

    # BH correction across groups
    if raw:
        p_series = pd.Series({g: v[1] for g, v in raw.items()})
        stat_series = pd.Series({g: v[2] for g, v in raw.items()})
        adj = _fdr_adjust(p_series, alpha=alpha)
        out = {}
        for g, (direction, p_raw,stat_raw) in raw.items():
            sig = adj.loc[g, "significant"]
            p_str = f"p={p_raw:.3f}" if sig else f"p={p_raw:.2f}"
            if sig:
                out[g] = f"{direction} {value_col.split('_')[0]} (r={stat_raw:.2f}, {p_str})"
            else:
                out[g]= ""
        return out
    return {}


## Plot and pivot

def pivot_matrices(df_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return φ‑, odds‑ratio‑ and significance‑matrices for heat‑maps."""
    phi_mat = df_long.pivot(index="country", columns="code", values="phi")
    odds_mat = df_long.pivot(index="country", columns="code", values="odds")
    sig_mat = df_long.pivot(index="country", columns="code", values="significant")
    return phi_mat, odds_mat, sig_mat


def plot_heatmap(
    phi_mat: pd.DataFrame,
    odds_mat: pd.DataFrame,
    sig_mat: pd.DataFrame,
    circle_scale: float = 250,
    cmap: str = "RdBu_r",
) -> plt.Axes:
    """Classic heat‑map (countries = x, codes = y) with odds‑ratio circles."""
    plt.figure(
        figsize=(max(8, phi_mat.shape[1] * 0.6), max(6, phi_mat.shape[0] * 0.5))
    )
    ax = sns.heatmap(
        phi_mat,
        cmap=cmap,
        center=0,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Φ‑coefficient"},
        square=False,
    )
    for i, ct in enumerate(phi_mat.index):
        for j, code in enumerate(phi_mat.columns):
            odds = odds_mat.at[ct, code]
            if np.isfinite(odds):
                size = min(np.log10(odds + 1) * circle_scale, 2000)
                ax.scatter(j + 0.5, i + 0.5, s=size,
                           facecolors="none", edgecolors="black", lw=0.8)
                if sig_mat.at[ct, code]:
                    ax.text(j + 0.5, i + 0.5, "*",
                            ha="center", va="center", color="black", fontsize=12)
    ax.set_xlabel("LLM‑coded reason")
    ax.set_ylabel("Country")
    ax.set_title(
        "Country dependence of open‑text reasons\n"
        "Red = over‑represented (Φ > 0), Blue = under‑represented (Φ < 0)\n"
        "Circle size ∝ odds‑ratio; * = FDR‑significant (p < 0.05)"
    )
    plt.tight_layout()
    return ax


def vizualize_topics(topic_model,sub=pd.DataFrame(),summaries={},label_font_size=12,figsize=(4.5, 3.5),filename="topics_publication_EU Ext shocks.pdf"):


    short_labels = {
        k: v.split(":")[0].strip().strip("*").replace("‑","-").title()
        for k, v in summaries.items()
    }
    
    docs=sub["QID8"].to_list()
    topic_model.set_topic_labels(short_labels)

    fig = topic_model.visualize_document_datamap(
        docs,
       # reduced_embeddings=reduced_embeddings,
        custom_labels=short_labels,
        interactive=False,
        datamap_kwds={"label_font_size": label_font_size}
    )

    fig.set_size_inches(figsize)


    # --- remove EVERYTHING visual except points ---
    for ax in fig.get_axes():
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
    
        # remove frame/spines
        for spine in ax.spines.values():
            spine.set_visible(False)

    # remove any figure-level title (important!)
    fig.suptitle("")
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    
    return