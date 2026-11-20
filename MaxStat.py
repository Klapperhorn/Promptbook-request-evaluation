"""Statistics helpers: group comparisons (Mann-Whitney U, Kruskal-Wallis), associations with FDR correction,
correlation matrices, heatmaps and table styling.

Single maintained version: Git_Sync/_shared_modules (merged 2026-11-06; group-comparison and correlation
functions moved here from MR_Qualtrics_module).
"""
from __future__ import annotations

import scipy

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu, spearmanr, t
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import Table2x2


# ------------------------------------------------------------------ association and effect sizes


def phi_coefficient(a: int, b: int, c: int, d: int) -> float:
    """Pearson’s φ for a 2 × 2 contingency table."""
    num = a * d - b * c
    den = ((a + b) * (c + d) * (a + c) * (b + d)) ** 0.5
    return num / den if den != 0 else 0.0


def _phi_ci_normal(phi: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Normal-approximation CI for φ (Cohen 1988)."""
    if n <= 2:
        return np.nan, np.nan
    se = ((1 - phi ** 2) / (n - 2)) ** 0.5
    z = 1.96 if alpha == 0.05 else scipy.stats.norm.ppf(1 - alpha / 2)
    low = max(phi - z * se, -1.0)
    high = min(phi + z * se, 1.0)
    return low, high


def _or_ci_woolf(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact CI for the odds-ratio (Haldane-Anscombe correction)."""
    tbl = Table2x2([[a, b], [c, d]], shift_zeros=True)
    return tbl.oddsratio_confint(alpha=alpha)


def _fdr_adjust(p_series: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini–Hochberg FDR correction."""
    adj = multipletests(p_series, method="fdr_bh", alpha=alpha)
    return pd.DataFrame({"p_adj": adj[1], "significant": adj[0]}, index=p_series.index)


def mwu_effect(x, y):
    """Return (U, p, Rosenthal-r)."""
    u, p = mannwhitneyu(x.dropna(), y.dropna(), alternative="two-sided")
    n0, n1 = len(y), len(x)
    # Rosenthal’s r (standardized)
    r = (u - n0*n1/2) / np.sqrt(n0*n1*(n0+n1+1)/12) / np.sqrt(n0+n1)
    return u, p, r


def MWU_ranking_min(df, pval_limit=0.05):
    """Minimal MWU ranking: returns mean, std, ci, n, rank, sig_to_prev, p, effect_size_r, MWU_rank."""
    data = []
    for c in df.columns:
        nums = df[c].dropna()
        if len(nums) == 0:
            continue
        mean, std, n = nums.mean(), nums.std(), len(nums)
        se = std / np.sqrt(n) if n > 0 else 0
        ci = t.interval(0.95, n-1, loc=mean, scale=se) if n > 1 else (mean, mean)
        data.append({
            'item': c,
            'mean': mean,
            'std': std,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'n': n,
            'nums': nums
        })

    df_s = pd.DataFrame(data).sort_values('mean', ascending=False).reset_index(drop=True)
    df_s['rank'] = df_s['mean'].rank(ascending=False, method='dense')

    sigs, pvalues, effects = [False], [None], [None]
    for i in range(1, len(df_s)):
        nums_i, nums_prev = df_s.loc[i, 'nums'], df_s.loc[i-1, 'nums']
        stat, p = mannwhitneyu(nums_i, nums_prev, alternative="two-sided")
        n1, n2 = len(nums_i), len(nums_prev)
        r = 1 - (2 * stat) / (n1 * n2)
        pvalues.append(p)
        sigs.append(p < pval_limit)
        effects.append(round(abs(r), 3))

    df_s['sig_to_prev'] = sigs
    df_s['p'] = pvalues
    df_s['effect_size_r'] = effects

    df_s['MWU_rank'] = 1
    for i in range(1, len(df_s)):
        df_s.loc[i, 'MWU_rank'] = df_s.loc[i-1, 'MWU_rank'] + 1 if sigs[i] else df_s.loc[i-1, 'MWU_rank']

    return df_s.drop(columns='nums').set_index('item')


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
        • φ, odds-ratio, raw p-value (Fisher exact)
        • optional 95 % CIs for φ and OR
        • BH-adjusted p-value and a ``significant`` flag

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


def pivot_matrices(df_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return φ-, odds-ratio- and significance-matrices for heat-maps."""
    phi_mat = df_long.pivot(index="country", columns="code", values="phi")
    odds_mat = df_long.pivot(index="country", columns="code", values="odds")
    sig_mat = df_long.pivot(index="country", columns="code", values="significant")
    return phi_mat, odds_mat, sig_mat


def plot_heatmap(
    phi_mat,
    sig_mat,
    cmap="RdBu_r",
    vlim=None,
):
    """Nature-style heatmap of country-attribution associations."""

    if vlim is None:
        vlim = np.nanmax(np.abs(phi_mat.values))

    plt.figure(
        figsize=(max(8, phi_mat.shape[1] * 0.6),
                 max(6, phi_mat.shape[0] * 0.5))
    )

    ax = sns.heatmap(
        phi_mat,
        cmap=cmap,
        center=0,
        vmin=-vlim,
        vmax=vlim,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Phi coefficient (Φ)"}
    )

    # significance markers
    for i in range(phi_mat.shape[0]):
        for j in range(phi_mat.shape[1]):
            if sig_mat.iloc[i, j]:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "•",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                    fontweight="bold"
                )

    ax.set_xlabel("Country")
    ax.set_ylabel("Causal attribution")

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    return ax


# ------------------------------------------------------------------ group comparisons and correlations (from MR_Qualtrics_module)


report_style = {
    'formatter': {
        'mean': '{:.2f}',
        'std': '{:.2f}',
        'p': '{:.3f}'
    },
    'precision': 0,  # Changed to 2; '0' was overriding your column formatters
    'na_rep': '',    # Empty for missing values in LaTeX
    'escape': 'latex',  # Auto-escapes special chars like %, $, ^ for LaTeX
}


def style_gradient(df, width=None,precision=3, max_val=None, min_val=None):
    if max_val==None:
        max_val = np.nanmax(np.abs(df.values))
    if min_val==None:
        min_val=-max_val
    styled = (df.style.background_gradient(
                  cmap="RdBu",
                  axis=None,
                  vmin=min_val,
                  vmax=max_val)
              .format(precision=precision))
    
    if width is not None:
        styled = styled.set_table_styles([{
                  'selector': 'th, td',
                  'props': [('min-width', width), ('max-width', width), ('width', width)]
              }])
    return styled


def MWU_ranking(df, pval_limit=0.05):

    data = []
    for c in df.columns:
        nums = df[c].dropna()
        mean, std, n = nums.mean(), nums.std(), len(nums)
        mi, ma = nums.min(), nums.max()
        se = std / np.sqrt(n) if n > 0 else 0
        ci = stats.t.interval(0.95, n-1, loc=mean, scale=se) if n > 1 else (mean, mean)

        data.append({
            'item': c,
            'mean': mean,
            'std': std,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'n': n,
            'nums': nums,
            'range': (mi, ma)
        })

    df_s = (
        pd.DataFrame(data)
        .sort_values('mean', ascending=False)
        .reset_index(drop=True)
    )

    # classical descriptive rank
    df_s['rank'] = df_s['mean'].rank(ascending=False, method='dense')

    sigs = [False]
    pvalues = [None]
    effects = [None]

    for i in range(1, len(df_s)):
        nums_i = df_s.loc[i, 'nums']
        nums_prev = df_s.loc[i-1, 'nums']

        stat, p = mannwhitneyu(nums_i, nums_prev, alternative="two-sided")

        n1, n2 = len(nums_i), len(nums_prev)
        # rank-biserial correlation
        r = 1 - (2 * stat) / (n1 * n2)

        pvalues.append(p)
        sigs.append(p < pval_limit)
        effects.append(round(abs(r), 3))

    df_s['sig_to_prev'] = sigs
    df_s['p'] = pvalues
    df_s['effect_size_r'] = effects

    def interpret_jump(p, r):
        if p is None or r is None:
            return ""
        if p < 0.05:
            if r >= 0.50: return "p < 0.05 & r >= 0.50: Large jump"
            elif r >= 0.30: return "p < 0.05 & 0.30 ≤ r < 0.50: Medium jump"
            else: return "p < 0.05 & r < 0.30: Small jump"
        elif p < 0.10:
            if r >= 0.50: return "p < 0.10 & r >= 0.50: Large trend"
            elif r >= 0.30: return "p < 0.10 & 0.30 ≤ r < 0.50: Medium trend"
            else: return "p < 0.10 & r < 0.30: Weak trend"
        else:
            return f"p = {round(p,2)}: No difference"

    df_s['interpretation'] = [
        interpret_jump(p, r) for p, r in zip(pvalues, effects)
    ]

    # MWU-based stepwise ranking (grouping)
    df_s['MWU_rank'] = 1
    for i in range(1, len(df_s)):
        if df_s.loc[i, 'sig_to_prev']:
            df_s.loc[i, 'MWU_rank'] = df_s.loc[i-1, 'MWU_rank'] + 1
        else:
            df_s.loc[i, 'MWU_rank'] = df_s.loc[i-1, 'MWU_rank']

    return df_s.drop(columns='nums').set_index('item')


def kruskal_groups(df,dependent_columns,indepentent_column, test="kruskal"):
    print(test,":\n")
    d = df.loc[2:, dependent_columns + [indepentent_column]].dropna()

    # Convert any text-like numeric values to proper floats
    for col in dependent_columns:
        d[col] = pd.to_numeric(d[col], errors='coerce')

    if test=="kruskal":
        result={}
        from scipy.stats import kruskal
        for col in dependent_columns:
            stat, p= kruskal(*[d.loc[d[indepentent_column]==g, col] for g in d[indepentent_column].unique()])
            print(f"{col}: U={stat:.3f}, p={p:.4f}")
            result[col]=(round(stat,3),round(p,4))
        

    if test=="mannwhitneyu":
        
        from scipy.stats import mannwhitneyu
        # Run Mann–Whitney U tests per trilemma variable
        groups = d['Salience Group'].unique()
        result={}
        for col in dependent_columns:
            x = d.loc[d['Salience Group'] == groups[0], col].dropna()
            y = d.loc[d['Salience Group'] == groups[1], col].dropna()
            stat, p = mannwhitneyu(x, y, alternative='two-sided')
            print(f"{col}: U={stat:.3f}, p={p:.4f}")
            result[col]=(round(stat,3),round(p,4))
    return  result


def kruskal_dependence_matrix(df, dep_vars, indep_vars,limit=0.05):
    from MR_Qualtrics_module import clean_text  # Qualtrics helpers, imported here so MaxStat also works without that module
    from scipy.stats import kruskal

    """
    Perform Kruskal-Wallis tests of dependence of ordinal dep_vars 
    on nominal indep_vars.

    Parameters:
        df: input DataFrame with nominal indep_vars and ordinal dep_vars as strings with numbers in parentheses.
        dep_vars: list of dependent variable column names (ordinal).
        indep_vars: list of independent variable column names (nominal).

    Returns:
        pval_df: DataFrame indexed by dep_vars, columns are indep_vars, values are p-values.
    """

    # Initialize empty DataFrame to store p-values
    pval_df = pd.DataFrame(index=dep_vars, columns=[f"{indep_var}_p" for indep_var in indep_vars], dtype=float)
    
    cleaned_indexnames = df.loc[0, dep_vars].apply(clean_text).str.split("- ").str[-1]
    question_names = df.loc[0, dep_vars].apply(clean_text).str.split("- ").str[0].str.split("(").str[0]
    index=pd.MultiIndex.from_tuples(list(zip(question_names,cleaned_indexnames)), names=["question", "item"])
    df_data = df.loc[2:].copy() 
    for indep_var in indep_vars:
        for dep_var in dep_vars:
            # Extract numeric values from dep_var strings

            col_name=dep_var + '_num'
            df_clean = df_data[[indep_var, dep_var]].dropna()
            df_clean[col_name] = pd.to_numeric(df_clean[dep_var], errors='coerce') # columns with numericals
            df_clean[col_name] = df_clean[col_name].fillna(df_clean[dep_var].astype(str).str.extract(r'^\((\d+)\)').astype(float)[0]) # columns with strings

            
            # Get unique groups for independent variable
            groups = df_clean[indep_var].unique()

            # Prepare samples per group to test
            samples = [df_clean.loc[df_clean[indep_var] == g, col_name].dropna().values for g in groups]
            if len(samples) >= 2 and all(len(s) > 0 for s in samples):
                try:
                    stat, p = kruskal(*samples)
                    pval_df.at[dep_var, f"{indep_var}_p"] = p
                    pval_df.at[dep_var,  f"{indep_var}_stat"] = stat
                except ValueError:
                    # Occurs if all numbers identical in samples
                    pval_df.at[dep_var, f"{indep_var}_p"] = pd.NA
                    pval_df.at[dep_var,  f"{indep_var}_stat"] = pd.NA
            else:
                pval_df.at[dep_var, f"{indep_var}_p"] = pd.NA
                pval_df.at[dep_var,  f"{indep_var}_stat"] = pd.NA
           # pval_df["question"]=question_name
    index=pd.MultiIndex.from_tuples(list(zip(question_names,cleaned_indexnames)), names=["question", "item"])
    pval_df.index=index
    
    if limit!=None:
        pval_df=pval_df[pval_df[f"{indep_var}_p"]<limit].dropna(how="all")
    
    return pval_df


def compare_likert_and_plot(df, indep_var, dep_var, save_path=None,plot=True):
    from MR_Qualtrics_module import plot_qid_likert  # Qualtrics helpers, imported here so MaxStat also works without that module
    
    from scipy.stats import kruskal
    
    groups = df.loc[2:,[indep_var, dep_var]].dropna().groupby(indep_var)[dep_var]
    
    group_names = list(groups.groups.keys())
    temp_cols = []
    group_vals = []
    for g in group_names:
        col = f"{g}"
        temp_cols.append(col)
        df[col] = None
        df.loc[0,col]=df.loc[0,dep_var].split("-")[-1]
        mask = df[indep_var] == g
        df.loc[mask, col] = df.loc[mask, dep_var]
        vals = groups.get_group(g).str.extract(r'\((\d+)\)').astype(float)[0].dropna().to_list()
        group_vals.append(vals)
    
    stat, p = kruskal(*group_vals)
    
    print(f"Kruskal-Wallis H: {stat:.2f}, p = {p:.4f}")
    if plot:
        plot_qid_likert(df, likert_cols=temp_cols, item_labels=temp_cols)
    
       # sns.set_theme(style="whitegrid")
        #plt.gcf().set_size_inches(12, 5)
        max_val = max(df[col].str.extract(r'\((\d+)\)').astype(float)[0].max() for col in temp_cols)
        #plt.xlim(right=max_val*15 + 5)
    
    
        plt.legend(title=f"Kruskal-Wallis H: {stat:.2f}, p = {p:.4f}", loc="center left", bbox_to_anchor=(1.0, 0.5))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    
        #df.drop(columns=temp_cols, inplace=True)

    return stat, p,df[temp_cols]


def CORR_prepare(df, columns, limit=0.2,focus_columns=None):
    from MR_Qualtrics_module import clean_text  # Qualtrics helpers, imported here so MaxStat also works without that module
    
    # Clean the column names text in row 0 for the desired columns
    cleaned_colnames = df.loc[0, columns].apply(clean_text).str.split(" - ").str[-1]

    if focus_columns!=None: 
        dep_vars=df[columns].columns.difference(focus_columns)
    
    # Extract numeric data from rows starting from index 1
    numeric_data = df.loc[1:, columns].apply(
        lambda col: col.astype(str)
                        .str.extract(r'(\d+(?:\.\d+)?)')[0]  # extract numeric part
                        .str.split('. ').str[0]               # take part before '. '
                        .astype(float)
    )
    
    # Combine into a DataFrame with cleaned column names
    df2 = numeric_data.copy()
    df2.columns = cleaned_colnames.to_list()
    
    # Compute Spearman correlations
    corrr = df2.corr(method="spearman")
    
    # Filter correlations based on threshold and exclude self-correlations (1)

    if focus_columns!=None:        

        cleaned_indexnames = df.loc[0, dep_vars].apply(clean_text).str.split("- ").str[-1]
        question_names = df.loc[0, dep_vars].apply(clean_text).str.split("- ").str[0].str.split("(").str[0].str.split("\n\n").str[0]
        tuples=list(zip(question_names,cleaned_indexnames))
        index=pd.MultiIndex.from_tuples(tuples, names=["question", "item"])
  
 
        ordinal_demo_questions=list(df.loc[0,focus_columns].values)
        
        focus_columns=[i for i in ordinal_demo_questions if i in corrr.columns]
        
        
        corrr = corrr.loc[focus_columns, corrr.columns.difference(focus_columns)]
        corrr=corrr.T
        corrr.index=index
        
    corrr = corrr[(corrr.abs() > limit) & (corrr.abs() != 1)].dropna(how="all").dropna(how="all", axis=1)
    return corrr


def CORR_plot(corrr, title='Spearman correlations', label_font_size=12, figsize=(25, 26), cbar=True, Poster=False,linecolor='lightgrey'):
    import textwrap
    
    cmap = sns.diverging_palette(10, 220, as_cmap=True)
    
    f, ax = plt.subplots(figsize=figsize)
    if Poster==True:
        sns.set_context('poster')
    else:
        sns.set_context('paper')
            
    mask = np.zeros_like(corrr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    sns_plot = sns.heatmap(corrr, mask=mask, cmap=cmap, vmax=0.7, vmin=-0.7,
                           square=True, linewidths=.50, 
                           linecolor=linecolor,cbar_kws={"shrink": .5, "pad": -0.07},
                           annot=True, fmt='.2g', annot_kws={"size": label_font_size}, cbar=False)
    
    
    # Word-wrap helper
    def wrap_labels(labels, width=25):
        return [textwrap.fill(l.get_text(), width) for l in labels]
    
    ax.set_title(title)
    ax.set_xticklabels(wrap_labels(ax.get_xticklabels()), fontsize=label_font_size)
    ax.set_yticklabels(wrap_labels(ax.get_yticklabels()), fontsize=label_font_size)
    
    plt.tight_layout()
    return sns_plot
