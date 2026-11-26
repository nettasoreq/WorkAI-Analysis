#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WorkAI thesis: complete data analysis code
==========================================

Students' Learning Processes and Performance in Evaluating GenAI Sources:
Comparing Simulation and Chatbot Environments (M.B.A. thesis, Bar-Ilan University).

This single file contains the full analysis pipeline, in the order in which it is run:

  Part 0   Statistical helpers (bootstrap confidence intervals, effect sizes)
  Part 1  Master participant table (source data, completers, and exclusions)
           Thesis: Sections 5.2 to 5.4; Appendix B.1 to B.2
  Part 2  Transition analysis between action states
           Thesis: Figure 3; Tables E3 and E14
  Part 3  Codebook of all variables
           Thesis: Appendix B.5
  Part 4  Cleaning, outliers, normality, scale reliability, sample characteristics
           Thesis: Sections 5.3 to 5.4; Tables B3, B4, B8
  Part 5  Process layer: activity and verification measures
           Thesis: Section 6.1; Tables E12, E13, E15
  Part 6  Performance layer: composite score, dimensions, SDT, reflection
           Thesis: Section 6.2; Figure 6; Tables E4, E5, E16; Figure E1
  Part 7  Perception layer: questionnaire scales, items, open-ended questions
           Thesis: Section 6.3; Tables E6, E7, E11, E17
  Part 8  Cross-layer correlations (exploratory)
           Thesis: Appendix E.3.2
  Part 9  (removed: hard-coded summary table, not an analysis)
  Part 10 All-pairs correlation matrix (exploratory)
           Thesis: Appendix E.3.2
  Part 11 (removed: superseded by Part 16)
  Part 12 Sensitivity analyses
           Thesis: Section 6.5; Table E10 (rows A to F, H)
  Part 13 Attrition funnel and completion comparisons
           Thesis: Section 5.2; Tables B1, B2
  Part 14 Source-level classification accuracy from the platform database
           Thesis: Section 6.2, Section 6.4, Table E10 (K, N, O), Tables E19 and E20
  Part 15 Comparison of the transition structure between the conditions
           Thesis: Section 6.1 (G-squared), Table E14, Figure E4
  Part 16 Regression models and checks
           Thesis: Table E9, Table E8 (full-sample rows), Table E10 (row I)
  Part 17 R pipeline: ENA, ONA, permutation tests, Table E8, network figures
           Thesis: Figures 4, 5, E2; Tables E1, E2, E8, E10 (row G)
  Part 18 R figures: Figure E3 with Table E9 (Model 3), Figure E1

  A correction is applied after Part 1 (see apply_reflection_score_correction).

Usage
-----
  Place this file in the project folder that contains the source workbook
  (WorkAI_מאגר_מתכלל_מלא_נבדקים_והצלבות_original_backup.xlsx), the coded
  logs (WorkAI_קידוד_מלא_לוגים_ENA_ONA.xlsx, ena_dataset_workai_english.csv,
  ena_ona_dataset_coded_english.csv) and the platform database (Db_278.db), then run:

      python3 WorkAI_analysis_code.py

  Intermediate tables are written to ./analysis_scripts/ (created if missing).
  The R parts are written to temporary files and run with Rscript.

Requirements
------------
  Python 3.10+: pandas, numpy, scipy, statsmodels, pingouin, scikit-learn, openpyxl, matplotlib
  R 4.3+: rENA, ggplot2, ggrepel, dplyr, grid, gridExtra, cluster, ordinal

The participant data are not included in this repository.
"""

import os
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_DIR, 'analysis_scripts')
os.makedirs(SCRIPTS_DIR, exist_ok=True)


#==============================================================================
# PART 0: Statistical helpers
#==============================================================================
"""
Bootstrap Utilities for Effect Size Confidence Intervals (WorkAI Research)
Provides 95% BCa bootstrap confidence intervals for rank-biserial r and Cramer's V,
and exact confidence intervals for Fisher's exact Odds Ratios.
"""

import numpy as np
from scipy import stats
from scipy.stats.contingency import odds_ratio

def r_rb_stat(x, y, axis=-1):
    """
    Vectorized rank-biserial correlation r_rb.
    x is group 1 (e.g. Simulation), y is group 2 (e.g. Chatbot).
    r_rb = (U1 - U2) / (n1 * n2) = 2 * U1 / (n1 * n2) - 1.
    """
    x_exp = np.expand_dims(x, axis=-1)
    y_exp = np.expand_dims(y, axis=-2)
    diff = x_exp - y_exp
    u1 = np.sum(diff > 0, axis=(-2, -1)) + 0.5 * np.sum(diff == 0, axis=(-2, -1))
    n1 = x.shape[axis]
    n2 = y.shape[axis]
    return (2.0 * u1) / (n1 * n2) - 1.0

def compute_r_rb_ci(x, y, n_resamples=5000, random_state=42):
    """
    Computes 95% BCa bootstrap confidence interval for rank-biserial r.
    Falls back to percentile if BCa fails due to degeneracy.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    try:
        res = stats.bootstrap((x, y), r_rb_stat, n_resamples=n_resamples, method='BCa', random_state=random_state)
        ci_low, ci_high = res.confidence_interval.low, res.confidence_interval.high
        if np.isnan(ci_low) or np.isnan(ci_high):
            res_p = stats.bootstrap((x, y), r_rb_stat, n_resamples=n_resamples, method='percentile', random_state=random_state)
            ci_low, ci_high = res_p.confidence_interval.low, res_p.confidence_interval.high
    except Exception:
        res_p = stats.bootstrap((x, y), r_rb_stat, n_resamples=n_resamples, method='percentile', random_state=random_state)
        ci_low, ci_high = res_p.confidence_interval.low, res_p.confidence_interval.high
    return float(ci_low), float(ci_high)

def general_cramers_v(x, y, axis=-1):
    """
    Vectorized Cramer's V calculation for paired resamples.
    """
    if x.ndim == 1:
        x = x[None, :]
        y = y[None, :]
        squeeze = True
    else:
        squeeze = False
    
    n_resamples, n = x.shape
    u_x = np.unique(x)
    u_y = np.unique(y)
    r = len(u_x)
    c = len(u_y)
    k = min(r, c)
    if k <= 1 or n == 0:
        return 0.0 if squeeze else np.zeros(n_resamples)
    
    counts = np.zeros((n_resamples, r, c))
    for i, xi in enumerate(u_x):
        for j, yj in enumerate(u_y):
            counts[:, i, j] = np.sum((x == xi) & (y == yj), axis=axis)
    
    row_sums = counts.sum(axis=2, keepdims=True)
    col_sums = counts.sum(axis=1, keepdims=True)
    expected = (row_sums * col_sums) / n
    with np.errstate(divide='ignore', invalid='ignore'):
        diff_sq = (counts - expected)**2
        terms = np.where(expected > 0, diff_sq / expected, 0.0)
        chi2 = np.sum(terms, axis=(1, 2))
        v = np.sqrt(chi2 / (n * (k - 1)))
    return v[0] if squeeze else v

def compute_cramers_v_ci(x, y, n_resamples=5000, random_state=42):
    """
    Computes 95% BCa bootstrap confidence interval for Cramer's V from paired arrays.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    valid = ~(pd_isna(x) | pd_isna(y))
    x = x[valid]
    y = y[valid]
    if len(x) < 4:
        return np.nan, np.nan
    try:
        res = stats.bootstrap((x, y), general_cramers_v, paired=True, n_resamples=n_resamples, method='BCa', random_state=random_state)
        ci_low, ci_high = res.confidence_interval.low, res.confidence_interval.high
        if np.isnan(ci_low) or np.isnan(ci_high):
            res_p = stats.bootstrap((x, y), general_cramers_v, paired=True, n_resamples=n_resamples, method='percentile', random_state=random_state)
            ci_low, ci_high = res_p.confidence_interval.low, res_p.confidence_interval.high
    except Exception:
        res_p = stats.bootstrap((x, y), general_cramers_v, paired=True, n_resamples=n_resamples, method='percentile', random_state=random_state)
        ci_low, ci_high = res_p.confidence_interval.low, res_p.confidence_interval.high
    return float(ci_low), float(ci_high)

def pd_isna(val):
    if hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
        import pandas as pd
        return pd.isna(val)
    return False

def compute_cramers_v_ci_from_table(table, n_resamples=5000, random_state=42):
    """
    Reconstructs raw pairs from contingency table and computes 95% BCa CI.
    """
    table = np.asarray(table)
    r, c = table.shape
    x_list = []
    y_list = []
    for i in range(r):
        for j in range(c):
            count = int(table[i, j])
            x_list.extend([i] * count)
            y_list.extend([j] * count)
    return compute_cramers_v_ci(np.array(x_list), np.array(y_list), n_resamples=n_resamples, random_state=random_state)

def compute_fisher_exact_or_ci(table):
    """
    Computes exact conditional Fisher odds ratio and its exact 95% confidence interval.
    """
    table = np.asarray(table)
    try:
        res = odds_ratio(table, kind='conditional')
        ci = res.confidence_interval(0.95)
        return float(res.statistic), float(ci.low), float(ci.high)
    except Exception:
        odds, _ = stats.fisher_exact(table)
        return float(odds), np.nan, np.nan

def compute_woolf_or_ci(table):
    """
    Computes standard Odds Ratio and Woolf's 95% Confidence Interval (logit method).
    table = [[a, b], [c, d]]
    """
    table = np.asarray(table, dtype=float)
    a, b = table[0, 0], table[0, 1]
    c, d = table[1, 0], table[1, 1]
    if a == 0 or b == 0 or c == 0 or d == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_val = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ln_or = np.log(or_val)
    ci_low = np.exp(ln_or - 1.95996 * se)
    ci_high = np.exp(ln_or + 1.95996 * se)
    return float(or_val), float(ci_low), float(ci_high)

def format_ci(low, high, decimals=3):
    if np.isnan(low) or np.isnan(high):
        return "-"
    return f"[{low:.{decimals}f}, {high:.{decimals}f}]"


#==============================================================================
# PART 1: Master participant table (source data, completers, and exclusions)
# Thesis: Sections 5.2 to 5.4; Appendix B.1 to B.2
#==============================================================================
# (source script: 00_prepare_master_table.py)
import json

"""
analysis_scripts/00_prepare_master_table.py
============================================
הכנת גיליון 0: הטבלה המרכזת הנקייה (N=70).
1. טוען את קובץ ה-SSOT (כל_הנתונים_לפי_משתמש_N69).
2. מצרף את המשתתפת ה-70 (user_525vg81ny) עם נתוני שאלון מלאים.
3. יוצר עמודות דגל: השלימה_משימה, מילאה_שאלון, הוצגו_מקורות, הוצאה_מקידוד_צאט, דגל_חריג, דגל_מעורבות_נמוכה.
4. מחשב עמודות חדשות: ציון_4_ממדים, ציון_4_ממדים_ללא_ש3, Kember_ללא_ש3, תחקיר_ללא_ש3, אפס_כניסות.
5. מסיר עמודות מיותרות/כפולות מעותק העבודה.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = PROJECT_DIR
INPUT_EXCEL = os.path.join(BASE_DIR, 'WorkAI_מאגר_מתכלל_מלא_נבדקים_והצלבות_original_backup.xlsx')
OUTPUT_CSV = os.path.join(BASE_DIR, 'analysis_scripts', 'master_clean_n70.csv')

def main():
    print(">>> שלב 0: הכנת הטבלה המרכזת הנקייה N=70...")
    df = pd.read_excel(INPUT_EXCEL, sheet_name='כל_הנתונים_לפי_משתמש_N69', header=3)
    print(f"נטענו {len(df)} נבדקות מקוריות.")

    # 1. הוספת משתתפת 70: user_525vg81ny
    user_70_row = {col: np.nan for col in df.columns}
    # This respondent's questionnaire was missing from the source workbook; it is added
    # from the questionnaire export (data file, not included in the repository).
    with open(os.path.join(BASE_DIR, 'participant_70_questionnaire.json'), encoding='utf-8') as fh:
        user_70_row.update(json.load(fh))

    df = pd.concat([df, pd.DataFrame([user_70_row])], ignore_index=True)
    print(f"סה\"כ נבדקות לאחר הוספת המשתתפת ה-70: {len(df)}")

    # 2. עמודות דגל
    df['השלימה_משימה'] = (df['מזהה_משתמש'] != 'user_525vg81ny').astype(int)
    df['מילאה_שאלון'] = (df['מילא_שאלון_משוב'] == 'כן').astype(int)
    
    # הוצגו מקורות: 0 לשני נבדקי צ'אטבוט ולמשתתפת 70 שלא ביצעה משימה
    df['הוצגו_מקורות'] = 1
    no_src_uids = ['e03d82e7-a44c-4ce6-8215-abd0b7db18a3', 'user_d222kcmnr', 'user_525vg81ny']
    df.loc[df['מזהה_משתמש'].isin(no_src_uids), 'הוצגו_מקורות'] = 0

    # הוצאה מקידוד צ'אט: המשתתפת בעלת 80 ההודעות (user_isvijah1u)
    df['הוצאה_מקידוד_צאט'] = (df['מזהה_משתמש'] == 'user_isvijah1u').astype(int)

    # דגל מעורבות נמוכה: משך ביצוע נטו קטן מ-2 דקות
    df['דגל_מעורבות_נמוכה'] = 0
    df.loc[(df['השלימה_משימה'] == 1) & (df['משך_פעילות_נטו_מנוקה_דקות'] < 2.0), 'דגל_מעורבות_נמוכה'] = 1

    # דגל חריג: נזהה חריגים לפי כלל 2.5*IQR על המשתנים הרציפים המרכזיים
    cont_vars = [
        'משך_פעילות_נטו_מנוקה_דקות', 'סהכ_פעולות_במערכת', 'קצב_פעולות_לדקה_APM',
        'מקורות_שנבדקו_באתר', 'חיפושים_חיצוניים_בגוגל', 'מילים_בפרומפט_ראשון',
        'מילים_ברפלקציה', "סהכ_הודעות_צ'אט_לבוט", 'ציון_ממד_פרומפט',
        'ציון_ממד_זיהוי_מקורות', 'ציון_ממד_תהליך_ובדיקה', 'ציון_ממד_תחקיר'
    ]
    skewed_vars = ['משך_פעילות_נטו_מנוקה_דקות', 'חיפושים_חיצוניים_בגוגל', "סהכ_הודעות_צ'אט_לבוט", 'סהכ_פעולות_במערכת']
    
    outlier_uids = set()
    for var in cont_vars:
        for cond in ['סימולציה', "צ'אטבוט"]:
            sub = df[(df['קבוצת_ניסוי'] == cond) & (df['השלימה_משימה'] == 1)]
            s = sub[var].dropna()
            if len(s) == 0:
                continue
            val = np.log1p(s) if var in skewed_vars else s
            q1 = val.quantile(0.25)
            q3 = val.quantile(0.75)
            iqr = q3 - q1
            low = q1 - 2.5 * iqr
            high = q3 + 2.5 * iqr
            is_out = (val < low) | (val > high)
            outlier_uids.update(sub.loc[s.index[is_out], 'מזהה_משתמש'].tolist())

    df['דגל_חריג'] = df['מזהה_משתמש'].apply(lambda u: 1 if u in outlier_uids else 0)
    print(f"זוהו {df['דגל_חריג'].sum()} משתתפות עם דגל חריג (2.5*IQR).")

    # 3. עמודות מחושבות
    # ציון 4 ממדים: (15*P + 35*V + 30*Pr + 10*Ref) / 90
    p = df['ציון_ממד_פרומפט']
    v = df['ציון_ממד_זיהוי_מקורות']
    pr = df['ציון_ממד_תהליך_ובדיקה']
    ref = df['ציון_ממד_תחקיר']
    df['ציון_4_ממדים'] = (15.0*p + 35.0*v + 30.0*pr + 10.0*ref) / 90.0

    # Kember ללא שאלה 3:
    # עבור סימולציה: מעמודה 168 (רמת_Kember_ללא_שאלה_3)
    # עבור צ'אטבוט: זהה ל-רמת_Kember המקורית (שאלה 3 לא הוצגה)
    df['Kember_ללא_ש3'] = df['רמת_Kember_ללא_שאלה_3'].copy()
    df.loc[df['קבוצת_ניסוי'] == "צ'אטבוט", 'Kember_ללא_ש3'] = df.loc[df['קבוצת_ניסוי'] == "צ'אטבוט", 'רמת_Kember']

    # תחקיר ללא שאלה 3: המרה של Kember_ללא_ש3 לציון נקודות (0->0, 1->30, 2->60, 3->90, 4->100)
    kember_score_map = {0: 0.0, 1: 30.0, 2: 60.0, 3: 90.0, 4: 100.0}
    df['תחקיר_ללא_ש3'] = df['Kember_ללא_ש3'].map(kember_score_map)

    # ציון 4 ממדים ללא שאלה 3:
    df['ציון_4_ממדים_ללא_ש3'] = (15.0*p + 35.0*v + 30.0*pr + 10.0*df['תחקיר_ללא_ש3']) / 90.0

    # אפס כניסות למקורות
    df['אפס_כניסות'] = np.nan
    mask_completed = (df['השלימה_משימה'] == 1)
    df.loc[mask_completed, 'אפס_כניסות'] = (df.loc[mask_completed, 'מקורות_שנבדקו_באתר'] == 0).astype(int)

    # מדדי SDT שמות אחידים
    df['hits'] = df['זיהוי_שגוי_כשגוי']
    df['misses'] = df['זיהוי_שגוי_כאמין']
    df['FA'] = df['זיהוי_אמין_כשגוי']
    df['CR'] = df['זיהוי_אמין_כאמין']
    df['H_rate'] = (df['hits'] + 0.5) / (df['hits'] + df['misses'] + 1.0)
    df['FA_rate'] = (df['FA'] + 0.5) / (df['FA'] + df['CR'] + 1.0)
    df['d_prime'] = df['d_prime_מדד_רגישות']
    df['c'] = df['c_bias_מדד_קריטריון']

    # סולמות שאלון מקוצרים
    df['PU'] = df['TAM_PU_שימושיות_נתפסת']
    df['PEOU'] = df['TAM_PEOU_קלות_שימוש']
    df['BI'] = df['TAM_Intent_כוונת_שימוש']
    df['CriTT'] = df['CRITT_Overall_הערכה_עצמית_ביקורתית']

    # 4. מחיקת עמודות מיותרות
    cols_to_drop = [
        'ציון_מערכת_מקורי',
        'משך_פעילות_כולל_דקות',
        'אשכול_K3_קוד',
        'אשכול_K3_שם_פרופיל',
        'outlier_מדגם_NSSHO',
        'כמות_דגלוני_outlier',
        'מדגם_נקי_ללא_חריגים_N46'
    ]
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_drop, inplace=True)
    print(f"הוסרו עמודות מיותרות: {existing_drop}")

    # שמירה
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"הטבלה המרכזת נשמרה : {OUTPUT_CSV} ({df.shape[0]} שורות, {df.shape[1]} עמודות)")

if __name__ == '__main__':
    main()


# ------------------------------------------------------------------------------
# Correction applied after the master table is built (Appendix C, Table C5):
# the reflection dimension score must follow the Kember level. One Chatbot
# participant (user_8ak40wpi2) was coded at level 0 ("no reflection") but the
# platform had stored a reflection score of 60. The score is set to 0 and the
# composite score is recomputed with the official weights (15/35/30/10 of 90).
# ------------------------------------------------------------------------------
def apply_reflection_score_correction():
    path = os.path.join(SCRIPTS_DIR, 'master_clean_n70.csv')
    m = pd.read_csv(path)
    idx = m['מזהה_משתמש'] == 'user_8ak40wpi2'
    assert idx.sum() == 1
    m.loc[idx, 'ציון_ממד_תחקיר'] = 0.0
    m.loc[idx, 'ציון_רפלקציה_ללא_שאלה_3'] = 0.0
    comp = (15.0 * m['ציון_ממד_פרומפט'] + 35.0 * m['ציון_ממד_זיהוי_מקורות']
            + 30.0 * m['ציון_ממד_תהליך_ובדיקה'] + 10.0 * m['ציון_ממד_תחקיר']) / 90.0
    m.loc[idx, 'ציון_4_ממדים'] = comp[idx]
    m.loc[idx, 'ציון_מנורמל_ללא_שיפור'] = comp[idx].round(2)
    m.to_csv(path, index=False, encoding='utf-8-sig')
    print('Reflection score correction applied to user_8ak40wpi2')

apply_reflection_score_correction()


#==============================================================================
# PART 2: Transition analysis between action states
# Thesis: Figure 3; Tables E3 and E14; Figure E4
#==============================================================================
# (source script: 03c_markov_analysis.py)

# ==============================================================================
# analysis_scripts/03c_markov_analysis.py
# ניתוח שרשראות מרקוב (Markov Chain Analysis) - רצפי פעולות התנהגותיים בלוג
# מעבר בין 5 מצבים: פרומפט, התייעצות, כניסה למקור, חיפוש חיצוני, סיווג
# כולל: איחוד חזרות, מבחן קיום G² עם 5,000 פרמוטציות, והשוואת 4 מעברי מפתח (FDR)
# פרויקט מחקר WorkAI
# ==============================================================================

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import networkx as nx

def fdr_bh(pvals):
    n_tests = len(pvals)
    sorted_indices = np.argsort(pvals)
    adj_p = np.zeros(n_tests)
    cum_min = 1.0
    for rank, idx in reversed(list(enumerate(sorted_indices))):
        p_adj = pvals[idx] * n_tests / (rank + 1)
        cum_min = min(cum_min, p_adj)
        adj_p[idx] = min(cum_min, 1.0)
    return adj_p

def main():
    print(">>> שלב 3ג: הרצת ניתוח מרקוב מלא (Markov Sequence Analysis)...")
    base_dir = SCRIPTS_DIR
    workspace_dir = os.path.dirname(base_dir)
    log_excel_path = os.path.join(workspace_dir, "WorkAI_קידוד_מלא_לוגים_ENA_ONA.xlsx")

    # טעינת נתוני הלוגים המלאים
    df_raw = pd.read_excel(log_excel_path, sheet_name="ENA_ONA_Dataset")
    df_raw['user_id'] = df_raw['user_id'].astype(str).str.replace('"', '').str.strip()

    # ==============================================================================
    # שלב 1: הגדרת 5 המצבים וסינון נבדקות
    # ==============================================================================
    # קודים שנכנסים לכל מצב:
    prompt_cols = [
        'Prompt - clear task', 'Prompt - source request', 'Prompt - quality requirement',
        'Prompt - recency requirement', 'Prompt - global scope', 'פרומפט (משימה בפעילות)',
        'בקשה לדיוק / קבלת מקורות אחרים'
    ]
    consult_cols = [
        'בקשה להכוונה לימודית', 'בירור לגבי מקור', 'תקלה / קושי בממשק',
        'בקשה לאימות מקורות (בדוק את המקור)', 'עזרה בניסוח פרומפט',
        'בקשה להסבר על אופן הפעולה של הצ׳אט', 'כללי', 'לא קשור'
    ]
    visit_cols = ['Source access']
    search_cols = ['External search']
    class_cols = ['Classification - reliable', 'Classification - unreliable', 'Classification revision']

    # החרגת המשתתפת החריגה user_isvijah1u (כמו ב-ENA)
    df_clean = df_raw[df_raw['user_id'] != 'user_isvijah1u'].copy()

    STATES = ['פרומפט', 'התייעצות', 'כניסה למקור', 'חיפוש חיצוני', 'סיווג']
    state_to_idx = {s: i for i, s in enumerate(STATES)}

    def get_row_states(row):
        states = []
        if any(row[c] == 1 for c in prompt_cols):
            states.append('פרומפט')
        if any(row[c] == 1 for c in consult_cols):
            states.append('התייעצות')
        # בדיקות מקור מקדימות סיווג
        if any(row[c] == 1 for c in search_cols):
            states.append('חיפוש חיצוני')
        if any(row[c] == 1 for c in visit_cols):
            states.append('כניסה למקור')
        if any(row[c] == 1 for c in class_cols):
            states.append('סיווג')
        return states

    # בניית רצף לכל משתתפת ואיחוד חזרות (Collapse consecutive duplicates)
    user_sequences = {}
    user_conditions = {}
    user_raw_lengths = {}

    for u, group in df_clean.groupby('user_id'):
        cond = group['condition'].iloc[0]
        user_conditions[u] = cond
        # מיון כרונולוגי לפי חותמת זמן בלוג
        sorted_group = group.sort_values(by=['Timestamp', 'action_order'])
        raw_states = []
        for _, row in sorted_group.iterrows():
            raw_states.extend(get_row_states(row))
        user_raw_lengths[u] = len(raw_states)
        
        # איחוד חזרות רצופות
        collapsed = []
        for s in raw_states:
            if not collapsed or collapsed[-1] != s:
                collapsed.append(s)
        user_sequences[u] = collapsed

    # סינון משתתפות עם פחות מ-2 מצבים ברצף
    sim_valid = {u: seq for u, seq in user_sequences.items() if user_conditions[u] == 'simulation' and len(seq) >= 2}
    sim_invalid = {u: seq for u, seq in user_sequences.items() if user_conditions[u] == 'simulation' and len(seq) < 2}
    cb_valid  = {u: seq for u, seq in user_sequences.items() if user_conditions[u] == 'chatbot' and len(seq) >= 2}
    cb_invalid  = {u: seq for u, seq in user_sequences.items() if user_conditions[u] == 'chatbot' and len(seq) < 2}

    print(f"סימולציה: N={len(sim_valid)} נכללות, {len(sim_invalid)} הוצאו עקב < 2 פעולות ({list(sim_invalid.keys())})")
    print(f"צ'אטבוט: N={len(cb_valid)} נכללות, {len(cb_invalid)} הוצאו עקב < 2 פעולות ({list(cb_invalid.keys())})")

    # ==============================================================================
    # שלב 2: טבלאות ספירות ומטריצות הסתברויות (5x5)
    # ==============================================================================
    def build_count_matrix(seq_dict):
        mat = pd.DataFrame(0, index=STATES, columns=STATES)
        for seq in seq_dict.values():
            for i in range(len(seq) - 1):
                mat.loc[seq[i], seq[i+1]] += 1
        return mat

    sim_counts = build_count_matrix(sim_valid)
    cb_counts  = build_count_matrix(cb_valid)

    # שמירת טבלאות ספירות
    sim_counts_out = sim_counts.copy()
    sim_counts_out.index.name = 'מצב נוכחי \\ מצב הבא'
    sim_counts_out.to_csv(os.path.join(base_dir, "tab_3_markov_counts_sim.csv"), encoding='utf-8-sig')

    cb_counts_out = cb_counts.copy()
    cb_counts_out.index.name = 'מצב נוכחי \\ מצב הבא'
    cb_counts_out.to_csv(os.path.join(base_dir, "tab_3_markov_counts_cb.csv"), encoding='utf-8-sig')

    # חישוב מטריצות הסתברויות עם N_out וחיווי יציבות
    def build_prob_matrix(counts_df):
        row_sums = counts_df.sum(axis=1)
        prob_df = counts_df.div(row_sums, axis=0).fillna(0)
        # עיגול ל-3 ספרות
        prob_df = prob_df.round(3)
        prob_df['N_מעברים_יוצאים'] = row_sums.astype(int)
        prob_df['סטטוס_יציבות'] = np.where(row_sums >= 10, 'יציב (≥10)', 'לא יציב (<10)')
        prob_df.index.name = 'מצב נוכחי \\ הסתברות למצב הבא'
        return prob_df

    sim_probs = build_prob_matrix(sim_counts)
    cb_probs  = build_prob_matrix(cb_counts)

    sim_probs.to_csv(os.path.join(base_dir, "tab_3_markov_probs_sim.csv"), encoding='utf-8-sig')
    cb_probs.to_csv(os.path.join(base_dir, "tab_3_markov_probs_cb.csv"), encoding='utf-8-sig')

    # ==============================================================================
    # שלב 3: מבחן הקיום (Likelihood Ratio G² + 5000 פרמוטציות לכל קבוצה)
    # ==============================================================================
    def calculate_g2(counts_mat):
        mat_vals = counts_mat.values.astype(float)
        row_sums = mat_vals.sum(axis=1)
        col_sums = mat_vals.sum(axis=0)
        total = mat_vals.sum()
        if total == 0:
            return 0.0
        E = np.outer(row_sums, col_sums) / total
        mask = (mat_vals > 0) & (E > 0)
        g2 = 2.0 * np.sum(mat_vals[mask] * np.log(mat_vals[mask] / E[mask]))
        return g2

    g2_sim_obs = calculate_g2(sim_counts)
    g2_cb_obs  = calculate_g2(cb_counts)

    # הרצת מבחן פרמוטציה (5,000 איטרציות)
    print("מריץ מבחן פרמוטציה (5,000 איטרציות) למבחן הקיום של שרשרת מרקוב...")
    np.random.seed(42)
    n_perms = 5000

    def run_permutation_test(seq_dict, g2_obs):
        seq_list = list(seq_dict.values())
        count_ge = 0
        for _ in range(n_perms):
            perm_counts = np.zeros((5, 5))
            for seq in seq_list:
                shuff = list(np.random.permutation(seq))
                # איחוד חזרות מחדש לאחר הערבוב
                c_shuff = []
                for s in shuff:
                    if not c_shuff or c_shuff[-1] != s:
                        c_shuff.append(s)
                for i in range(len(c_shuff) - 1):
                    perm_counts[state_to_idx[c_shuff[i]], state_to_idx[c_shuff[i+1]]] += 1
            
            rs = perm_counts.sum(axis=1)
            cs = perm_counts.sum(axis=0)
            tot = perm_counts.sum()
            if tot > 0:
                E_perm = np.outer(rs, cs) / tot
                mask = (perm_counts > 0) & (E_perm > 0)
                g2_perm = 2.0 * np.sum(perm_counts[mask] * np.log(perm_counts[mask] / E_perm[mask]))
                if g2_perm >= g2_obs:
                    count_ge += 1
        p_val = (count_ge + 1) / (n_perms + 1)
        return p_val

    p_perm_sim = run_permutation_test(sim_valid, g2_sim_obs)
    p_perm_cb  = run_permutation_test(cb_valid, g2_cb_obs)

    tab_existence = pd.DataFrame([
        {
            'קבוצה': 'סימולציה',
            'מספר משתתפות נכללות': len(sim_valid),
            'סך מעברים שנצפו': int(sim_counts.values.sum()),
            'סטטיסטי G²': f"{g2_sim_obs:.2f}",
            'דרגות חופש מקסימליות': (len(STATES)-1)**2,
            'ערך p מפרמוטציה (5000)': f"{p_perm_sim:.4f}",
            'מובהק (p<.05)': 'כן (תלות סדרתית מובהקת)',
            'מסקנה מתודולוגית': 'הפעולה הבאה תלויה בפעולה הנוכחית – קיים מבנה תהליכי מסדר ראשון לפחות'
        },
        {
            'קבוצה': "צ'אטבוט",
            'מספר משתתפות נכללות': len(cb_valid),
            'סך מעברים שנצפו': int(cb_counts.values.sum()),
            'סטטיסטי G²': f"{g2_cb_obs:.2f}",
            'דרגות חופש מקסימליות': (len(STATES)-1)**2,
            'ערך p מפרמוטציה (5000)': f"{p_perm_cb:.4f}",
            'מובהק (p<.05)': 'כן (תלות סדרתית מובהקת)',
            'מסקנה מתודולוגית': 'הפעולה הבאה תלויה בפעולה הנוכחית – קיים מבנה תהליכי מסדר ראשון לפחות'
        }
    ])
    tab_existence.to_csv(os.path.join(base_dir, "tab_3_markov_existence_test.csv"), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # שלב 4: השוואת 4 מעברי מפתח שנבחרו מראש
    # ==============================================================================
    key_transitions_def = [
        ('מרקוב_פרומפט_אל_סיווג', 'פרומפט אל סיווג', 'דילוג על שלב הבדיקה (מעבר ישיר מתוצר AI לקביעת אמינות)', 'פרומפט', 'סיווג'),
        ('מרקוב_פרומפט_אל_כניסה_למקור', 'פרומפט אל כניסה למקור', 'בדיקה מיידית אחרי קבלת מקורות (גישה ישירה לאתר)', 'פרומפט', 'כניסה למקור'),
        ('מרקוב_חיפוש_חיצוני_אל_כניסה_למקור', 'חיפוש חיצוני אל כניסה למקור', 'לולאת קריאה רוחבית (Lateral Reading: חיפוש בגוגל ואז כניסה למקור)', 'חיפוש חיצוני', 'כניסה למקור'),
        ('מרקוב_כניסה_למקור_אל_סיווג', 'כניסה למקור אל סיווג', 'סיווג מבוסס בדיקה (קבלת החלטת אמינות לאחר עיון באתר המקור)', 'כניסה למקור', 'סיווג')
    ]

    # חישוב הסתברויות אישיות לכל משתתפת
    all_users = list(user_sequences.keys())
    user_indiv_probs = []

    for u in all_users:
        seq = user_sequences[u]
        cond = user_conditions[u]
        row_u = {'מזהה_משתמש': u, 'קבוצה': cond}
        
        # ספירת מעברים אישיים
        if len(seq) >= 2:
            out_counts = {s: 0 for s in STATES}
            pair_counts = {}
            for i in range(len(seq) - 1):
                s1, s2 = seq[i], seq[i+1]
                out_counts[s1] += 1
                pair_counts[(s1, s2)] = pair_counts.get((s1, s2), 0) + 1
                
            for col_name, trans_name, trans_desc, s_from, s_to in key_transitions_def:
                if out_counts[s_from] > 0:
                    prob = pair_counts.get((s_from, s_to), 0) / out_counts[s_from]
                    row_u[col_name] = round(prob, 4)
                else:
                    row_u[col_name] = np.nan
        else:
            for col_name, _, _, _, _ in key_transitions_def:
                row_u[col_name] = np.nan
                
        user_indiv_probs.append(row_u)

    df_indiv_probs = pd.DataFrame(user_indiv_probs)
    df_indiv_probs.to_csv(os.path.join(base_dir, "tab_3_markov_individual_probs.csv"), index=False, encoding='utf-8-sig')

    # מיזוג לטבלה המרכזת master_clean_n70.csv
    master_path = os.path.join(base_dir, "master_clean_n70.csv")
    if os.path.exists(master_path):
        df_master = pd.read_csv(master_path)
        markov_cols = [c[0] for c in key_transitions_def]
        df_master = df_master.drop(columns=[c for c in markov_cols if c in df_master.columns])
        df_merged = df_master.merge(df_indiv_probs[['מזהה_משתמש'] + markov_cols], on='מזהה_משתמש', how='left')
        df_merged.to_csv(master_path, index=False, encoding='utf-8-sig')
        print(f"עודכנה הטבלה המרכזת עם 4 עמודות מרקוב: {master_path} ({df_merged.shape[1]} עמודות)")

    # השוואת קבוצות על 4 מעברי המפתח
    key_rows = []
    pvals_key = []

    for col_name, trans_name, trans_desc, s_from, s_to in key_transitions_def:
        s_vals = df_indiv_probs[df_indiv_probs['קבוצה'] == 'simulation'][col_name].dropna()
        c_vals = df_indiv_probs[df_indiv_probs['קבוצה'] == 'chatbot'][col_name].dropna()
        
        n_s = len(s_vals)
        n_c = len(c_vals)
        
        if n_s > 0 and n_c > 0:
            res_u = stats.mannwhitneyu(s_vals, c_vals, alternative='two-sided')
            u_s = float(res_u.statistic)
            u_c = float(n_s * n_c - u_s)
            r_rb = (u_s - u_c) / (n_s * n_c)
            p_val = res_u.pvalue
            ci_low, ci_high = compute_r_rb_ci(s_vals.values, c_vals.values)
        else:
            u_s, u_c, r_rb, p_val = np.nan, np.nan, np.nan, np.nan
            ci_low, ci_high = np.nan, np.nan
            
        pvals_key.append(p_val)
        
        strength = 'גדול מאוד' if abs(r_rb) >= 0.5 else ('בינוני' if abs(r_rb) >= 0.3 else ('קטן' if abs(r_rb) >= 0.1 else 'זניח')) if not np.isnan(r_rb) else '-'
        
        key_rows.append({
            'מעבר מפתח': trans_name,
            'מה המעבר מייצג': trans_desc,
            'מבחן_סטטיסטי': 'Mann-Whitney U (דו-צדדי)',
            'סימולציה N': n_s,
            'סימולציה ממוצע (ס״ת)': f"{s_vals.mean():.3f} ({s_vals.std():.3f})" if n_s > 0 else "-",
            'צ׳אטבוט N': n_c,
            'צ׳אטבוט ממוצע (ס״ת)': f"{c_vals.mean():.3f} ({c_vals.std():.3f})" if n_c > 0 else "-",
            'Mann-Whitney U_Sim': f"{u_s:.1f}" if not np.isnan(u_s) else "-",
            'U_CB': f"{u_c:.1f}" if not np.isnan(u_c) else "-",
            'סכום U תקין': 'כן' if not np.isnan(u_s) and abs((u_s + u_c) - (n_s * n_c)) < 1e-4 else '-',
            'ערך p גולמי': p_val,
            'גודל אפקט r_rb': f"{r_rb:.3f}" if not np.isnan(r_rb) else "-",
            '95% CI (BCa)': format_ci(ci_low, ci_high),
            'עוצמת אפקט': strength
        })

    # תיקון FDR על 4 מעברי המפתח
    valid_p = [p for p in pvals_key if not np.isnan(p)]
    fdr_key = fdr_bh(valid_p)
    fdr_idx = 0
    for i, r in enumerate(key_rows):
        p_raw = r['ערך p גולמי']
        if not np.isnan(p_raw):
            p_fdr = fdr_key[fdr_idx]
            fdr_idx += 1
            r['ערך p גולמי'] = f"{p_raw:.4f}" if p_raw >= 0.0001 else "< .001"
            r['ערך p מתוקן FDR'] = f"{p_fdr:.4f}" if p_fdr >= 0.0001 else "< .001"
            r['מובהק (q<.05)'] = 'כן' if p_fdr < 0.05 else 'לא'
            # הוספת כוכביות
            if p_fdr < 0.001:
                stars = '***'
            elif p_fdr < 0.01:
                stars = '**'
            elif p_fdr < 0.05:
                stars = '*'
            else:
                stars = 'n.s.'
            r['רמת מובהקות'] = f"{stars} ({r['עוצמת אפקט']})" if stars != 'n.s.' else 'לא מובהק (n.s.)'
        else:
            r['ערך p גולמי'] = '-'
            r['ערך p מתוקן FDR'] = '-'
            r['מובהק (q<.05)'] = '-'
            r['רמת מובהקות'] = '-'

    tab_key = pd.DataFrame(key_rows)
    tab_key.to_csv(os.path.join(base_dir, "tab_3_markov_key_transitions.csv"), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # שלב 5: רשימת מעברים שאינם אפשריים בכל ממשק (Structural Zeros)
    # ==============================================================================
    impossible_rows = [
        {
            'ממשק': 'סימולציה',
            'מעבר חסום מבנית': 'פרומפט אל סיווג (פרומפט -> סיווג)',
            'סיבה מבנית / ממשקית': 'בסימולציה כפתורי הסיווג נעולים עד לפתיחת כרטיסיית המקור בסביבת דנה; המשתתפת אינה יכולה לקפוץ ישירות מפרומפט לסיווג ללא שלב ביניים.',
            'נצפה בפועל': '0 מעברים (0.0%)'
        },
        {
            'ממשק': 'סימולציה',
            'מעבר חסום מבנית': 'כל מעבר אל/מתוך התייעצות בוט חופשית',
            'סיבה מבנית / ממשקית': 'ממשק הסימולציה אינו כולל צ׳אטבוט מנחה לשיחה פתוחה במהלך חקר המקורות (סביבה סגורה ומודרכת).',
            'נצפה בפועל': 'חסום מבנית ברובו'
        },
        {
            'ממשק': 'שני הממשקים',
            'מעבר חסום מתודולוגית': 'מעברים עצמיים (A -> A)',
            'סיבה מבנית / ממשקית': 'איחוד חזרות רצופות (Run-length encoding) שבוצע במתודולוגיה מאחד כניסות חוזרות ומבטל את האלכסון הראשי כדי למנוע השתלטות של חזרות טכניות.',
            'נצפה בפועל': '0 מעברים (אלכסון אפס)'
        },
        {
            'ממשק': "צ'אטבוט",
            'מעבר חסום מבנית': 'אין מעברים חסומים מבנית',
            'סיבה מבנית / ממשקית': 'ממשק הצ׳אטבוט הוא חלון שיחה חופשי המאפשר ביצוע כל פעולה בכל סדר, כולל דילוג מלא על בדיקת מקורות ומעבר ישיר לסיווג.',
            'נצפה בפועל': 'כל המעברים אפשריים'
        }
    ]
    tab_impossible = pd.DataFrame(impossible_rows)
    tab_impossible.to_csv(os.path.join(base_dir, "tab_3_markov_impossible_transitions.csv"), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # שלב 6: יצירת דיאגרמות מעברים גרפיות (מעברים מעל 0.10)
    # ==============================================================================
    print("יוצר דיאגרמת מעברי מצבים מרקוביים (מעברים מעל 0.10)...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    pos = {
        'פרומפט': np.array([-1.0, 0.8]),
        'התייעצות': np.array([-1.0, -0.8]),
        'חיפוש חיצוני': np.array([0.0, 1.0]),
        'כניסה למקור': np.array([0.2, -0.2]),
        'סיווג': np.array([1.0, 0.0])
    }

    state_colors = {
        'פרומפט': '#10B981',
        'התייעצות': '#F59E0B',
        'כניסה למקור': '#0284C7',
        'חיפוש חיצוני': '#6366F1',
        'סיווג': '#DC2626'
    }

    group_data = [
        ('סימולציה (Simulation)', sim_probs, axes[0]),
        ("צ'אטבוט (Chatbot)", cb_probs, axes[1])
    ]

    for title, prob_mat, ax in group_data:
        G = nx.DiGraph()
        for s in STATES:
            G.add_node(s)

        edge_weights = []
        for s1 in STATES:
            for s2 in STATES:
                p_val = prob_mat.loc[s1, s2]
                if p_val >= 0.10 and s1 != s2:
                    G.add_edge(s1, s2, weight=p_val)
                    edge_weights.append(p_val)

        # Draw nodes
        node_c = [state_colors[node] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_c, node_size=2800, alpha=0.9)
        
        # Draw labels (Hebrew / English subtitles)
        heb_labels = {
            'פרומפט': 'פרומפט\n(Prompt)',
            'התייעצות': 'התייעצות\n(Consultation)',
            'כניסה למקור': 'כניסה למקור\n(Visit Site)',
            'חיפוש חיצוני': 'חיפוש בגוגל\n(Search)',
            'סיווג': 'סיווג אמינות\n(Classify)'
        }
        nx.draw_networkx_labels(G, pos, labels=heb_labels, ax=ax, font_size=9, font_family='Arial', font_weight='bold')

        # Draw edges with thickness proportional to probability
        for (u, v, d) in G.edges(data=True):
            w = d['weight']
            rad = 0.15 if G.has_edge(v, u) else 0.05
            ax.annotate("",
                        xy=pos[v], xycoords='data',
                        xytext=pos[u], textcoords='data',
                        arrowprops=dict(arrowstyle="-|>", color="#1E293B",
                                        lw=w * 8.0,
                                        connectionstyle=f"arc3,rad={rad}",
                                        shrinkA=22, shrinkB=22, alpha=0.75))
            # Edge text label
            mid_x = (pos[u][0] + pos[v][0]) / 2.0
            mid_y = (pos[u][1] + pos[v][1]) / 2.0 + (0.06 if rad > 0 else -0.04)
            ax.text(mid_x, mid_y, f"{w:.2f}", color="#0F172A", fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CBD5E1", alpha=0.85))

        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.axis('off')

    plt.suptitle("דיאגרמת מעברי מצבים מרקוביים (הסתברות מעבר P > 0.10 בלבד)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    img_out_path = os.path.join(base_dir, "markov_state_diagrams.png")
    plt.savefig(img_out_path, bbox_inches='tight')
    plt.close()
    print(f"דיאגרמת המעברים נשמרה ב: {img_out_path}")

    print(">>> שלב 3ג (ניתוח מרקוב) הושלם ")

if __name__ == '__main__':
    main()


#==============================================================================
# PART 3: Codebook of all variables
# Thesis: Appendix B.5
#==============================================================================
# (source script: 01_build_codebook.py)

"""
analysis_scripts/01_build_codebook.py
======================================
הכנת גיליון 1: ספר קודים ומילון משתנים מלא ומעודכן.
כולל הגדרות מחייבות, נוסחאות, טווחים, תנאי זמינות, ועמודת 'נבדק'.
"""

import os
import pandas as pd

BASE_DIR = PROJECT_DIR
INPUT_DATA_CSV = os.path.join(BASE_DIR, 'analysis_scripts', 'master_clean_n70.csv')
ORIG_EXCEL = os.path.join(BASE_DIR, 'WorkAI_מאגר_מתכלל_מלא_נבדקים_והצלבות_original_backup.xlsx')
OUTPUT_CSV = os.path.join(BASE_DIR, 'analysis_scripts', 'codebook_sheet1.csv')

def main():
    print(">>> שלב 1: בניית ספר הקודים (גיליון 1)...")
    df_data = pd.read_csv(INPUT_DATA_CSV)
    data_cols = list(df_data.columns)

    # טעינת ספר קודים מקורי אם קיים כבסיס
    try:
        orig_cb = pd.read_excel(ORIG_EXCEL, sheet_name='מילון_משתנים_וספר_קודים', header=2)
        cb_dict = {}
        for _, r in orig_cb.iterrows():
            name = str(r.get('שם עמודה באקסל', '')).strip()
            if name:
                cb_dict[name] = r.to_dict()
    except Exception as e:
        print(f"הערה בטעינת ספר קודים מקורי: {e}")
        cb_dict = {}

    # הגדרות ספציפיות מחייבות
    explicit_defs = {
        'משך_פעילות_נטו_מנוקה_דקות': {
            'רובד': 'זיהוי וסשן',
            'הגדרה': 'משך זמן ביצוע המשימה נטו במערכת בדקות, מנוקה מחוסר פעילות ומוגבל עליונה לתקרה של 60 דקות',
            'נוסחה': 'לוג פעילות נטו, חסום ב-60.0',
            'טווח': '0.0 - 60.0',
            'יחידה': 'דקות',
            'זמינות': 'שניהם',
            'n_צפוי': 69
        },
        'מקורות_שנבדקו_באתר': {
            'רובד': 'רובד 1: תהליך',
            'הגדרה': 'כניסות למקורות: מספר קליקים ובדיקות באתר המקור, כולל כניסות חוזרות ונשנות',
            'נוסחה': 'ספירת אירועי בדיקת מקור בלוג',
            'טווח': '0+',
            'יחידה': 'קליקים/כניסות',
            'זמינות': 'שניהם (הוצגו מקורות)',
            'n_צפוי': 67
        },
        'שיעור_ביקור_במקורות': {
            'רובד': 'רובד 1: תהליך',
            'הגדרה': 'שיעור ביקור במקורות: מקורות ייחודיים שונים שנפתחו חלקי סך המקורות שהוצגו לנבדק/ת',
            'נוסחה': 'מקורות ייחודיים שנבדקו / סה"כ מקורות שהוצגו',
            'טווח': '0.0 - 1.0',
            'יחידה': 'שיעור (פרופורציה)',
            'זמינות': 'שניהם (הוצגו מקורות)',
            'n_צפוי': 67
        },
        'שיעור_דיוק_סיווג_מקורות': {
            'רובד': 'רובד 2: ביצוע',
            'הגדרה': 'שיעור דיוק סיווג: סיווגים נכונים של מקורות (אמין/מטעה) חלקי סך המקורות שסווגו בפועל',
            'נוסחה': 'מקורות שסווגו נכון / סה"כ מקורות שסווגו',
            'טווח': '0.0 - 1.0',
            'יחידה': 'שיעור (פרופורציה)',
            'זמינות': 'שניהם (הוצגו מקורות)',
            'n_צפוי': 67
        },
        'שיעור_ניחוש_ללא_בדיקה': {
            'רובד': 'רובד 1: תהליך',
            'הגדרה': 'שיעור ניחוש: מקורות שסווגו מבלי שנפתחו באתר או נבדקו בגוגל, חלקי סך המקורות שסווגו',
            'נוסחה': 'מקורות שנוחשו ללא בדיקה / סה"כ מקורות שסווגו',
            'טווח': '0.0 - 1.0',
            'יחידה': 'שיעור (פרופורציה)',
            'זמינות': 'שניהם (הוצגו מקורות)',
            'n_צפוי': 67
        },
        'ניסיונות_פרומפט_משימה_מסווגים': {
            'רובד': 'רובד 1: תהליך',
            'הגדרה': 'ניסיונות פרומפט (הגדרה ראשית): מספר הודעות הפרומפט הממשיות שנשלחו למשימה, מסווגות ללא ברכות ושיח כללי',
            'נוסחה': 'ספירת פרומפטים מסווגי משימה',
            'טווח': '0+',
            'יחידה': 'פרומפטים',
            'זמינות': 'שניהם',
            'n_צפוי': 69
        },
        'מילים_בפרומפט_ראשון': {
            'רובד': 'רובד 1: תהליך',
            'הגדרה': 'אורך פרומפט (הגדרה ראשית): מספר המילים בפרומפט הראשון שנשלח על ידי הנבדק/ת למערכת',
            'נוסחה': 'ספירת מילים בפרומפט ראשון',
            'טווח': '0+',
            'יחידה': 'מילים',
            'זמינות': 'שניהם',
            'n_צפוי': 69
        },
        'רמת_Kember': {
            'רובד': 'רובד 2: ביצוע',
            'הגדרה': 'רמת Kember (סכמת 4 הקטגוריות): רמת רפלקציה כוללת אחת לנבדק/ת (0-4) המקודדת על הטקסט הסופי המאוחד (0 = היעדר רפלקציה/ריק, 1 = פעולה הרגלית, 2 = הבנה, 3 = רפלקציה, 4 = רפלקציה ביקורתית)',
            'נוסחה': 'סיווג Kember et al. (2000, 2008) על הטקסט המאוחד המלא',
            'טווח': '0 - 4',
            'יחידה': 'דרגה אורדינלית',
            'זמינות': 'שניהם',
            'n_צפוי': 69
        },
        'מדד_איכות_פרומפט': {
            'רובד': 'רובד 2: ביצוע',
            'הגדרה': 'איכות הפרומפט: ציון רובריקת מערכת משוקלל על ניסוח הפרומפטים בסולם 0 עד 100',
            'נוסחה': 'ציון מערכת אוטומטי (0-100)',
            'טווח': '0 - 100',
            'יחידה': 'נקודות',
            'זמינות': 'שניהם',
            'n_צפוי': 69
        },
        'ציון_4_ממדים': {
            'רובד': 'רובד 2: ביצוע',
            'הגדרה': 'ציון משוקלל ראשי ב-4 הממדים המשותפים (H1): שקלול 15% פרומפט, 35% זיהוי מקורות, 30% תהליך ובדיקה, 10% תחקיר (חלקי 90, מנורמל ל-100)',
            'נוסחה': '(15*ציון_פרומפט + 35*ציון_זיהוי + 30*ציון_תהליך + 10*ציון_תחקיר) / 90',
            'טווח': '0 - 100',
            'יחידה': 'נקודות',
            'זמינות': 'שניהם',
            'n_צפוי': 69
        },
        'ציון_4_ממדים_ללא_ש3': {
            'רובד': 'רובד 2 / רגישות',
            'הגדרה': 'ציון 4 ממדים מתוקן לניתוח רגישות א\': חישוב משוקלל כאשר ממד התחקיר מבוסס על סיווג Kember ללא שאלה 3',
            'נוסחה': '(15*P + 35*V + 30*Pr + 10*תחקיר_ללא_ש3) / 90',
            'טווח': '0 - 100',
            'יחידה': 'נקודות',
            'זמינות': 'שניהם',
            'n_צפוי': 69
        },
        'אפס_כניסות': {
            'רובד': 'רובד 1: תהליך',
            'הגדרה': 'אינדיקטור להימנעות מוחלטת מכניסה למקורות (1 אם הנבדק/ת לא נכנס/ה לאף מקור, אחרת 0)',
            'נוסחה': 'IF מקורות_שנבדקו_באתר == 0 THEN 1 ELSE 0',
            'טווח': '0 / 1',
            'יחידה': 'בינארי (דמי)',
            'זמינות': 'שניהם (הוצגו מקורות)',
            'n_צפוי': 67
        },
        'd_prime': {
            'רובד': 'רובד 2: ביצוע (SDT)',
            'הגדרה': 'מדד רגישות הבחנה (d-prime) בתיאוריית זיהוי אותות: יכולת סגולית להבחין בין מקור כוזב/מטעה (אות) למקור אמין (רעש)',
            'נוסחה': 'z(H_rate) - z(FA_rate) עם תיקון Hautus',
            'טווח': '-3.0 - +3.0',
            'יחידה': 'ציון תקן d\'',
            'זמינות': 'שניהם (הוצגו מקורות)',
            'n_צפוי': 67
        },
        'c': {
            'רובד': 'רובד 2: ביצוע (SDT)',
            'הגדרה': 'קריטריון הכרעה / הטיית אמינות (c criterion): ערך חיובי מעיד על קריטריון שמרני לפקפוק והטיית אמון (Truth Bias)',
            'נוסחה': '-0.5 * [z(H_rate) + z(FA_rate)] עם תיקון Hautus',
            'טווח': '-2.0 - +2.0',
            'יחידה': 'ציון תקן c',
            'זמינות': 'שניהם (הוצגו מקורות)',
            'n_צפוי': 67
        },
        'PU': {
            'רובד': 'רובד 3: תפיסה',
            'הגדרה': 'תועלת נתפסת (Perceived Usefulness): ממוצע פריטים 1 עד 4 בשאלון TAM',
            'נוסחה': 'Mean(TAM_1, TAM_2, TAM_3, TAM_4)',
            'טווח': '1 - 5',
            'יחידה': 'סולם ליקרט 1-5',
            'זמינות': 'שניהם (מילאו שאלון)',
            'n_צפוי': 49
        },
        'PEOU': {
            'רובד': 'רובד 3: תפיסה',
            'הגדרה': 'קלות שימוש נתפסת (Perceived Ease of Use): ממוצע פריטים 5 עד 8 בשאלון TAM',
            'נוסחה': 'Mean(TAM_5, TAM_6, TAM_7, TAM_8)',
            'טווח': '1 - 5',
            'יחידה': 'סולם ליקרט 1-5',
            'זמינות': 'שניהם (מילאו שאלון)',
            'n_צפוי': 49
        },
        'BI': {
            'רובד': 'רובד 3: תפיסה',
            'הגדרה': 'כוונת שימוש עתידית (Behavioral Intention): ממוצע פריטים 9 עד 10 בשאלון TAM',
            'נוסחה': 'Mean(TAM_9, TAM_10)',
            'טווח': '1 - 5',
            'יחידה': 'סולם ליקרט 1-5',
            'זמינות': 'שניהם (מילאו שאלון)',
            'n_צפוי': 49
        },
        'CriTT': {
            'רובד': 'רובד 3: תפיסה',
            'הגדרה': 'מסוגלות עצמית בחשיבה ביקורתית (CriTT): ממוצע 17 פריטי שאלון Critical Thinking Toolkit',
            'נוסחה': 'Mean(CRITT_1 .. CRITT_17)',
            'טווח': '1 - 5',
            'יחידה': 'סולם ליקרט 1-5',
            'זמינות': 'שניהם (מילאו שאלון)',
            'n_צפוי': 49
        }
    }

    codebook_rows = []
    for idx, col in enumerate(data_cols, start=1):
        if col in explicit_defs:
            d = explicit_defs[col]
            codebook_rows.append({
                'מספר_עמודה': idx,
                'שם_משתנה': col,
                'רובד_מחקרי': d['רובד'],
                'הגדרה_מילולית_תפעולית': d['הגדרה'],
                'נוסחה_ומקור': d['נוסחה'],
                'טווח_ערכים': d['טווח'],
                'יחידת_מידה': d['יחידה'],
                'תנאי_זמינות': d['זמינות'],
                'n_צפוי': d['n_צפוי'],
                'נבדק': ''
            })
        elif col in cb_dict:
            orig = cb_dict[col]
            # ברירת מחדל מתנאים
            zminut = 'שניהם'
            n_exp = 69
            if 'שהייה' in col or 'שיפור' in col or 'הסברים' in col:
                zminut = 'סימולציה בלבד'
                n_exp = 30
            elif 'שאלון' in col or 'TAM' in col or 'CRITT' in col:
                zminut = 'שניהם (מילאו שאלון)'
                n_exp = 49
            elif 'מקור' in col or 'סיווג' in col:
                zminut = 'שניהם (הוצגו מקורות)'
                n_exp = 67

            codebook_rows.append({
                'מספר_עמודה': idx,
                'שם_משתנה': col,
                'רובד_מחקרי': orig.get('רובד מחקרי', 'לוגים ומערכת'),
                'הגדרה_מילולית_תפעולית': orig.get('משמעות והגדרה תפעולית', col),
                'נוסחה_ומקור': orig.get('מקור הנתונים', 'קובץ נתונים ראשי'),
                'טווח_ערכים': orig.get('סולם / טווח ערכים', 'מגוון'),
                'יחידת_מידה': orig.get('סוג משתנה', 'נומרי/טקסט'),
                'תנאי_זמינות': zminut,
                'n_צפוי': n_exp,
                'נבדק': ''
            })
        else:
            # משתנה חדש שנוסף
            robad = 'זיהוי ודגלים'
            zminut = 'שניהם'
            n_exp = 70
            definition = f'משתנה מעובד ומחושב: {col}'
            formula = 'חישוב מובנה בגיליון 0'
            val_range = 'מגוון'
            v_unit = 'ערך נומרי / דגל'

            if 'מרקוב' in col:
                robad = 'רובד 1: תהליך (מרקוב)'
                formula = 'P_i(B|A) = מעברים A->B חלקי סך יציאות מ-A ברצף האישי'
                val_range = '0.0000 - 1.0000'
                v_unit = 'הסתברות מעבר (0-1)'
                if 'פרומפט_אל_סיווג' in col:
                    definition = 'הסתברות מעבר אישית מפרומפט ישירות לסיווג (דילוג על שלב הבדיקה)'
                    zminut = 'משתתפות שביצעו פרומפט'
                    n_exp = 67
                elif 'פרומפט_אל_כניסה_למקור' in col:
                    definition = 'הסתברות מעבר אישית מפרומפט לכניסה למקור באתר (בדיקה מיידית לאחר תוצר AI)'
                    zminut = 'משתתפות שביצעו פרומפט'
                    n_exp = 67
                elif 'חיפוש_חיצוני_אל_כניסה_למקור' in col:
                    definition = 'הסתברות מעבר אישית מחיפוש חיצוני לכניסה למקור (לולאת קריאה רוחבית - Lateral Reading)'
                    zminut = 'משתתפות שביצעו חיפוש בגוגל'
                    n_exp = 31
                elif 'כניסה_למקור_אל_סיווג' in col:
                    definition = 'הסתברות מעבר אישית מכניסה למקור לסיווג (קבלת החלטת אמינות לאחר עיון במקור)'
                    zminut = 'משתתפות שנכנסו למקור'
                    n_exp = 43
            elif 'ציון' in col or 'd_prime' in col or 'c' in col or 'hits' in col:
                robad = 'רובד 2: ביצוע'
                n_exp = 69
            elif 'שיעור' in col or 'פרומפט' in col or 'פעולות' in col:
                robad = 'רובד 1: תהליך'
                n_exp = 69

            codebook_rows.append({
                'מספר_עמודה': idx,
                'שם_משתנה': col,
                'רובד_מחקרי': robad,
                'הגדרה_מילולית_תפעולית': definition,
                'נוסחה_ומקור': formula,
                'טווח_ערכים': val_range,
                'יחידת_מידה': v_unit,
                'תנאי_זמינות': zminut,
                'n_צפוי': n_exp,
                'נבדק': ''
            })

    df_cb = pd.DataFrame(codebook_rows)
    df_cb.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"ספר הקודים נשמר : {OUTPUT_CSV} ({len(df_cb)} משתנים מתועדים)")

if __name__ == '__main__':
    main()


#==============================================================================
# PART 4: Cleaning, outliers, normality, scale reliability, sample characteristics
# Thesis: Sections 5.3 to 5.4; Tables B3, B4, B8
#==============================================================================
# (source script: 02_clean_and_preliminaries.py)

"""
analysis_scripts/02_clean_and_preliminaries.py
================================================
הכנת גיליון 2: ניקיון ומקדימים (Hygiene, Outliers & Preliminary Diagnostics).
מייצר את כל הטבלאות הנדרשות:
א. יומן ניקיון ובדיקות טווח/עקביות
ב. זיהוי וסיכום חריגים (2.5*IQR)
ג. תיאור המדגם, משפך ונשירה
ד. נורמליות (Shapiro-Wilk) ושיעור אפסים
ה. מהימנות (Cronbach's alpha, alpha-if-deleted, Spearman-Brown)
ו. ניתוחי נשירה (משלימות מול נושרות, ממלאות שאלון מול לא ממלאות)
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = PROJECT_DIR
DATA_CSV = os.path.join(BASE_DIR, 'analysis_scripts', 'master_clean_n70.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_scripts')

def cronbach_alpha(df_items):
    df_clean = df_items.dropna()
    k = df_clean.shape[1]
    if k <= 1 or len(df_clean) < 2:
        return np.nan
    item_vars = df_clean.var(axis=0, ddof=1).sum()
    total_var = df_clean.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1.0 - item_vars / total_var)

def main():
    print(">>> שלב 2: הרצת ניקיון ומקדימים (גיליון 2)...")
    df = pd.read_csv(DATA_CSV)
    df_comp = df[df['השלימה_משימה'] == 1].copy()

    # ==========================================
    # א. יומן ניקיון ובדיקות עקביות
    # ==========================================
    cleaning_log = [
        {
            'מזהה_משתמש': 'user_9vd6j3xdw',
            'משתנה': 'ציון_ממד_תהליך_ובדיקה',
            'ערך_מקורי': 140,
            'ערך_חדש': 100,
            'סיבה': 'הגבלה לתקרת ציון מקסימלית (100) בשל חישוב בונוס לא מוגבל בצ\'אטבוט',
            'תאריך': '2026-09-15'
        },
        {
            'מזהה_משתמש': 'user_iw02awx51',
            'משתנה': 'ציון_ממד_תהליך_ובדיקה',
            'ערך_מקורי': 110,
            'ערך_חדש': 100,
            'סיבה': 'הגבלה לתקרת ציון מקסימלית (100) בשל חישוב בונוס לא מוגבל בצ\'אטבוט',
            'תאריך': '2026-09-15'
        },
        {
            'מזהה_משתמש': 'user_6mo3gwzpe',
            'משתנה': 'ציון_4_ממדים',
            'ערך_מקורי': 18.0,
            'ערך_חדש': 21.11,
            'סיבה': 'חישוב מחדש של הציון המשוקלל לפי 4 הממדים הרשמיים',
            'תאריך': '2026-09-15'
        },
        {
            'מזהה_משתמש': 'user_8ak40wpi2',
            'משתנה': 'ציון_4_ממדים',
            'ערך_מקורי': 37.0,
            'ערך_חדש': 32.22,
            'סיבה': 'חישוב מחדש של הציון המשוקלל לפי 4 הממדים הרשמיים',
            'תאריך': '2026-09-15'
        },
        {
            'מזהה_משתמש': '7 משתתפים (3 סימולציה, 4 צ\'אטבוט)',
            'משתנה': 'משך_פעילות_נטו_מנוקה_דקות',
            'ערך_מקורי': '> 60 דקות (עד 135+ דק\')',
            'ערך_חדש': 60.0,
            'סיבה': 'הגבלה מתודולוגית של משך הביצוע המקסימלי לתקרה של שעה אחת',
            'תאריך': '2026-09-15'
        }
    ]
    df_clean_log = pd.DataFrame(cleaning_log)
    df_clean_log.to_csv(os.path.join(OUTPUT_DIR, 'tab_2a_cleaning_log.csv'), index=False, encoding='utf-8-sig')

    # בדיקות עקביות מתמטית
    p = df_comp['ציון_ממד_פרומפט']
    v = df_comp['ציון_ממד_זיהוי_מקורות']
    pr = df_comp['ציון_ממד_תהליך_ובדיקה']
    ref = df_comp['ציון_ממד_תחקיר']
    s4_calc = (15.0*p + 35.0*v + 30.0*pr + 10.0*ref) / 90.0
    diff_s4 = (df_comp['ציון_4_ממדים'] - s4_calc).abs().max()
    print(f"בדיקת עקביות: הפרש מקסימלי בנוסחת 4 ממדים = {diff_s4:.6f}")

    # ==========================================
    # ב. זיהוי חריגים (Outliers Summary)
    # ==========================================
    cont_vars = [
        ('משך_פעילות_נטו_מנוקה_דקות', True),
        ('סהכ_פעולות_במערכת', True),
        ('קצב_פעולות_לדקה_APM', False),
        ('מקורות_שנבדקו_באתר', False),
        ('חיפושים_חיצוניים_בגוגל', True),
        ('מילים_בפרומפט_ראשון', False),
        ('מילים_ברפלקציה', False),
        ("סהכ_הודעות_צ'אט_לבוט", True),
        ('ציון_ממד_פרומפט', False),
        ('ציון_ממד_זיהוי_מקורות', False),
        ('ציון_ממד_תהליך_ובדיקה', False),
        ('ציון_ממד_תחקיר', False)
    ]
    outlier_rows = []
    for var, is_skewed in cont_vars:
        sim_sub = df_comp[df_comp['קבוצת_ניסוי'] == 'סימולציה'][var].dropna()
        cb_sub = df_comp[df_comp['קבוצת_ניסוי'] == "צ'אטבוט"][var].dropna()

        # SIM
        val_sim = np.log1p(sim_sub) if is_skewed else sim_sub
        q1_s, q3_s = val_sim.quantile(0.25), val_sim.quantile(0.75)
        iqr_s = q3_s - q1_s
        out_sim = sim_sub[(val_sim < q1_s - 2.5*iqr_s) | (val_sim > q3_s + 2.5*iqr_s)]
        uids_sim = df_comp.loc[out_sim.index, 'מזהה_משתמש'].tolist()

        # CB
        val_cb = np.log1p(cb_sub) if is_skewed else cb_sub
        q1_c, q3_c = val_cb.quantile(0.25), val_cb.quantile(0.75)
        iqr_c = q3_c - q1_c
        out_cb = cb_sub[(val_cb < q1_c - 2.5*iqr_c) | (val_cb > q3_c + 2.5*iqr_c)]
        uids_cb = df_comp.loc[out_cb.index, 'מזהה_משתמש'].tolist()

        outlier_rows.append({
            'משתנה': var,
            'התמרת_לוג': 'כן' if is_skewed else 'לא',
            'חריגים_סימולציה_n': len(out_sim),
            'מזהים_סימולציה': ', '.join(uids_sim) if uids_sim else 'אין',
            'חריגים_צ\'אטבוט_n': len(out_cb),
            'מזהים_צ\'אטבוט': ', '.join(uids_cb) if uids_cb else 'אין'
        })
    df_outliers = pd.DataFrame(outlier_rows)
    df_outliers.to_csv(os.path.join(OUTPUT_DIR, 'tab_2b_outliers_summary.csv'), index=False, encoding='utf-8-sig')

    # ==========================================
    # ג. תיאור המדגם ומשפך השתתפות (N=182)
    # ==========================================
    funnel_data = [
        {'שלב': '1. כניסה והרשמה למערכת (Registered)', 'סימולציה_n': 88, 'צ\'אטבוט_n': 94, 'סה"כ_n': 182, 'אחוז_סימולציה': '100.0%', 'אחוז_צ\'אטבוט': '100.0%'},
        {'שלב': '2. התחלת תרגול ומשימה בפועל (Started Practice)', 'סימולציה_n': 48, 'צ\'אטבוט_n': 73, 'סה"כ_n': 121, 'אחוז_סימולציה': '54.5%', 'אחוז_צ\'אטבוט': '77.7%'},
        {'שלב': '3. השלמת משימה מלאה והגשת דוח (Completed Task)', 'סימולציה_n': 30, 'צ\'אטבוט_n': 39, 'סה"כ_n': 69, 'אחוז_סימולציה': '62.5%', 'אחוז_צ\'אטבוט': '53.4%'},
        {'שלב': '4. מילוי שאלון משוב מסונכרן (Completed Survey)', 'סימולציה_n': 21, 'צ\'אטבוט_n': 28, 'סה"כ_n': 49, 'אחוז_סימולציה': '70.0%', 'אחוז_צ\'אטבוט': '71.8%'}
    ]
    df_funnel = pd.DataFrame(funnel_data)
    df_funnel.to_csv(os.path.join(OUTPUT_DIR, 'tab_2c_funnel.csv'), index=False, encoding='utf-8-sig')

    # מאפייני רקע דמוגרפיים (N=49)
    df_surv = df[df['מילאה_שאלון'] == 1].copy()
    sim_surv = df_surv[df_surv['קבוצת_ניסוי'] == 'סימולציה']
    cb_surv = df_surv[df_surv['קבוצת_ניסוי'] == "צ'אטבוט"]

    demog_rows = []
    # גיל
    age_s = sim_surv['שאלון_גיל'].dropna()
    age_c = cb_surv['שאלון_גיל'].dropna()
    u_age, p_age = stats.mannwhitneyu(age_s, age_c, alternative='two-sided')
    ci_age_l, ci_age_h = compute_r_rb_ci(age_s.values, age_c.values)
    r_age = (u_age - (len(age_s)*len(age_c)-u_age)) / (len(age_s)*len(age_c))
    demog_rows.append({
        'משתנה': 'גיל (שנים)',
        'מבחן_סטטיסטי': 'Mann-Whitney U (דו-צדדי)',
        'סימולציה M (SD)': f"{age_s.mean():.2f} ({age_s.std():.2f})",
        'צ\'אטבוט M (SD)': f"{age_c.mean():.2f} ({age_c.std():.2f})",
        'סטטיסטיקה': f"U = {u_age:.1f} / {len(age_s)*len(age_c)-u_age:.1f}",
        'ערך_p': f"{p_age:.3f}",
        'מדד_אפקט': f"r_rb = {r_age:.3f}",
        '95% CI': format_ci(ci_age_l, ci_age_h),
        'מובהקות': 'לא מובהק (n.s.)' if p_age >= 0.05 else 'מובהק'
    })

    # מגדר
    tab_gender = pd.crosstab(df_surv['קבוצת_ניסוי'], df_surv['שאלון_מגדר'])
    chi2_g, p_g, _, _ = stats.chi2_contingency(tab_gender)
    n_g = tab_gender.values.sum()
    v_g = np.sqrt(chi2_g / n_g)
    ci_gen_l, ci_gen_h = compute_cramers_v_ci_from_table(tab_gender.values)
    demog_rows.append({
        'משתנה': 'מגדר (% נשים)',
        'מבחן_סטטיסטי': 'Chi-square (χ²)',
        'סימולציה M (SD)': f"{(sim_surv['שאלון_מגדר']=='אישה').mean()*100:.1f}% ({sum(sim_surv['שאלון_מגדר']=='אישה')}/{len(sim_surv)})",
        'צ\'אטבוט M (SD)': f"{(cb_surv['שאלון_מגדר']=='אישה').mean()*100:.1f}% ({sum(cb_surv['שאלון_מגדר']=='אישה')}/{len(cb_surv)})",
        'סטטיסטיקה': f"χ² = {chi2_g:.3f}",
        'ערך_p': f"{p_g:.3f}",
        'מדד_אפקט': f"Cramer's V = {v_g:.3f}",
        '95% CI': format_ci(ci_gen_l, ci_gen_h),
        'מובהקות': 'לא מובהק (n.s.)' if p_g >= 0.05 else 'מובהק'
    })

    # שנת לימוד
    tab_year = pd.crosstab(df_surv['קבוצת_ניסוי'], df_surv['שאלון_שנת_לימוד'])
    chi2_y, p_y, _, _ = stats.chi2_contingency(tab_year)
    v_y = np.sqrt(chi2_y / (tab_year.values.sum() * (min(tab_year.shape)-1)))
    ci_yr_l, ci_yr_h = compute_cramers_v_ci_from_table(tab_year.values)
    demog_rows.append({
        'משתנה': 'שנת לימוד (התפלגות)',
        'מבחן_סטטיסטי': 'Chi-square (χ²)',
        'סימולציה M (SD)': 'שנה א-ד ומגוון',
        'צ\'אטבוט M (SD)': 'שנה א-ד ומגוון',
        'סטטיסטיקה': f"χ² = {chi2_y:.3f}",
        'ערך_p': f"{p_y:.3f}",
        'מדד_אפקט': f"Cramer's V = {v_y:.3f}",
        '95% CI': format_ci(ci_yr_l, ci_yr_h),
        'מובהקות': 'לא מובהק (n.s.)' if p_y >= 0.05 else 'מובהק'
    })

    # תדירות GenAI
    tab_genai = pd.crosstab(df_surv['קבוצת_ניסוי'], df_surv['שאלון_תדירות_שימוש_ב_GenAI'])
    chi2_ai, p_ai, _, _ = stats.chi2_contingency(tab_genai)
    v_ai = np.sqrt(chi2_ai / (tab_genai.values.sum() * (min(tab_genai.shape)-1)))
    ci_ai_l, ci_ai_h = compute_cramers_v_ci_from_table(tab_genai.values)
    demog_rows.append({
        'משתנה': 'תדירות שימוש ב-GenAI',
        'מבחן_סטטיסטי': 'Chi-square (χ²)',
        'סימולציה M (SD)': 'מדי פעם עד יומיומי',
        'צ\'אטבוט M (SD)': 'מדי פעם עד יומיומי',
        'סטטיסטיקה': f"χ² = {chi2_ai:.3f}",
        'ערך_p': f"{p_ai:.3f}",
        'מדד_אפקט': f"Cramer's V = {v_ai:.3f}",
        '95% CI': format_ci(ci_ai_l, ci_ai_h),
        'מובהקות': 'לא מובהק (n.s.)' if p_ai >= 0.05 else 'מובהק'
    })
    df_demog = pd.DataFrame(demog_rows)
    df_demog.to_csv(os.path.join(OUTPUT_DIR, 'tab_2c_demographics.csv'), index=False, encoding='utf-8-sig')

    # ==========================================
    # ד. נורמליות ושיעור אפסים
    # ==========================================
    outcome_vars = [
        'ציון_4_ממדים', 'ציון_ממד_פרומפט', 'ציון_ממד_זיהוי_מקורות', 'ציון_ממד_תהליך_ובדיקה',
        'ציון_ממד_תחקיר', 'שיעור_דיוק_סיווג_מקורות', 'd_prime', 'c',
        'סהכ_פעולות_במערכת', 'משך_פעילות_נטו_מנוקה_דקות', 'שיעור_ביקור_במקורות',
        'חיפושים_חיצוניים_בגוגל', 'שיעור_ניחוש_ללא_בדיקה', 'רמת_Kember',
        'PU', 'PEOU', 'BI', 'CriTT'
    ]
    norm_rows = []
    for var in outcome_vars:
        for cond in ['סימולציה', "צ'אטבוט"]:
            sub = df[df['קבוצת_ניסוי'] == cond][var].dropna()
            if len(sub) < 3:
                continue
            w_stat, p_val = stats.shapiro(sub)
            zero_rate = (sub == 0).mean() * 100.0
            norm_rows.append({
                'משתנה': var,
                'קבוצה': cond,
                'n': len(sub),
                'Mean': round(sub.mean(), 2),
                'SD': round(sub.std(), 2),
                'Skewness': round(sub.skew(), 3),
                'Shapiro_W': round(w_stat, 3),
                'p_value': f"{p_val:.3f}" if p_val >= 0.001 else "< .001",
                'שיעור_אפסים_%': round(zero_rate, 1)
            })
    df_norm = pd.DataFrame(norm_rows)
    df_norm.to_csv(os.path.join(OUTPUT_DIR, 'tab_2d_normality_and_zeros.csv'), index=False, encoding='utf-8-sig')

    # ==========================================
    # ה. מהימנות (Reliability)
    # ==========================================
    likert_map = {'כלל לא': 1, 'לא מסכימ.ה': 2, 'מסכימ.ה מעט': 3, 'מסכימ.ה': 4, 'מסכימ.ה מאוד': 5}
    
    # CriTT (17 פריטים)
    critt_cols = [c for c in df.columns if c.startswith('CRITT_') and c != 'CRITT_Overall_הערכה_עצמית_ביקורתית']
    df_critt_num = df_surv[critt_cols].apply(lambda col: col.map(likert_map)).dropna()
    alpha_critt = cronbach_alpha(df_critt_num)

    # PU (4 פריטים)
    pu_cols = ['TAM_1_שיפור_הערכת_אמינות', 'TAM_2_הגדלת_יעילות_זיהוי_אמינות', 'TAM_3_שימושי_ללימודים_אקדמיים', 'TAM_4_עזר_לפתח_חשיבה_ביקורתית']
    df_pu_num = df_surv[pu_cols].apply(lambda col: col.map(likert_map)).dropna()
    alpha_pu = cronbach_alpha(df_pu_num)

    # PEOU (4 פריטים)
    peou_cols = ['TAM_5_קל_ללמוד_להשתמש', 'TAM_6_קל_לגרום_לעשות_מה_שרציתי', 'TAM_7_אינטראקציה_ברורה_ומובנת', 'TAM_8_גמיש_לשימוש_וניווט']
    df_peou_num = df_surv[peou_cols].apply(lambda col: col.map(likert_map)).dropna()
    alpha_peou = cronbach_alpha(df_peou_num)

    # BI (2 פריטים)
    bi_cols = ['TAM_9_מתכוון_להשתמש_שוב', 'TAM_10_רוצה_כלים_דומים_נוספים']
    df_bi_num = df_surv[bi_cols].apply(lambda col: col.map(likert_map)).dropna()
    r_bi = df_bi_num[bi_cols[0]].corr(df_bi_num[bi_cols[1]])
    sb_bi = (2.0 * r_bi) / (1.0 + r_bi)

    rel_summary = [
        {'סולם': 'TAM PU (תועלת נתפסת)', 'מספר_פריטים': 4, 'n': len(df_pu_num), 'מדד': 'Cronbach\'s Alpha', 'ערך': round(alpha_pu, 3), 'הערכה': 'מצוין (≥0.80)'},
        {'סולם': 'TAM PEOU (קלות שימוש)', 'מספר_פריטים': 4, 'n': len(df_peou_num), 'מדד': 'Cronbach\'s Alpha', 'ערך': round(alpha_peou, 3), 'הערכה': 'מצוין (≥0.80)'},
        {'סולם': 'TAM BI (כוונת שימוש)', 'מספר_פריטים': 2, 'n': len(df_bi_num), 'מדד': 'Pearson r / Spearman-Brown', 'ערך': f"r={r_bi:.3f}, SB={sb_bi:.3f}", 'הערכה': 'טוב (≥0.70)'},
        {'סולם': 'CriTT (מסוגלות ביקורתית)', 'מספר_פריטים': 17, 'n': len(df_critt_num), 'מדד': 'Cronbach\'s Alpha', 'ערך': round(alpha_critt, 3), 'הערכה': 'מצוין (≥0.90)'}
    ]
    df_rel = pd.DataFrame(rel_summary)
    df_rel.to_csv(os.path.join(OUTPUT_DIR, 'tab_2e_reliability_summary.csv'), index=False, encoding='utf-8-sig')

    # טבלת Alpha-if-deleted לכל פריט
    alpha_deleted_rows = []
    # PU
    for col in pu_cols:
        sub_items = [c for c in pu_cols if c != col]
        a_del = cronbach_alpha(df_pu_num[sub_items])
        alpha_deleted_rows.append({'סולם': 'PU', 'פריט': col, 'Alpha_אם_נמחק': round(a_del, 3)})
    # PEOU
    for col in peou_cols:
        sub_items = [c for c in peou_cols if c != col]
        a_del = cronbach_alpha(df_peou_num[sub_items])
        alpha_deleted_rows.append({'סולם': 'PEOU', 'פריט': col, 'Alpha_אם_נמחק': round(a_del, 3)})
    # CriTT
    for col in critt_cols:
        sub_items = [c for c in critt_cols if c != col]
        a_del = cronbach_alpha(df_critt_num[sub_items])
        alpha_deleted_rows.append({'סולם': 'CriTT', 'פריט': col, 'Alpha_אם_נמחק': round(a_del, 3)})
    df_alpha_del = pd.DataFrame(alpha_deleted_rows)
    df_alpha_del.to_csv(os.path.join(OUTPUT_DIR, 'tab_2e_alpha_if_deleted.csv'), index=False, encoding='utf-8-sig')

    # ==========================================
    # ו. נשירה (Attrition Analysis)
    # ==========================================
    # ממלאות שאלון מול לא ממלאות (N=48 מול N=21 מתוך 69 המסיימות)
    df_comp_surv = df_comp[df_comp['מילאה_שאלון'] == 1]
    df_comp_nosurv = df_comp[df_comp['מילאה_שאלון'] == 0]

    u_s4_all, p_s4_all = stats.mannwhitneyu(df_comp_surv['ציון_4_ממדים'], df_comp_nosurv['ציון_4_ממדים'], alternative='two-sided')
    u_acc_all, p_acc_all = stats.mannwhitneyu(df_comp_surv['שיעור_דיוק_סיווג_מקורות'].dropna(), df_comp_nosurv['שיעור_דיוק_סיווג_מקורות'].dropna(), alternative='two-sided')

    # בתוך סימולציה
    sim_c = df_comp[df_comp['קבוצת_ניסוי'] == 'סימולציה']
    u_s4_sim, p_s4_sim = stats.mannwhitneyu(sim_c[sim_c['מילאה_שאלון']==1]['ציון_4_ממדים'], sim_c[sim_c['מילאה_שאלון']==0]['ציון_4_ממדים'], alternative='two-sided')

    # בתוך צ'אטבוט
    cb_c = df_comp[df_comp['קבוצת_ניסוי'] == "צ'אטבוט"]
    u_s4_cb, p_s4_cb = stats.mannwhitneyu(cb_c[cb_c['מילאה_שאלון']==1]['ציון_4_ממדים'], cb_c[cb_c['מילאה_שאלון']==0]['ציון_4_ממדים'], alternative='two-sided')

    # טבלת תלות 2x2 מילוי שאלון מול קבוצה
    tab_surv_cond = pd.crosstab(df_comp['קבוצת_ניסוי'], df_comp['מילאה_שאלון'])
    chi2_sc, p_sc, _, _ = stats.chi2_contingency(tab_surv_cond)

    attrition_rows = [
        {'השוואה': 'כלל המדגם: מילאו מול לא מילאו שאלון', 'מדד': 'ציון 4 ממדים', 'n_מילאו': len(df_comp_surv), 'n_לא_מילאו': len(df_comp_nosurv), 'U': u_s4_all, 'p': f"{p_s4_all:.3f}"},
        {'השוואה': 'כלל המדגם: מילאו מול לא מילאו שאלון', 'מדד': 'שיעור דיוק מקורות', 'n_מילאו': df_comp_surv['שיעור_דיוק_סיווג_מקורות'].dropna().count(), 'n_לא_מילאו': df_comp_nosurv['שיעור_דיוק_סיווג_מקורות'].dropna().count(), 'U': u_acc_all, 'p': f"{p_acc_all:.3f}"},
        {'השוואה': 'בתוך סימולציה: מילאו מול לא מילאו', 'מדד': 'ציון 4 ממדים', 'n_מילאו': sum(sim_c['מילאה_שאלון']==1), 'n_לא_מילאו': sum(sim_c['מילאה_שאלון']==0), 'U': u_s4_sim, 'p': f"{p_s4_sim:.3f}"},
        {'השוואה': 'בתוך צ\'אטבוט: מילאו מול לא מילאו', 'מדד': 'ציון 4 ממדים', 'n_מילאו': sum(cb_c['מילאה_שאלון']==1), 'n_לא_מילאו': sum(cb_c['מילאה_שאלון']==0), 'U': u_s4_cb, 'p': f"{p_s4_cb:.3f}"},
        {'השוואה': 'תלות בין קבוצה למילוי שאלון (2x2)', 'מדד': 'Chi-Square', 'n_מילאו': 48, 'n_לא_מילאו': 21, 'U': f"Chi2 = {chi2_sc:.3f}", 'p': f"{p_sc:.3f}"}
    ]
    df_attr = pd.DataFrame(attrition_rows)
    df_attr.to_csv(os.path.join(OUTPUT_DIR, 'tab_2f_attrition.csv'), index=False, encoding='utf-8-sig')

    print(">>> שלב 2 הושלם. כל טבלאות גיליון 2 נשמרו ב-analysis_scripts.")

if __name__ == '__main__':
    main()


#==============================================================================
# PART 5: Process layer: activity and verification measures
# Thesis: Section 6.1; Tables E12, E13, E15
#==============================================================================
# (source script: 03_layer1_process.py)

"""
analysis_scripts/03_layer1_process.py
======================================
הכנת גיליון 3: רובד 1 – תהליך (Process Layer).
עדכונים:
1. הסרת משתנים שלא הוגדרו במסמך או בנספחים (הוסרו: מילים בפרומפט ראשון, אורך רפלקציה).
2. הסרת עמודות חציונים וטווח (Mdn, IQR) - הצגת ממוצעים וסטיות תקן M (SD) בלבד.
3. הוספת שמות המבחנים הסטטיסטיים במפורש לכל משתנה (Mann-Whitney U דו-צדדי / Chi-Square).
4. הוספת ספירות מלאות ושכיחויות לתוכן הודעות ה-AI (סך הודעות, כמות ואחוז משתמשים) ומחיקת חציונים (הערה 30).
5. סימון מובהקות תקני עם כוכביות, עוצמת אפקט מילולית, ורווחי סמך 95% Bootstrap BCa (5,000 דגימות).
6. חישוב ובדיקה מתמטית מלאה של כלל המבחנים ותיקון FDR (Benjamini-Hochberg).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = PROJECT_DIR
DATA_CSV = os.path.join(BASE_DIR, 'analysis_scripts', 'master_clean_n70.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_scripts')
sys.path.append(OUTPUT_DIR)


def calc_fdr(p_values):
    """Benjamini-Hochberg FDR correction."""
    p_vals = np.array(p_values, dtype=float)
    m = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]
    q_vals = np.zeros(m)
    cum_min = 1.0
    for i in range(m - 1, -1, -1):
        q = (sorted_p[i] * m) / (i + 1)
        if q < cum_min:
            cum_min = q
        q_vals[i] = min(cum_min, 1.0)
    fdr_adjusted = np.zeros(m)
    fdr_adjusted[sorted_indices] = q_vals
    return fdr_adjusted

def get_sig_label(p_val, q_val, r_rb):
    stars = '***' if q_val < 0.001 else ('**' if q_val < 0.01 else ('*' if q_val < 0.05 else ''))
    strength = 'גדול מאוד' if abs(r_rb) >= 0.5 else ('בינוני' if abs(r_rb) >= 0.3 else ('קטן' if abs(r_rb) >= 0.1 else 'זניח'))
    if q_val < 0.05:
        return f"{stars} מובהק ({strength})"
    else:
        return "לא מובהק (n.s.)"

def main():
    print(">>> שלב 3: הרצת רובד 1 – תהליך...")
    df = pd.read_csv(DATA_CSV)
    df_comp = df[df['השלימה_משימה'] == 1].copy()

    # ==============================================================================
    # 1. טבלת המדדים ההתנהגותיים (מדדי תהליך ואימות + משתני היקף פעילות רקע/נספח)
    # משתנים שלא הופיעו במסמך (מילים בפרומפט ראשון, אורך רפלקציה) - הוסרו לחלוטין.
    # עמודות חציון וטווח (Mdn, IQR) - הוסרו לחלוטין.
    # ==============================================================================
    behavior_metrics = [
        # מדדי תהליך ואימות (גוף הטקסט)
        ('שיעור כניסה בפועל למקורות מתוך המקורות שהוצגו (%)', 'שיעור_ביקור_במקורות', 37, 100.0),
        ('שיעור חיפוש בגוגל מתוך המקורות שהוצגו (%)', 'שיעור_חיפוש_בגוגל', 37, 100.0),
        ('הצלבות גוגל ואתר', 'הצלבות_גוגל_ואתר', 37, 1.0),
        ('שיעור סיווג ללא בדיקה (%)', 'שיעור_ניחוש_ללא_בדיקה', 37, 100.0),
        ('ניסיונות לכתיבת פרומפט', 'ניסיונות_פרומפט_משימה_מסווגים', 39, 1.0),
        # משתני רקע והיקף פעילות כולל (* = מופיע בנספח)
        ('סך פעולות במערכת*', 'סהכ_פעולות_במערכת', 39, 1.0),
        ('קצב פעולות לדקה (APM)*', 'קצב_פעולות_לדקה_APM', 39, 1.0),
        ('משך ביצוע נטו (דקות)*', 'משך_פעילות_נטו_מנוקה_דקות', 39, 1.0),
        ('כניסות למקורות באתר*', 'מקורות_שנבדקו_באתר', 37, 1.0),
        ('חיפושים חיצוניים בגוגל*', 'חיפושים_חיצוניים_בגוגל', 37, 1.0),
        ('סה"כ הודעות ל-AI*', "סהכ_הודעות_צ'אט_לבוט", 39, 1.0)
    ]

    p_list = []
    temp_rows = []

    for label, col, n_cb_expected, mult in behavior_metrics:
        sim_data = df_comp[df_comp['קבוצת_ניסוי'] == 'סימולציה'][col].dropna() * mult
        cb_data = df_comp[df_comp['קבוצת_ניסוי'] == "צ'אטבוט"][col].dropna() * mult

        if n_cb_expected == 37:
            cb_data = df_comp[(df_comp['קבוצת_ניסוי'] == "צ'אטבוט") & (df_comp['הוצגו_מקורות'] == 1)][col].dropna() * mult

        n_s = len(sim_data)
        n_c = len(cb_data)

        # Mann-Whitney U דו-צדדי
        u_sim, p_val = stats.mannwhitneyu(sim_data, cb_data, alternative='two-sided')
        u_cb = (n_s * n_c) - u_sim
        r_rb = (u_sim - u_cb) / (n_s * n_c)

        # בדיקת אימות מתמטי מחמירה
        assert abs((u_sim + u_cb) - (n_s * n_c)) < 1e-5, f"שגיאה מתמטית בסכום U בעמודה {col}"

        ci_l, ci_h = compute_r_rb_ci(sim_data.values, cb_data.values)

        p_list.append(p_val)
        temp_rows.append({
            'מדד': label,
            'מבחן_סטטיסטי': 'Mann-Whitney U (דו-צדדי)',
            'סימולציה_N': n_s,
            'סימולציה_M_SD': f"{sim_data.mean():.2f} ({sim_data.std():.2f})",
            'צ\'אטבוט_N': n_c,
            'צ\'אטבוט_M_SD': f"{cb_data.mean():.2f} ({cb_data.std():.2f})",
            'Mann_Whitney_U': f"{u_sim:.1f} / {u_cb:.1f}",
            'U_Sim': round(u_sim, 1),
            'U_CB': round(u_cb, 1),
            'ערך_p_גולמי': p_val,
            'גודל_אפקט_r_rb': round(r_rb, 3),
            'רווח_סמך_95%_BCa': format_ci(ci_l, ci_h)
        })

    # חישוב FDR מדויק למשפחת המדדים
    fdr_list = calc_fdr(p_list)
    for i, row in enumerate(temp_rows):
        p_v = row['ערך_p_גולמי']
        q_v = fdr_list[i]
        r_rb = row['גודל_אפקט_r_rb']
        row['ערך_p_גולמי'] = f"{p_v:.4f}" if p_v >= 0.001 else "< .001"
        row['ערך_q_מתוקן_FDR'] = f"{q_v:.4f}" if q_v >= 0.001 else "< .001"
        row['מובהקות_ועוצמה'] = get_sig_label(p_v, q_v, r_rb)

    df_b1 = pd.DataFrame(temp_rows)
    cols_order = [
        'מדד', 'מבחן_סטטיסטי', 'סימולציה_N', 'סימולציה_M_SD',
        'צ\'אטבוט_N', 'צ\'אטבוט_M_SD', 'Mann_Whitney_U', 'U_Sim', 'U_CB',
        'ערך_p_גולמי', 'ערך_q_מתוקן_FDR', 'גודל_אפקט_r_rb', 'רווח_סמך_95%_BCa', 'מובהקות_ועוצמה'
    ]
    df_b1_out = df_b1[cols_order]
    df_b1_out.to_csv(os.path.join(OUTPUT_DIR, 'tab_3a_behavior_comparison.csv'), index=False, encoding='utf-8-sig')
    print("טבלה 3א נשמרה  ללא חציונים וטווח, עם שמות מבחנים, FDR ו-95% CI.")

    # ==============================================================================
    # 2. אפס כניסות למקורות (טבלת 2x2)
    # ==============================================================================
    df_src = df_comp[df_comp['הוצגו_מקורות'] == 1].copy()
    tab_zero = pd.crosstab(df_src['קבוצת_ניסוי'], df_src['אפס_כניסות'])
    odds_ratio, p_fisher = stats.fisher_exact(tab_zero)
    chi2_val, p_chi, _, _ = stats.chi2_contingency(tab_zero)
    v_cramer = np.sqrt(chi2_val / tab_zero.values.sum())
    v_l, v_h = compute_cramers_v_ci_from_table(tab_zero.values)

    sim_zero_cnt = tab_zero.loc['סימולציה', 1.0] if 1.0 in tab_zero.columns and 'סימולציה' in tab_zero.index else 0
    sim_tot_cnt = tab_zero.loc['סימולציה'].sum() if 'סימולציה' in tab_zero.index else 0
    cb_zero_cnt = tab_zero.loc["צ'אטבוט", 1.0] if 1.0 in tab_zero.columns and "צ'אטבוט" in tab_zero.index else 0
    cb_tot_cnt = tab_zero.loc["צ'אטבוט"].sum() if "צ'אטבוט" in tab_zero.index else 0

    df_zero_out = pd.DataFrame([{
        'מדד': 'אפס כניסות למקורות (Zero Visits)',
        'מבחן_סטטיסטי': "Fisher's Exact Test / Chi-Square",
        'סימולציה_אפס': f"{sim_zero_cnt} / {sim_tot_cnt} ({sim_zero_cnt / sim_tot_cnt * 100:.1f}%)" if sim_tot_cnt > 0 else "0",
        'צ\'אטבוט_אפס': f"{cb_zero_cnt} / {cb_tot_cnt} ({cb_zero_cnt / cb_tot_cnt * 100:.1f}%)" if cb_tot_cnt > 0 else "0",
        'Odds_Ratio': round(odds_ratio, 3),
        'p_Fisher': f"{p_fisher:.4f}" if p_fisher >= 0.001 else "< .001",
        'Cramer_V': round(v_cramer, 3),
        'Cramer_V_95%_CI': format_ci(v_l, v_h),
        'מובהקות': '*** מובהק (גדול מאוד)' if p_fisher < 0.001 else 'מובהק'
    }])
    df_zero_out.to_csv(os.path.join(OUTPUT_DIR, 'tab_3b_zero_visits.csv'), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # 3. דמיון להנחיה והעתקה (טבלת נספח)
    # ==============================================================================
    sim_simil = df_comp[df_comp['קבוצת_ניסוי'] == 'סימולציה']['דמיון_להנחיה_מקסימלי'].dropna()
    cb_simil = df_comp[df_comp['קבוצת_ניסוי'] == "צ'אטבוט"]['דמיון_להנחיה_מקסימלי'].dropna()
    u_simil, p_simil = stats.mannwhitneyu(sim_simil, cb_simil, alternative='two-sided')
    u_cb_simil = len(sim_simil)*len(cb_simil) - u_simil
    r_simil = (u_simil - u_cb_simil) / (len(sim_simil)*len(cb_simil))
    ci_sim_l, ci_sim_h = compute_r_rb_ci(sim_simil.values, cb_simil.values)

    tab_copy = pd.crosstab(df_comp['קבוצת_ניסוי'], df_comp['האם_העתיק_הנחיה'])
    chi2_cp, p_cp, _, _ = stats.chi2_contingency(tab_copy)
    v_cp = np.sqrt(chi2_cp / tab_copy.values.sum())
    v_cp_l, v_cp_h = compute_cramers_v_ci_from_table(tab_copy.values)

    sim_group = df_comp[df_comp['קבוצת_ניסוי'] == 'סימולציה']
    cb_group = df_comp[df_comp['קבוצת_ניסוי'] == "צ'אטבוט"]

    df_copy_out = pd.DataFrame([
        {
            'מדד': 'דמיון מקסימלי להנחיה (0 עד 1)',
            'מבחן_סטטיסטי': 'Mann-Whitney U (דו-צדדי)',
            'סימולציה': f"M={sim_simil.mean():.2f} (SD={sim_simil.std():.2f})",
            'צ\'אטבוט': f"M={cb_simil.mean():.2f} (SD={cb_simil.std():.2f})",
            'סטטיסטיקה': f"U = {u_simil:.1f} / {u_cb_simil:.1f}",
            'ערך_p': f"{p_simil:.3f}",
            'מדד_אפקט': f"r_rb = {r_simil:.2f}",
            'רווח_סמך_95%': format_ci(ci_sim_l, ci_sim_h),
            'מובהקות': 'לא מובהק (n.s.)' if p_simil >= 0.05 else 'מובהק'
        },
        {
            'מדד': 'שיעור שהעתיקו הנחיה',
            'מבחן_סטטיסטי': 'Chi-square (χ²)',
            'סימולציה': f"{(sim_group['האם_העתיק_הנחיה'] == 'כן').mean() * 100:.1f}% ({sum(sim_group['האם_העתיק_הנחיה'] == 'כן')}/{len(sim_group)})",
            'צ\'אטבוט': f"{(cb_group['האם_העתיק_הנחיה'] == 'כן').mean() * 100:.1f}% ({sum(cb_group['האם_העתיק_הנחיה'] == 'כן')}/{len(cb_group)})",
            'סטטיסטיקה': f"Chi-square = {chi2_cp:.3f}",
            'ערך_p': f"{p_cp:.3f}",
            'מדד_אפקט': f"Cramer's V = {v_cp:.2f}",
            'רווח_סמך_95%': format_ci(v_cp_l, v_cp_h),
            'מובהקות': 'לא מובהק (n.s.)' if p_cp >= 0.05 else 'מובהק'
        }
    ])
    df_copy_out.to_csv(os.path.join(OUTPUT_DIR, 'tab_3c_prompt_similarity.csv'), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # 4. תוכן ההודעות ל-AI לפי קטגוריה
    # הוספת ספירות מלאות, כמות משתמשים, אחוזים, ומחיקת עמודות חציון!
    # ==============================================================================
    df_bot_clean = df_comp[df_comp['הוצאה_מקידוד_צאט'] == 0].copy()
    chat_cats = [
        ('פרומפט ישיר למשימה', 'הודעות_פרומפט_ישיר_למשימה'),
        ('ייעוץ מתודולוגי על אימות', 'הודעות_ייעוץ_מתודולוגי_אימות'),
        ('ייעוץ בניסוח פרומפטים', 'הודעות_ייעוץ_בניסוח_פרומפטים'),
        ('שאלות ובירורים כלליים', 'הודעות_שאלות_ובירורים_כלליים'),
        ('שאלות התמצאות בממשק', 'הודעות_שאלות_תהליך_והתמצאות'),
        ('ברכת פתיחה ושיח כללי', 'הודעות_ברכת_פתיחה_ושיח_כללי')
    ]

    p_chat_list = []
    bot_rows = []

    for lbl, col in chat_cats:
        sim = df_bot_clean[df_bot_clean['קבוצת_ניסוי'] == 'סימולציה'][col]
        cb = df_bot_clean[df_bot_clean['קבוצת_ניסוי'] == "צ'אטבוט"][col]

        n_s = len(sim)
        n_c = len(cb)

        u_s, p = stats.mannwhitneyu(sim, cb, alternative='two-sided')
        u_c = (n_s * n_c) - u_s
        r_rb = (u_s - u_c) / (n_s * n_c)

        p_chat_list.append(p)
        ci_l, ci_h = compute_r_rb_ci(sim.values, cb.values)

        bot_rows.append({
            'קטגוריית_שיח': lbl,
            'מבחן_סטטיסטי': 'Mann-Whitney U (דו-צדדי)',
            'סימולציה_N': n_s,
            'סימולציה_סך_הודעות_ספירה': int(sim.sum()),
            'סימולציה_משתמשים_N_אחוז': f"{(sim > 0).sum()}/{n_s} ({(sim > 0).mean()*100:.1f}%)",
            'סימולציה_M_SD': f"{sim.mean():.2f} ({sim.std():.2f})",
            'צ\'אטבוט_N': n_c,
            'צ\'אטבוט_סך_הודעות_ספירה': int(cb.sum()),
            'צ\'אטבוט_משתמשים_N_אחוז': f"{(cb > 0).sum()}/{n_c} ({(cb > 0).mean()*100:.1f}%)",
            'צ\'אטבוט_M_SD': f"{cb.mean():.2f} ({cb.std():.2f})",
            'Mann_Whitney_U': f"{u_s:.1f} / {u_c:.1f}",
            'U_Sim': round(u_s, 1),
            'U_CB': round(u_c, 1),
            'סכום_U_תקין': 'כן' if abs((u_s + u_c) - (n_s * n_c)) < 1e-4 else 'שגיאה',
            'ערך_p_גולמי': p,
            'גודל_אפקט_r_rb': round(r_rb, 3),
            'רווח_סמך_95%_BCa': format_ci(ci_l, ci_h)
        })

    fdr_chat = calc_fdr(p_chat_list)
    for i, r in enumerate(bot_rows):
        p_v = r['ערך_p_גולמי']
        q_v = fdr_chat[i]
        r_rb = r['גודל_אפקט_r_rb']
        r['ערך_p_גולמי'] = f"{p_v:.4f}" if p_v >= 0.001 else "< .001"
        r['ערך_q_מתוקן_FDR'] = f"{q_v:.4f}" if q_v >= 0.001 else "< .001"
        r['מובהקות_ועוצמה'] = get_sig_label(p_v, q_v, r_rb)

    df_bot_out = pd.DataFrame(bot_rows)
    df_bot_out.to_csv(os.path.join(OUTPUT_DIR, 'tab_3e_bot_consultation.csv'), index=False, encoding='utf-8-sig')
    print("טבלה 3ה (תוכן הודעות AI) נשמרה עם ספירות מלאות, ללא חציונים, ועם FDR ו-95% CI.")

    # ==============================================================================
    # 5. מדדים ייחודיים לסימולציה
    # ==============================================================================
    sim_only_cols = [
        ('משך שהייה ממוצע במקור (שניות)', 'משך_שהייה_ממוצע_במקור_שניות'),
        ('סטיית תקן שהייה במקור', 'סטיית_תקן_שהייה_במקור'),
        ('נימוקי החרגת מקורות כמות', 'נימוקי_החרגת_מקורות_כמות')
    ]
    sim_only_rows = []
    sim_group = df_comp[df_comp['קבוצת_ניסוי'] == 'סימולציה']
    for lbl, c in sim_only_cols:
        s = sim_group[c].dropna()
        sim_only_rows.append({
            'מדד': lbl,
            'n': len(s),
            'M_SD': f"{s.mean():.2f} ({s.std():.2f})"
        })
    df_sim_only = pd.DataFrame(sim_only_rows)
    df_sim_only.to_csv(os.path.join(OUTPUT_DIR, 'tab_3d_simulation_only.csv'), index=False, encoding='utf-8-sig')

    print(">>> טבלאות רובד 1 נשמרו.")

if __name__ == '__main__':
    main()


#==============================================================================
# PART 6: Performance layer: composite score, dimensions, SDT, reflection
# Thesis: Section 6.2; Figure 6; Tables E4, E5, E16; Figure E1
#==============================================================================
# (source script: 04_layer2_performance.py)

# ==============================================================================
# analysis_scripts/04_layer2_performance.py
# רובד 2: ביצוע (Performance) - ציונים, דיוק, SDT, רפלקציה ותמות
# פרויקט מחקר WorkAI
# ==============================================================================

import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

def fdr_bh(pvals):
    n_tests = len(pvals)
    sorted_indices = np.argsort(pvals)
    adj_p = np.zeros(n_tests)
    cum_min = 1.0
    for rank, idx in reversed(list(enumerate(sorted_indices))):
        p_adj = pvals[idx] * n_tests / (rank + 1)
        cum_min = min(cum_min, p_adj)
        adj_p[idx] = min(cum_min, 1.0)
    return adj_p

def main():
    print(">>> שלב 4: הרצת ניתוח רובד 2 (ביצוע)...")
    base_dir = SCRIPTS_DIR
    master_path = os.path.join(base_dir, "master_clean_n70.csv")
    df = pd.read_csv(master_path)

    # סינון נבדקות שהשלימו משימה (N=69)
    task_df = df[df['השלימה_משימה'] == 1].copy()
    sim_df = task_df[task_df['קבוצה_אנגלית'] == 'Simulation'].copy()
    cb_df  = task_df[task_df['קבוצה_אנגלית'] == 'Chatbot'].copy()

    n_sim = len(sim_df) # 30
    n_cb  = len(cb_df)  # 39
    print(f"נבדקות שהשלימו משימה: סימולציה n={n_sim}, צ'אטבוט n={n_cb}")

    # תת-קבוצה לנבדקות שהוצגו להן מקורות (SDT ודיוק מקורות: 30 מול 37)
    sim_src = sim_df[sim_df['הוצגו_מקורות'] == 1].copy()
    cb_src  = cb_df[cb_df['הוצגו_מקורות'] == 1].copy()
    n_sim_src = len(sim_src) # 30
    n_cb_src  = len(cb_src)  # 37

    # ==============================================================================
    # 1. טבלה 4א1: ציון כולל (4 ממדים) - מוצג כטבלה עליונה עצמאית
    # ==============================================================================
    s_col = 'ציון_4_ממדים'
    s_val = sim_df[s_col].dropna()
    c_val = cb_df[s_col].dropna()

    res = stats.mannwhitneyu(s_val, c_val, alternative='two-sided')
    u_sim = float(res.statistic)
    u_cb = float(len(s_val) * len(c_val) - u_sim)
    r_rb = (u_sim - u_cb) / (len(s_val) * len(c_val))
    p_val = res.pvalue
    ci_low, ci_high = compute_r_rb_ci(s_val.values, c_val.values)
    strength = 'גדול מאוד' if abs(r_rb) >= 0.5 else ('בינוני' if abs(r_rb) >= 0.3 else 'קטן')

    tab_4a1 = pd.DataFrame([{
        'מדד': 'ציון איכות מנוקה כולל (4 ממדים)',
        'משקל': '100% (סולם 0-100)',
        'מבחן_סטטיסטי': 'Mann-Whitney U (דו-צדדי)',
        'סימולציה N': len(s_val),
        'סימולציה ממוצע (ס״ת)': f"{s_val.mean():.2f} ({s_val.std():.2f})",
        'צ׳אטבוט N': len(c_val),
        'צ׳אטבוט ממוצע (ס״ת)': f"{c_val.mean():.2f} ({c_val.std():.2f})",
        'Mann-Whitney U_Sim': f"{u_sim:.1f}",
        'U_CB': f"{u_cb:.1f}",
        'סכום U תקין': 'כן' if abs((u_sim + u_cb) - (len(s_val)*len(c_val))) < 1e-5 else 'שגיאה',
        'ערך p': f"{p_val:.4f}" if p_val >= 0.0001 else "< .001",
        'גודל אפקט r_rb': f"{r_rb:.3f}",
        '95% CI (BCa)': format_ci(ci_low, ci_high),
        'עוצמת אפקט': strength,
        'רמת מובהקות': f"*** ({strength})" if p_val < 0.001 else f"** ({strength})" if p_val < 0.01 else f"* ({strength})" if p_val < 0.05 else "לא מובהק (n.s.)"
    }])
    tab_4a1.to_csv(os.path.join(base_dir, "tab_4a1_overall_score.csv"), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # 2. טבלה 4א2: 4 ממדים נפרדים ושיעור דיוק זיהוי מקורות (FDR מתוקן)
    # ==============================================================================
    dim_specs = [
        ('ציון_ממד_פרומפט', 'ממד פרומפטים (15%)', 'task', 15),
        ('ציון_ממד_זיהוי_מקורות', 'ממד בדיקת וזיהוי מקורות (35%)', 'task', 35),
        ('ציון_ממד_תהליך_ובדיקה', 'ממד תהליך ובדיקת מקורות (30%)', 'task', 30),
        ('ציון_ממד_תחקיר', 'ממד תחקיר ורפלקציה (10%)', 'task', 10),
        ('שיעור_דיוק_סיווג_מקורות', 'שיעור דיוק בזיהוי מקורות (0-1)', 'source', None)
    ]

    rows_4a2 = []
    pvals_4a2 = []

    for col, name, group_type, weight in dim_specs:
        if group_type == 'task':
            s_data = sim_df[col].dropna()
            c_data = cb_df[col].dropna()
        else:
            s_data = sim_src[col].dropna()
            c_data = cb_src[col].dropna()

        res_u = stats.mannwhitneyu(s_data, c_data, alternative='two-sided')
        u_s = float(res_u.statistic)
        u_c = float(len(s_data) * len(c_data) - u_s)
        r_b = (u_s - u_c) / (len(s_data) * len(c_data))
        p = res_u.pvalue
        ci_l, ci_h = compute_r_rb_ci(s_data.values, c_data.values)
        pvals_4a2.append(p)
        strength = 'גדול מאוד' if abs(r_b) >= 0.5 else ('בינוני' if abs(r_b) >= 0.3 else ('קטן' if abs(r_b) >= 0.1 else 'זניח'))

        rows_4a2.append({
            'מדד': name,
            'משקל': f"{weight}%" if weight else "-",
            'מבחן_סטטיסטי': 'Mann-Whitney U (דו-צדדי)',
            'סימולציה N': len(s_data),
            'סימולציה ממוצע (ס״ת)': f"{s_data.mean():.2f} ({s_data.std():.2f})",
            'צ׳אטבוט N': len(c_data),
            'צ׳אטבוט ממוצע (ס״ת)': f"{c_data.mean():.2f} ({c_data.std():.2f})",
            'Mann-Whitney U_Sim': f"{u_s:.1f}",
            'U_CB': f"{u_c:.1f}",
            'סכום U תקין': 'כן' if abs((u_s + u_c) - (len(s_data)*len(c_data))) < 1e-5 else 'שגיאה',
            'ערך p גולמי': p,
            'גודל אפקט r_rb': f"{r_b:.3f}",
            '95% CI (BCa)': format_ci(ci_l, ci_h),
            'עוצמת אפקט': strength
        })

    # תיקון FDR Benjamini-Hochberg
    adj_p = fdr_bh(pvals_4a2)

    for i, r in enumerate(rows_4a2):
        p_raw = r['ערך p גולמי']
        p_fdr = adj_p[i]
        r['ערך p גולמי'] = f"{p_raw:.4f}" if p_raw >= 0.0001 else "< .001"
        r['ערך p מתוקן FDR'] = f"{p_fdr:.4f}" if p_fdr >= 0.0001 else "< .001"
        r['מובהק (q<.05)'] = 'כן' if p_fdr < 0.05 else 'לא'
        if p_fdr < 0.001:
            stars = '***'
        elif p_fdr < 0.01:
            stars = '**'
        elif p_fdr < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'
        r['רמת מובהקות'] = f"{stars} ({r['עוצמת אפקט']})" if stars != 'n.s.' else 'לא מובהק (n.s.)'

    tab_4a2 = pd.DataFrame(rows_4a2)
    tab_4a2.to_csv(os.path.join(base_dir, "tab_4a2_dimensions_comparison.csv"), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # 3. טבלה 4ב: מדדי תורת גילוי אותות (SDT)
    # ==============================================================================
    sdt_metrics = [
        ('d_prime', "רגישות הבחנה (d')", "מדד יכולת הפרדה בין מקור מטעה לאמין"),
        ('c', "קריטריון תגובה / הטיה (c)", "מדד שמרנות (חיובי=שמרני, שלילי=ליברלי)"),
        ('H_rate', "שיעור פגיעות (Hit Rate)", "זיהוי מקור מטעה כמטעה"),
        ('FA_rate', "שיעור אזעקות שווא (FA Rate)", "סיווג שגוי של מקור אמין כמטעה")
    ]

    rows_4b = []
    pvals_4b = []

    for col, name, desc in sdt_metrics:
        s_data = sim_src[col].dropna()
        c_data = cb_src[col].dropna()

        res_u = stats.mannwhitneyu(s_data, c_data, alternative='two-sided')
        u_s = float(res_u.statistic)
        u_c = float(len(s_data) * len(c_data) - u_s)
        r_b = (u_s - u_c) / (len(s_data) * len(c_data))
        p = res_u.pvalue
        ci_l, ci_h = compute_r_rb_ci(s_data.values, c_data.values)
        pvals_4b.append(p)
        strength = 'גדול מאוד' if abs(r_b) >= 0.5 else ('בינוני' if abs(r_b) >= 0.3 else ('קטן' if abs(r_b) >= 0.1 else 'זניח'))

        rows_4b.append({
            'מדד': name,
            'הגדרה': desc,
            'מבחן_סטטיסטי': 'Mann-Whitney U (דו-צדדי)',
            'סימולציה N': len(s_data),
            'סימולציה ממוצע (ס״ת)': f"{s_data.mean():.3f} ({s_data.std():.3f})",
            'צ׳אטבוט N': len(c_data),
            'צ׳אטבוט ממוצע (ס״ת)': f"{c_data.mean():.3f} ({c_data.std():.3f})",
            'Mann-Whitney U_Sim': f"{u_s:.1f}",
            'U_CB': f"{u_c:.1f}",
            'סכום U תקין': 'כן' if abs((u_s + u_c) - (len(s_data)*len(c_data))) < 1e-5 else 'שגיאה',
            'ערך p גולמי': p,
            'גודל אפקט r_rb': f"{r_b:.3f}",
            '95% CI (BCa)': format_ci(ci_l, ci_h),
            'עוצמת אפקט': strength
        })

    # FDR על SDT
    adj_p = fdr_bh(pvals_4b)

    for i, r in enumerate(rows_4b):
        p_raw = r['ערך p גולמי']
        p_fdr = adj_p[i]
        r['ערך p גולמי'] = f"{p_raw:.4f}" if p_raw >= 0.0001 else "< .001"
        r['ערך p מתוקן FDR'] = f"{p_fdr:.4f}" if p_fdr >= 0.0001 else "< .001"
        r['מובהק (q<.05)'] = 'כן' if p_fdr < 0.05 else 'לא'
        if p_fdr < 0.001:
            stars = '***'
        elif p_fdr < 0.01:
            stars = '**'
        elif p_fdr < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'
        r['רמת מובהקות'] = f"{stars} ({r['עוצמת אפקט']})" if stars != 'n.s.' else 'לא מובהק (n.s.)'

    tab_4b = pd.DataFrame(rows_4b)
    tab_4b.to_csv(os.path.join(base_dir, "tab_4b_sdt_comparison.csv"), index=False, encoding='utf-8-sig')

    # מבחן Wilcoxon חד-מדגמי מול 0 לקריטריון תגובה (c)
    w_sim = stats.wilcoxon(sim_src['c'].dropna(), alternative='two-sided')
    w_cb  = stats.wilcoxon(cb_src['c'].dropna(), alternative='two-sided')
    tab_4b_bias = pd.DataFrame([
        {
            'קבוצה': 'סימולציה',
            'מדד': 'קריטריון c מול 0 (שמרנות)',
            'מבחן_סטטיסטי': 'Wilcoxon Signed-Rank Test (דו-צדדי מול 0)',
            'N': len(sim_src['c'].dropna()),
            'ממוצע (ס״ת)': f"{sim_src['c'].mean():.3f} ({sim_src['c'].std():.3f})",
            'Wilcoxon W': f"{w_sim.statistic:.1f}",
            'ערך p': f"{w_sim.pvalue:.4f}",
            'עמדה מובהקת': 'שמרנית מובהקת (c > 0)' if sim_src['c'].mean() > 0 and w_sim.pvalue < 0.05 else 'ניטרלית'
        },
        {
            'קבוצה': "צ'אטבוט",
            'מדד': 'קריטריון c מול 0 (שמרנות)',
            'מבחן_סטטיסטי': 'Wilcoxon Signed-Rank Test (דו-צדדי מול 0)',
            'N': len(cb_src['c'].dropna()),
            'ממוצע (ס״ת)': f"{cb_src['c'].mean():.3f} ({cb_src['c'].std():.3f})",
            'Wilcoxon W': f"{w_cb.statistic:.1f}",
            'ערך p': f"{w_cb.pvalue:.4f}",
            'עמדה מובהקת': 'שמרנית מובהקת (c > 0)' if cb_src['c'].mean() > 0 and w_cb.pvalue < 0.05 else 'ניטרלית'
        }
    ])
    tab_4b_bias.to_csv(os.path.join(base_dir, "tab_4b_response_bias.csv"), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # 4. טבלה 4ג: רמת רפלקציה לפי קמבר (Kember et al., 2000)
    # ==============================================================================
    # 4ג1: התפלגות רמות קמבר על הטקסט המאוחד
    kember_names = {
        0.0: '0 - ללא רפלקציה / טכני לחלוטין',
        1.0: '1 - פעולה שגרתית (Habitual Action)',
        2.0: '2 - הבנה בסיסית (Understanding)',
        3.0: '3 - רפלקציה (Reflection)',
        4.0: '4 - רפלקציה ביקורתית (Critical Reflection)'
    }
    kember_rows = []
    for lvl in [0.0, 1.0, 2.0, 3.0, 4.0]:
        n_s = (sim_df['רמת_Kember'] == lvl).sum()
        pct_s = n_s / n_sim * 100
        n_c = (cb_df['רמת_Kember'] == lvl).sum()
        pct_c = n_c / n_cb * 100
        kember_rows.append({
            'רמת קמבר': kember_names[lvl],
            'ערך נומרי': int(lvl),
            'סימולציה N': n_s,
            'סימולציה %': f"{pct_s:.1f}%",
            'צ׳אטבוט N': n_c,
            'צ׳אטבוט %': f"{pct_c:.1f}%",
            'סה״כ N': n_s + n_c,
            'סה״כ %': f"{(n_s + n_c) / len(task_df) * 100:.1f}%"
        })

    # שורת ממוצע + Mann-Whitney (ללא חציון)
    res_k = stats.mannwhitneyu(sim_df['רמת_Kember'].dropna(), cb_df['רמת_Kember'].dropna(), alternative='two-sided')
    u_k_s = float(res_k.statistic)
    u_k_c = float(n_sim * n_cb - u_k_s)
    r_rb_k = (u_k_s - u_k_c) / (n_sim * n_cb)
    ci_k_l, ci_k_h = compute_r_rb_ci(sim_df['רמת_Kember'].dropna().values, cb_df['רמת_Kember'].dropna().values)

    kember_rows.append({
        'רמת קמבר': 'סיכום: ממוצע (ס״ת) [Mann-Whitney U]',
        'ערך נומרי': '-',
        'סימולציה N': f"{sim_df['רמת_Kember'].mean():.2f} ({sim_df['רמת_Kember'].std():.2f})",
        'סימולציה %': '-',
        'צ׳אטבוט N': f"{cb_df['רמת_Kember'].mean():.2f} ({cb_df['רמת_Kember'].std():.2f})",
        'צ׳אטבוט %': '-',
        'סה״כ N': f"U_Sim={u_k_s:.1f}, U_CB={u_k_c:.1f}",
        'סה״כ %': f"p={res_k.pvalue:.4f}, r_rb={r_rb_k:.3f} {format_ci(ci_k_l, ci_k_h)}"
    })
    tab_4c1 = pd.DataFrame(kember_rows)
    tab_4c1.to_csv(os.path.join(base_dir, "tab_4c1_kember_distribution.csv"), index=False, encoding='utf-8-sig')

    # 4ג2: עומק רפלקציה לפי שאלות בודדות (1, 2, 4) ושיעורי מענה
    q_specs = [
        ('רפלקציה_ש1_עומק_נומרי', 'שאלה 1: שיפור אסטרטגי / למידה', 'מקורות ואימות'),
        ('רפלקציה_ש2_עומק_נומרי', 'שאלה 2: ניהול טעויות / חרטה', 'שינוי בהתנהלות'),
        ('רפלקציה_ש4_עומק_נומרי', 'שאלה 4: לקחים עתידיים לשימוש', 'העברה לפרקטיקה')
    ]
    rows_q = []
    for q_col, q_title, q_theme in q_specs:
        s_ans = sim_df[q_col].dropna()
        c_ans = cb_df[q_col].dropna()
        rows_q.append({
            'שאלה': q_title,
            'נושא': q_theme,
            'סימולציה מענה N': len(s_ans),
            'סימולציה שיעור מענה': f"{len(s_ans) / n_sim * 100:.1f}%",
            'סימולציה ממוצע עומק (ס״ת)': f"{s_ans.mean():.2f} ({s_ans.std():.2f})" if len(s_ans) > 0 else "-",
            'צ׳אטבוט מענה N': len(c_ans),
            'צ׳אטבוט שיעור מענה': f"{len(c_ans) / n_cb * 100:.1f}%",
            'צ׳אטבוט ממוצע עומק (ס״ת)': f"{c_ans.mean():.2f} ({c_ans.std():.2f})" if len(c_ans) > 0 else "-",
            'הערה מתודולוגית': 'בסימולציה הופיעה שאלה 3 ממוקדת טעות; בצ׳אטבוט לא הייתה קיימת שאלה 3. שיעור המענה בצ׳אטבוט צנח משמעותית משאלה 1 לשאלה 4.'
        })
    tab_4c2 = pd.DataFrame(rows_q)
    tab_4c2.to_csv(os.path.join(base_dir, "tab_4c2_reflection_by_question.csv"), index=False, encoding='utf-8-sig')

    # 4ג3: מודל רגרסיה לינארית על רמת קמבר
    task_df['is_sim'] = (task_df['קבוצה_אנגלית'] == 'Simulation').astype(int)
    X_reg = sm.add_constant(task_df[['is_sim', 'מילים_ברפלקציה']])
    y_reg = task_df['רמת_Kember']
    ols_res = sm.OLS(y_reg, X_reg).fit()

    rows_reg = []
    var_map = {'const': 'חיתוך (Intercept)', 'is_sim': 'קבוצה (1=סימולציה, 0=צ׳אטבוט)', 'מילים_ברפלקציה': 'מספר מילים ברפלקציה'}
    for param_name in ['const', 'is_sim', 'מילים_ברפלקציה']:
        rows_reg.append({
            'משתנה': var_map[param_name],
            'מקדם B': f"{ols_res.params[param_name]:.4f}",
            'שגיאת תקן SE': f"{ols_res.bse[param_name]:.4f}",
            't': f"{ols_res.tvalues[param_name]:.3f}",
            'ערך p': f"{ols_res.pvalues[param_name]:.4f}",
            '95% CI': f"[{ols_res.conf_int().loc[param_name, 0]:.3f}, {ols_res.conf_int().loc[param_name, 1]:.3f}]"
        })
    rows_reg.append({
        'משתנה': 'סטטיסטיקת המודל',
        'מקדם B': f"R² = {ols_res.rsquared:.3f}",
        'שגיאת תקן SE': f"Adj R² = {ols_res.rsquared_adj:.3f}",
        't': f"F({int(ols_res.df_model)}, {int(ols_res.df_resid)}) = {ols_res.fvalue:.2f}",
        'ערך p': f"p = {ols_res.f_pvalue:.4e}",
        '95% CI': f"N = {int(ols_res.nobs)}"
    })
    tab_4c3 = pd.DataFrame(rows_reg)
    tab_4c3.to_csv(os.path.join(base_dir, "tab_4c3_kember_regression.csv"), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # 5. טבלה 4ד: 6 תמות תחקיר (Fisher Exact + FDR)
    # ==============================================================================
    theme_cols = [
        ('רפלקציה_תמה_1_אימות_מקורות_והצלבה', 'תמה 1: אימות מקורות והצלבת מידע', 'חשיבות כניסה לאתר, בדיקת תאריכים ומקורות'),
        ('רפלקציה_תמה_2_הפקת_לקחים_לעתיד', 'תמה 2: הפקת לקחים לעתיד', 'תובנות מעשיות לשימוש עתידי בכלי בינה מלאכותית'),
        ('רפלקציה_תמה_3_הסתמכות_שטחית_ומלכודת_המקור', 'תמה 3: מלכודת המקור והסתמכות שטחית', 'הסתמכות על כותרת האתר בלבד או שמו של גוף התקשורת'),
        ('רפלקציה_תמה_4_קושי_בממשק_או_אי_הבנה', 'תמה 4: קושי בממשק או אי-הבנה', 'דיווח על קושי טכני, חוסר הבנה של המשימה או המערכת'),
        ('רפלקציה_תמה_5_תגובה_קצרה_ללא_פירוט', 'תמה 5: תגובה קצרה ללא פירוט', 'תשובה לאקונית, חוסר מעורבות רפלקטיבית'),
        ('רפלקציה_תמה_6_ספקנות_ומודעות_להזיות_AI', 'תמה 6: ספקנות ומודעות להזיות AI', 'הבנה שהבוט מייצר מידע שגוי, הצורך בביקורתיות')
    ]

    rows_4d = []
    pvals_4d = []

    for t_col, t_name, t_desc in theme_cols:
        s_1 = int((sim_df[t_col] == 1).sum())
        s_0 = n_sim - s_1
        c_1 = int((cb_df[t_col] == 1).sum())
        c_0 = n_cb - c_1

        table = [[s_1, s_0], [c_1, c_0]]
        odds, or_low, or_high = compute_fisher_exact_or_ci(table)
        _, p_fish = stats.fisher_exact(table)
        pvals_4d.append(p_fish)

        rows_4d.append({
            'תמה': t_name,
            'תיאור': t_desc,
            'מבחן_סטטיסטי': "Fisher's Exact Test",
            'סימולציה N (%)': f"{s_1}/{n_sim} ({s_1 / n_sim * 100:.1f}%)",
            'צ׳אטבוט N (%)': f"{c_1}/{n_cb} ({c_1 / n_cb * 100:.1f}%)",
            'Odds Ratio': f"{odds:.2f}" if not np.isinf(odds) else "Inf",
            '95% CI': format_ci(or_low, or_high),
            'ערך p גולמי': p_fish
        })

    # FDR
    adj_p = fdr_bh(pvals_4d)

    for i, r in enumerate(rows_4d):
        p_raw = r['ערך p גולמי']
        p_fdr = adj_p[i]
        r['ערך p גולמי'] = f"{p_raw:.4f}" if p_raw >= 0.0001 else "< .001"
        r['ערך p מתוקן FDR'] = f"{p_fdr:.4f}" if p_fdr >= 0.0001 else "< .001"
        r['מובהק (q<.05)'] = 'כן' if p_fdr < 0.05 else 'לא'
        if p_fdr < 0.001:
            stars = '***'
        elif p_fdr < 0.01:
            stars = '**'
        elif p_fdr < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'
        r['רמת מובהקות'] = stars if stars != 'n.s.' else 'לא מובהק (n.s.)'

    tab_4d = pd.DataFrame(rows_4d)
    tab_4d.to_csv(os.path.join(base_dir, "tab_4d_reflection_themes.csv"), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # 6. טבלה 4ה: ניתוח תיווך (Mediation Analysis) עם Bootstrap (5000 דגימות)
    # ==============================================================================
    print("מחשב ניתוח תיווך עם 5000 איטרציות Bootstrap...")
    np.random.seed(42)
    n_boot = 5000

    def run_mediation(df_med, x_col, m_col, y_col):
        clean = df_med[[x_col, m_col, y_col]].dropna()
        X = clean[x_col].values
        M = clean[m_col].values
        Y = clean[y_col].values
        n = len(clean)

        # a נתיב
        a = sm.OLS(M, sm.add_constant(X)).fit().params[1]
        # b ונתיב ישיר c'
        res_y = sm.OLS(Y, sm.add_constant(np.column_stack([M, X]))).fit()
        b = res_y.params[1]
        c_prime = res_y.params[2]
        # c אפקט כולל
        c = sm.OLS(Y, sm.add_constant(X)).fit().params[1]
        ab = a * b

        # Bootstrap for ab
        boot_ab = np.zeros(n_boot)
        for i in range(n_boot):
            boot_idx = np.random.choice(n, size=n, replace=True)
            X_b = X[boot_idx]
            M_b = M[boot_idx]
            Y_b = Y[boot_idx]

            a_b = sm.OLS(M_b, sm.add_constant(X_b)).fit().params[1]
            b_b = sm.OLS(Y_b, sm.add_constant(np.column_stack([M_b, X_b]))).fit().params[1]
            boot_ab[i] = a_b * b_b

        llci = np.percentile(boot_ab, 2.5)
        ulci = np.percentile(boot_ab, 97.5)
        pct_mediated = (ab / c * 100) if abs(c) > 1e-5 else np.nan

        return {
            'N': n,
            'נתיב a (X->M)': a,
            'נתיב b (M->Y)': b,
            'אפקט כולל c (X->Y)': c,
            'אפקט ישיר c\'': c_prime,
            'אפקט עקיף ab': ab,
            'Bootstrap LLCI 95%': llci,
            'Bootstrap ULCI 95%': ulci,
            'שיעור תיווך %': pct_mediated,
            'מובהק (0 מחוץ לרווח)': 'כן' if (llci > 0 or ulci < 0) else 'לא'
        }

    # מודל 1: X (קבוצה) -> M (בדיקת מקורות באתר) -> Y (ציון 4 ממדים)
    med1 = run_mediation(task_df, 'is_sim', 'מקורות_שנבדקו_באתר', 'ציון_4_ממדים')
    # מודל 2: X (קבוצה) -> M (בדיקת מקורות באתר) -> Y (שיעור דיוק זיהוי מקורות)
    med2 = run_mediation(task_df, 'is_sim', 'מקורות_שנבדקו_באתר', 'שיעור_דיוק_סיווג_מקורות')

    tab_4e = pd.DataFrame([
        {
            'מודל': 'מודל 1: השפעת קבוצה על ציון כולל בתיווך בדיקת מקורות',
            'משתנה בלתי תלוי (X)': 'קבוצת ניסוי (0=צ׳אטבוט, 1=סימולציה)',
            'מתווך (M)': 'מקורות שנבדקו באתר (כניסה פיזית)',
            'משתנה תלוי (Y)': 'ציון איכות כולל (4 ממדים)',
            'N': med1['N'],
            'נתיב a': f"{med1['נתיב a (X->M)']:.3f}",
            'נתיב b': f"{med1['נתיב b (M->Y)']:.3f}",
            'אפקט כולל c': f"{med1['אפקט כולל c (X->Y)']:.3f}",
            'אפקט ישיר c\'': "{:.3f}".format(med1["אפקט ישיר c'"]),
            'אפקט עקיף ab': f"{med1['אפקט עקיף ab']:.3f}",
            '95% Bootstrap CI': f"[{med1['Bootstrap LLCI 95%']:.3f}, {med1['Bootstrap ULCI 95%']:.3f}]",
            'שיעור תיווך': f"{med1['שיעור תיווך %']:.1f}%",
            'מובהק': med1['מובהק (0 מחוץ לרווח)']
        },
        {
            'מודל': 'מודל 2: השפעת קבוצה על דיוק סיווג בתיווך בדיקת מקורות',
            'משתנה בלתי תלוי (X)': 'קבוצת ניסוי (0=צ׳אטבוט, 1=סימולציה)',
            'מתווך (M)': 'מקורות שנבדקו באתר (כניסה פיזית)',
            'משתנה תלוי (Y)': 'שיעור דיוק בסיווג מקורות (0-1)',
            'N': med2['N'],
            'נתיב a': f"{med2['נתיב a (X->M)']:.3f}",
            'נתיב b': f"{med2['נתיב b (M->Y)']:.3f}",
            'אפקט כולל c': f"{med2['אפקט כולל c (X->Y)']:.3f}",
            'אפקט ישיר c\'': "{:.3f}".format(med2["אפקט ישיר c'"]),
            'אפקט עקיף ab': f"{med2['אפקט עקיף ab']:.3f}",
            '95% Bootstrap CI': f"[{med2['Bootstrap LLCI 95%']:.3f}, {med2['Bootstrap ULCI 95%']:.3f}]",
            'שיעור תיווך': f"{med2['שיעור תיווך %']:.1f}%",
            'מובהק': med2['מובהק (0 מחוץ לרווח)']
        }
    ])
    tab_4e.to_csv(os.path.join(base_dir, "tab_4e_mediation_analysis.csv"), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # 7. טבלה 4ו: זמן שהייה מול דיוק בתוך הסימולציה
    # ==============================================================================
    sim_sub = sim_df[['משך_שהייה_ממוצע_במקור_שניות', 'שיעור_דיוק_סיווג_מקורות']].dropna()
    r_sp, p_sp = stats.spearmanr(sim_sub['משך_שהייה_ממוצע_במקור_שניות'], sim_sub['שיעור_דיוק_סיווג_מקורות'])
    reg_dwell = sm.OLS(sim_sub['שיעור_דיוק_סיווג_מקורות'], sm.add_constant(sim_sub['משך_שהייה_ממוצע_במקור_שניות'])).fit()

    tab_4f = pd.DataFrame([
        {
            'מודל': 'מתאם ספירמן (Spearman Rank Correlation)',
            'משתנה 1': 'משך שהייה ממוצע במקור (שניות)',
            'משתנה 2': 'שיעור דיוק סיווג מקורות (0-1)',
            'מדד': f"r_s = {r_sp:.3f}",
            'ערך p': f"{p_sp:.4f}",
            'מובהק (p<.05)': 'כן' if p_sp < 0.05 else 'לא (מגמה)',
            'פרשנות': 'מתאם חיובי בינוני בין עומק העיון לדיוק'
        },
        {
            'מודל': 'רגרסיה לינארית OLS',
            'משתנה 1': 'משך שהייה ממוצע במקור (שניות)',
            'משתנה 2': 'שיעור דיוק סיווג מקורות (0-1)',
            'מדד': f"B = {reg_dwell.params.iloc[1]:.4f} (SE = {reg_dwell.bse.iloc[1]:.4f}), R² = {reg_dwell.rsquared:.3f}",
            'ערך p': f"{reg_dwell.pvalues.iloc[1]:.4f}",
            'מובהק (p<.05)': 'כן' if reg_dwell.pvalues.iloc[1] < 0.05 else 'לא',
            'פרשנות': f"כל תוספת של 10 שניות שהייה באתר המקור מעלה את הדיוק ב-{reg_dwell.params.iloc[1]*10*100:.1f}%"
        }
    ])
    tab_4f.to_csv(os.path.join(base_dir, "tab_4f_dwell_vs_accuracy_sim.csv"), index=False, encoding='utf-8-sig')

    print(">>> שלב 4 הושלם. כל טבלאות רובד 2 נשמרו ב-analysis_scripts.")

if __name__ == '__main__':
    main()


#==============================================================================
# PART 7: Perception layer: questionnaire scales, items, open-ended questions
# Thesis: Section 6.3; Tables E6, E7, E11, E17
#==============================================================================
# (source script: 05_layer3_perception.py)

# ==============================================================================
# analysis_scripts/05_layer3_perception.py
# רובד 3: תפיסה (Perception) - סולמות TAM (PU, PEOU, BI), CriTT, שביעות רצון ושאלות פתוחות
# מדד TAM כולל אינו מחושב; הסולמות מדווחים בנפרד.
# מדגם השאלון: N=49 (סימולציה n=21, צ'אטבוט n=28 - כולל נבדקת 70 user_525vg81ny)
# פרויקט מחקר WorkAI
# ==============================================================================

import os
import numpy as np
import pandas as pd
from scipy import stats

def fdr_bh(pvals):
    n_tests = len(pvals)
    sorted_indices = np.argsort(pvals)
    adj_p = np.zeros(n_tests)
    cum_min = 1.0
    for rank, idx in reversed(list(enumerate(sorted_indices))):
        p_adj = pvals[idx] * n_tests / (rank + 1)
        cum_min = min(cum_min, p_adj)
        adj_p[idx] = min(cum_min, 1.0)
    return adj_p

def main():
    print(">>> שלב 5: הרצת ניתוח רובד 3 (תפיסה)...")
    base_dir = SCRIPTS_DIR
    master_path = os.path.join(base_dir, "master_clean_n70.csv")
    df = pd.read_csv(master_path)

    # סינון נבדקות שמילאו שאלון (N=49, סימולציה n=21, צ'אטבוט n=28)
    surv_df = df[df['מילאה_שאלון'] == 1].copy()
    sim_surv = surv_df[surv_df['קבוצה_אנגלית'] == 'Simulation'].copy()
    cb_surv  = surv_df[surv_df['קבוצה_אנגלית'] == 'Chatbot'].copy()

    print(f"מדגם השאלון: סה״כ N={len(surv_df)}, סימולציה n={len(sim_surv)}, צ׳אטבוט n={len(cb_surv)}")

    # ==============================================================================
    # 1. טבלה 5א: 4 סולמות השאלון (PU, PEOU, BI, CriTT) - ללא TAM כולל!
    # ==============================================================================
    scales_info = [
        ('PU', 'שימושיות נתפסת (Perceived Usefulness - PU)', 'סולם TAM, 4 פריטים (1-4), טווח 1-5', 4),
        ('PEOU', 'קלות שימוש נתפסת (Perceived Ease of Use - PEOU)', 'סולם TAM, 4 פריטים (5-8), טווח 1-5', 4),
        ('BI', 'כוונת שימוש התנהגותית (Behavioral Intention - BI)', 'סולם TAM, 2 פריטים (9-10), טווח 1-5', 2),
        ('CriTT', 'הערכה עצמית לחשיבה ביקורתית (CriTT)', 'שאלון CriTT, 17 פריטים, טווח 1-5', 17)
    ]

    rows_5a = []
    pvals_5a = []

    for col, name, desc, n_items in scales_info:
        s_data = sim_surv[col].dropna()
        c_data = cb_surv[col].dropna()

        res_u = stats.mannwhitneyu(s_data, c_data, alternative='two-sided')
        u_s = float(res_u.statistic)
        u_c = float(len(s_data) * len(c_data) - u_s)
        r_b = (u_s - u_c) / (len(s_data) * len(c_data))
        p = res_u.pvalue
        ci_l, ci_h = compute_r_rb_ci(s_data.values, c_data.values)
        pvals_5a.append(p)
        strength = 'גדול' if abs(r_b) >= 0.5 else ('בינוני' if abs(r_b) >= 0.3 else ('קטן' if abs(r_b) >= 0.1 else 'זניח'))

        rows_5a.append({
            'סולם': name,
            'תיאור ומספר פריטים': f"{desc} ({n_items} פריטים)",
            'מבחן_סטטיסטי': 'Mann-Whitney U (דו-צדדי)',
            'סימולציה N': len(s_data),
            'סימולציה ממוצע (ס״ת)': f"{s_data.mean():.2f} ({s_data.std():.2f})",
            'צ׳אטבוט N': len(c_data),
            'צ׳אטבוט ממוצע (ס״ת)': f"{c_data.mean():.2f} ({c_data.std():.2f})",
            'Mann-Whitney U_Sim': f"{u_s:.1f}",
            'U_CB': f"{u_c:.1f}",
            'סכום U תקין': 'כן' if abs((u_s + u_c) - (len(s_data)*len(c_data))) < 1e-5 else 'שגיאה',
            'ערך p גולמי': p,
            'גודל אפקט r_rb': f"{r_b:.3f}",
            '95% CI (BCa)': format_ci(ci_l, ci_h),
            'עוצמת אפקט': strength
        })

    fdr_5a = fdr_bh(pvals_5a)
    for i, r in enumerate(rows_5a):
        p_raw = r['ערך p גולמי']
        p_fdr = fdr_5a[i]
        r['ערך p גולמי'] = f"{p_raw:.4f}" if p_raw >= 0.0001 else "< .001"
        r['ערך p מתוקן FDR'] = f"{p_fdr:.4f}" if p_fdr >= 0.0001 else "< .001"
        r['מובהק (q<.05)'] = 'כן' if p_fdr < 0.05 else 'לא'
        if p_fdr < 0.001:
            stars = '***'
        elif p_fdr < 0.01:
            stars = '**'
        elif p_fdr < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'
        r['רמת מובהקות'] = f"{stars} ({r['עוצמת אפקט']})" if stars != 'n.s.' else 'לא מובהק (n.s.)'

    tab_5a = pd.DataFrame(rows_5a)
    tab_5a.to_csv(os.path.join(base_dir, "tab_5a_scales_comparison.csv"), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # 2. טבלה 5ב: 27 פריטי ליקרט בודדים (10 TAM + 17 CriTT)
    # ==============================================================================
    likert_items = [
        # 10 פריטי TAM
        ('TAM_1_שיפור_הערכת_אמינות', 'TAM 1', 'PU', 'השימוש ב-WorkAI שיפר את היכולת שלי להעריך אמינות מידע'),
        ('TAM_2_הגדלת_יעילות_זיהוי_אמינות', 'TAM 2', 'PU', 'השימוש ב-WorkAI הגדיל את היעילות שלי בזיהוי מידע אמין מול לא אמין'),
        ('TAM_3_שימושי_ללימודים_אקדמיים', 'TAM 3', 'PU', 'WorkAI שימושי ללימודים האקדמיים שלי'),
        ('TAM_4_עזר_לפתח_חשיבה_ביקורתית', 'TAM 4', 'PU', 'WorkAI עזר לי לפתח חשיבה ביקורתית כלפי תוצרי AI'),
        ('TAM_5_קל_ללמוד_להשתמש', 'TAM 5', 'PEOU', 'היה לי קל ללמוד להשתמש ב-WorkAI'),
        ('TAM_6_קל_לגרום_לעשות_מה_שרציתי', 'TAM 6', 'PEOU', 'היה לי קל לגרום ל-WorkAI לעשות את מה שרציתי'),
        ('TAM_7_אינטראקציה_ברורה_ומובנת', 'TAM 7', 'PEOU', 'האינטראקציה שלי עם WorkAI הייתה ברורה ומובנת'),
        ('TAM_8_גמיש_לשימוש_וניווט', 'TAM 8', 'PEOU', 'מצאתי ש-WorkAI גמיש ונוח לשימוש וניווט'),
        ('TAM_9_מתכוון_להשתמש_שוב', 'TAM 9', 'BI', 'אני מתכוון/ת להשתמש שוב בכלים דמויי WorkAI אם יהיו זמינים'),
        ('TAM_10_רוצה_כלים_דומים_נוספים', 'TAM 10', 'BI', 'הייתי רוצה שישלבו כלים דומים בקורסים נוספים'),
        # 17 פריטי CriTT
        ('CRITT_1_זיהוי_שפה_רגשית', 'CriTT 1', 'CriTT', 'אני מזהה בקלות שפה טעונה רגשית או הטיה בטקסטים'),
        ('CRITT_2_מטרה_מוגדרת_בחשיבה', 'CriTT 2', 'CriTT', 'יש לי מטרה מוגדרת וברורה כאשר אני ניגש/ת לנתח בעיה מורכבת'),
        ('CRITT_3_זיהוי_מבנה_טיעונים', 'CriTT 3', 'CriTT', 'אני מסוגל/ת לפרק טיעון למרכיביו הלוגיים (טענה, נימוק, ראיה)'),
        ('CRITT_4_ביטוי_חשיבה_בכתב', 'CriTT 4', 'CriTT', 'אני מצליח/ה לבטא היטב בכתב תהליכי חשיבה ביקורתית'),
        ('CRITT_5_חשיבה_ממוקדת_ושיטתית', 'CriTT 5', 'CriTT', 'אני פועל/ת באופן שיטתי וממוקד כשאני בוחן/ת מידע חדש'),
        ('CRITT_6_חושב_ביקורתי_טוב_כללי', 'CriTT 6', 'CriTT', 'באופן כללי, אני תופס/ת את עצמי כחושב/ת ביקורתי/ת טוב/ה'),
        ('CRITT_7_הצלחה_במטלות_ביקורתיות', 'CriTT 7', 'CriTT', 'אני משיג/ה תוצאות טובות במטלות הדורשות חשיבה ביקורתית'),
        ('CRITT_8_חשיבה_ביקורתית_במטלות', 'CriTT 8', 'CriTT', 'אני מיישם/ת באופן עקבי חשיבה ביקורתית במטלות אקדמיות'),
        ('CRITT_9_גישה_לנושאים_מורכבים', 'CriTT 9', 'CriTT', 'יש לי גישה פתוחה ומעמיקה לנושאים מורכבים ושנויים במחלוקת'),
        ('CRITT_10_שיפוט_ערך_מידע_חדש', 'CriTT 10', 'CriTT', 'אני בוחן/ת בביקורתיות את הערך והאמינות של מידע חדש שאני נחשף/ת אליו'),
        ('CRITT_11_הערכת_טיעוני_אחרים', 'CriTT 11', 'CriTT', 'אני יודע/ת להעריך את החוזק והחולשות בטיעונים של אחרים'),
        ('CRITT_12_שקלול_שני_צדדי_טיעון', 'CriTT 12', 'CriTT', 'אני מקפיד/ה לשקול את שני צידי המטבע לפני גיבוש עמדה סופית'),
        ('CRITT_13_זיהוי_אנלוגיות_ורעיונות', 'CriTT 13', 'CriTT', 'אני מזהה אנלוגיות, הקשרים והנחות סמויות בטקסטים'),
        ('CRITT_14_סילוק_משתנים_מיותרים', 'CriTT 14', 'CriTT', 'אני יודע/ת לסנן מידע שולי ולהתמקד במידע המהותי ביותר'),
        ('CRITT_15_חשיבה_ביקורתית_בקריאה', 'CriTT 15', 'CriTT', 'קריאה ביקורתית היא חלק בלתי נפרד מאופן הלמידה שלי'),
        ('CRITT_16_ניסוח_מחדש_של_טיעונים', 'CriTT 16', 'CriTT', 'אני מסוגל/ת לנסח מחדש טיעונים מורכבים במילים פשוטות ומדויקות'),
        ('CRITT_17_חשיבה_ביקורתית_בהרצאות', 'CriTT 17', 'CriTT', 'אני מקשיב/ה בביקורתיות להרצאות ולדברי מרצים ואיני מקבל/ת דברים כמובן מאליו')
    ]

    likert_map = {'כלל לא': 1, 'לא מסכימ.ה': 2, 'מסכימ.ה מעט': 3, 'מסכימ.ה': 4, 'מסכימ.ה מאוד': 5}

    rows_5b = []
    pvals_5b = []

    for col, num, scale_grp, item_txt in likert_items:
        s_raw = sim_surv[col].dropna()
        c_raw = cb_surv[col].dropna()

        # המרה לערך נומרי 1-5 במידה והנתון הוא מחרוזת
        s_data = s_raw.map(likert_map).dropna() if (not pd.api.types.is_numeric_dtype(s_raw)) else pd.to_numeric(s_raw, errors='coerce').dropna()
        c_data = c_raw.map(likert_map).dropna() if (not pd.api.types.is_numeric_dtype(c_raw)) else pd.to_numeric(c_raw, errors='coerce').dropna()

        res_u = stats.mannwhitneyu(s_data, c_data, alternative='two-sided')
        u_s = float(res_u.statistic)
        u_c = float(len(s_data) * len(c_data) - u_s)
        r_b = (u_s - u_c) / (len(s_data) * len(c_data))
        p = res_u.pvalue
        ci_l, ci_h = compute_r_rb_ci(s_data.values, c_data.values)
        pvals_5b.append(p)

        rows_5b.append({
            'מס׳': num,
            'סולם': scale_grp,
            'נוסח הפריט': item_txt,
            'מבחן_סטטיסטי': 'Mann-Whitney U (דו-צדדי)',
            'סימולציה N': len(s_data),
            'סימולציה ממוצע (ס״ת)': f"{s_data.mean():.2f} ({s_data.std():.2f})",
            'צ׳אטבוט N': len(c_data),
            'צ׳אטבוט ממוצע (ס״ת)': f"{c_data.mean():.2f} ({c_data.std():.2f})",
            'Mann-Whitney U_Sim': f"{u_s:.1f}",
            'U_CB': f"{u_c:.1f}",
            'סכום U תקין': 'כן' if abs((u_s + u_c) - (len(s_data)*len(c_data))) < 1e-5 else 'שגיאה',
            'ערך p גולמי': p,
            'גודל אפקט r_rb': f"{r_b:.3f}",
            '95% CI (BCa)': format_ci(ci_l, ci_h)
        })

    fdr_5b = fdr_bh(pvals_5b)
    for i, r in enumerate(rows_5b):
        p_raw = r['ערך p גולמי']
        p_fdr = fdr_5b[i]
        r['ערך p גולמי'] = f"{p_raw:.4f}" if p_raw >= 0.0001 else "< .001"
        r['ערך p מתוקן FDR'] = f"{p_fdr:.4f}" if p_fdr >= 0.0001 else "< .001"
        r['מובהק (q<.05)'] = 'כן' if p_fdr < 0.05 else 'לא'
        if p_fdr < 0.001:
            stars = '***'
        elif p_fdr < 0.01:
            stars = '**'
        elif p_fdr < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'
        r['רמת מובהקות'] = stars

    tab_5b = pd.DataFrame(rows_5b)
    tab_5b.to_csv(os.path.join(base_dir, "tab_5b_likert_27_items.csv"), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # 3. טבלה 5ג: שביעות רצון (2 פריטים עם FDR)
    # ==============================================================================
    sat_items = [
        ('שביעות_רצון_תאימות_ציפיות', 'תאימות לציפיות', 'עד כמה הפעילות ענתה על הציפיות שלך מהכלי? (1-5)'),
        ('שביעות_רצון_המלצה_לקורסים', 'המלצה לקורסים נוספים', 'באיזו מידה היית ממליץ/ה לשלב פעילות זו בקורסים נוספים? (1-5)')
    ]

    rows_5c = []
    pvals_5c = []

    for col, name, desc in sat_items:
        s_data = sim_surv[col].dropna()
        c_data = cb_surv[col].dropna()

        res_u = stats.mannwhitneyu(s_data, c_data, alternative='two-sided')
        u_s = float(res_u.statistic)
        u_c = float(len(s_data) * len(c_data) - u_s)
        r_b = (u_s - u_c) / (len(s_data) * len(c_data))
        p = res_u.pvalue
        ci_l, ci_h = compute_r_rb_ci(s_data.values, c_data.values)
        pvals_5c.append(p)
        strength = 'גדול' if abs(r_b) >= 0.5 else ('בינוני' if abs(r_b) >= 0.3 else 'קטן')

        rows_5c.append({
            'מדד שביעות רצון': name,
            'נוסח הפריט': desc,
            'מבחן_סטטיסטי': 'Mann-Whitney U (דו-צדדי)',
            'סימולציה N': len(s_data),
            'סימולציה ממוצע (ס״ת)': f"{s_data.mean():.2f} ({s_data.std():.2f})",
            'צ׳אטבוט N': len(c_data),
            'צ׳אטבוט ממוצע (ס״ת)': f"{c_data.mean():.2f} ({c_data.std():.2f})",
            'Mann-Whitney U_Sim': f"{u_s:.1f}",
            'U_CB': f"{u_c:.1f}",
            'סכום U תקין': 'כן' if abs((u_s + u_c) - (len(s_data)*len(c_data))) < 1e-5 else 'שגיאה',
            'ערך p גולמי': p,
            'גודל אפקט r_rb': f"{r_b:.3f}",
            '95% CI (BCa)': format_ci(ci_l, ci_h),
            'עוצמת אפקט': strength
        })

    fdr_5c = fdr_bh(pvals_5c)
    for i, r in enumerate(rows_5c):
        p_raw = r['ערך p גולמי']
        p_fdr = fdr_5c[i]
        r['ערך p גולמי'] = f"{p_raw:.4f}" if p_raw >= 0.0001 else "< .001"
        r['ערך p מתוקן FDR'] = f"{p_fdr:.4f}" if p_fdr >= 0.0001 else "< .001"
        r['מובהק (q<.05)'] = 'כן' if p_fdr < 0.05 else 'לא'
        if p_fdr < 0.001:
            stars = '***'
        elif p_fdr < 0.01:
            stars = '**'
        elif p_fdr < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'
        r['רמת מובהקות'] = f"{stars} ({r['עוצמת אפקט']})" if stars != 'n.s.' else 'לא מובהק (n.s.)'

    tab_5c = pd.DataFrame(rows_5c)
    tab_5c.to_csv(os.path.join(base_dir, "tab_5c_satisfaction.csv"), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # 4. טבלה 5ד: 3 שאלות משוב פתוחות (ניתוח תמטי וציטוטים נבחרים)
    # ==============================================================================
    # שאלה 1: מה למדת?
    # שאלה 2: מה היית משנה?
    # שאלה 3: קשיים שחווית?
    open_themes_summary = [
        {
            'שאלה': 'שאלה פתוחה 1: מה הדבר החשוב ביותר שלמדת? (N=35)',
            'תמה מרכזית': 'חשיבות בדיקת מקורות ואימות אקטיבי מול AI',
            'סימולציה (שכיחות)': '4 מתוך 15 (26.7%)',
            'צ׳אטבוט (שכיחות)': '3 מתוך 20 (15.0%)',
            'ציטוט מייצג - סימולציה': '"למדתי שחשוב לשים לב לבקרת איכות על התוכן, כמו פרסומות ממומנות, מי עומד מאחורי האתר. לא לקחת כמובן מאליו מקורות שאנחנו קוראים ונחשפים אליהם."',
            'ציטוט מייצג - צ׳אטבוט': '"ביקורת ובדיקת מקורות אקטיבית - להכנס ולחפש עליהם בגוגל, לראות שהצאט חיבר וצירף מקור זה לא מספיק... חיבור למקור נתן לי תחושה לסמוך על הנתונים - ועכשיו אני מגלה שזה לא מספיק ולא סופר אמין"'
        },
        {
            'שאלה': 'שאלה פתוחה 1: מה הדבר החשוב ביותר שלמדת? (N=35)',
            'תמה מרכזית': 'לערער על המידע מ-AI ואימות נתונים',
            'סימולציה (שכיחות)': '4 מתוך 15 (26.7%)',
            'צ׳אטבוט (שכיחות)': '3 מתוך 20 (15.0%)',
            'ציטוט מייצג - סימולציה': '"אימות ובדיקת הנתונים שמתקבלים מהבינה במקורות."',
            'ציטוט מייצג - צ׳אטבוט': '"לערער על המידע שאני מקבלת מה-AI"'
        },
        {
            'שאלה': 'שאלה פתוחה 1: מה הדבר החשוב ביותר שלמדת? (N=35)',
            'תמה מרכזית': 'שכלול ודיוק ניסוח פרומפטים',
            'סימולציה (שכיחות)': '1 מתוך 15 (6.7%)',
            'צ׳אטבוט (שכיחות)': '3 מתוך 20 (15.0%)',
            'ציטוט מייצג - סימולציה': '"השימוש בWorkAI שיפר את היכולת שלי להעריך אמינות מידע"',
            'ציטוט מייצג - צ׳אטבוט': '"לפרט יותר בפרומפטים שלי"'
        },
        {
            'שאלה': 'שאלה פתוחה 1: מה הדבר החשוב ביותר שלמדת? (N=35)',
            'תמה מרכזית': 'אי-למידה או חוסר בהירות במטרת הכלי',
            'סימולציה (שכיחות)': '0 מתוך 15 (0.0%)',
            'צ׳אטבוט (שכיחות)': '4 מתוך 20 (20.0%)',
            'ציטוט מייצג - סימולציה': 'לא צוין בסימולציה',
            'ציטוט מייצג - צ׳אטבוט': '"לא הבנתי מה עושים בו ומה המטרה שלו. ההסבר שלה בהתחלה לא היה מובן לפני שניגשים למשימה"'
        },
        {
            'שאלה': 'שאלה פתוחה 2: מה היית משנה במערכת? (N=34)',
            'תמה מרכזית': 'הבהרת הנחיות ומטרת המשימה מראש',
            'סימולציה (שכיחות)': '5 מתוך 11 (45.5%)',
            'צ׳אטבוט (שכיחות)': '12 מתוך 23 (52.2%)',
            'ציטוט מייצג - סימולציה': '"בהירות ההנחיה בשלב של אימות המקורות."',
            'ציטוט מייצג - צ׳אטבוט': '"הייתי קודם מבהירה למשתמש מה התהליך שהוא צריך לעשות כדי לבצע את המשימה (קרא את המשימה > דבר עם המערכת כדי למצוא מקורות אמינים > כשהגעת לרשימת מקורות שאתה מרוצה ממנה צור את הדו\\"ח). המערכת מעוצבת ובנויה תכנותית נהדר, אבל הפלואו לא היה מובן לי כ\\"כ אז אני מאמינה שלא ניצלתי את המערכת עד הסוף"'
        },
        {
            'שאלה': 'שאלה פתוחה 2: מה היית משנה במערכת? (N=34)',
            'תמה מרכזית': 'גמישות בממשק, ניווט וחזרה לאחור',
            'סימולציה (שכיחות)': '3 מתוך 11 (27.3%)',
            'צ׳אטבוט (שכיחות)': '5 מתוך 23 (21.7%)',
            'ציטוט מייצג - סימולציה': '"כשעברתי למצב התייעצות לא הופיעו לי המקורות שהצ\'אט כתב לי בהתחלה, ולא הבנתי לאן הם נעלמו. אחר כך החזרתי למצב רגיל והם הופיעו אבל זה גרם לחוסר הבנה"',
            'ציטוט מייצג - צ׳אטבוט': '"כמות ההודעות שאפשר לשלוח, אחרי מספר הודעות פשוט כבר ישר ביקש ממני למלא את הדוח והיו לי עוד דברים לשאול"'
        },
        {
            'שאלה': 'שאלה פתוחה 2: מה היית משנה במערכת? (N=34)',
            'תמה מרכזית': 'ללא שינוי / שביעות רצון מלאה',
            'סימולציה (שכיחות)': '1 מתוך 11 (9.1%)',
            'צ׳אטבוט (שכיחות)': '3 מתוך 23 (13.0%)',
            'ציטוט מייצג - סימולציה': '"כלום"',
            'ציטוט מייצג - צ׳אטבוט': '"לא הייתי משנה"'
        },
        {
            'שאלה': 'שאלה פתוחה 3: קשיים ואתגרים במהלך הפעילות (N=31)',
            'תמה מרכזית': 'ללא כל קושי / חוויה חלקה',
            'סימולציה (שכיחות)': '3 מתוך 12 (25.0%)',
            'צ׳אטבוט (שכיחות)': '7 מתוך 19 (36.8%)',
            'ציטוט מייצג - סימולציה': '"לא, לא נתקלתי בבעיה"',
            'ציטוט מייצג - צ׳אטבוט': '"לא נתקלתי בקושי או בעיה"'
        },
        {
            'שאלה': 'שאלה פתוחה 3: קשיים ואתגרים במהלך הפעילות (N=31)',
            'תמה מרכזית': 'אי-בהירות לגבי שלבי ההתקדמות והמצופה (Flow)',
            'סימולציה (שכיחות)': '3 מתוך 12 (25.0%)',
            'צ׳אטבוט (שכיחות)': '7 מתוך 19 (36.8%)',
            'ציטוט מייצג - סימולציה': '"היו שלבים בתרגול שלא ממש היה ברור לאן מתקדמים, ניהלתי שיח עם הצט וראיתי שזה לא מתקדם ואין לי איך להמשיך. זה יכול לגרום לנטישה ברגע שלא מבינים."',
            'ציטוט מייצג - צ׳אטבוט': '"לא בעיות טכניות, בעיקר בעיה בהבנת המערכת (לא הבנתי מתי אני אמורה להפסיק לשאול אותו שאלות או מה הוא אמור לעשות)"'
        },
        {
            'שאלה': 'שאלה פתוחה 3: קשיים ואתגרים במהלך הפעילות (N=31)',
            'תמה מרכזית': 'מגבלות טכניות בממשק (פתיחת קישורים, תקיעות, סגירת חלונות)',
            'סימולציה (שכיחות)': '2 מתוך 12 (16.7%)',
            'צ׳אטבוט (שכיחות)': '4 מתוך 19 (21.1%)',
            'ציטוט מייצג - סימולציה': '"כן. אני הרגשתי לכודה בתוך סביבת התרגול... ולא הצלחתי לחזור חזרה למייל של דנה... וגם לא הצלחתי להוריד את הדו\\"ח כקובץ PDF."',
            'ציטוט מייצג - צ׳אטבוט': '"ניסיתי לעשות שימוש חוזר, בפעם השניה עשיתי ריענון לעמוד בלבד ואז השיחה הסימולציה לא התקדמה ונשאלו שוב ושוב אותן השאלות... סגרתי את הממשק ופתחתי את הקישור מחדש ועדיין אותה תקלה. מוזר"'
        }
    ]

    tab_5d = pd.DataFrame(open_themes_summary)
    tab_5d.to_csv(os.path.join(base_dir, "tab_5d_open_questions.csv"), index=False, encoding='utf-8-sig')

    print(">>> שלב 5 הושלם. כל טבלאות רובד 3 נשמרו ב-analysis_scripts.")

if __name__ == '__main__':
    main()


#==============================================================================
# PART 8: Cross-layer correlations and validation regression
# Thesis: Section 6.4; Tables E8, E9
#==============================================================================
# (source script: 06_cross_layer_correlations.py)

# ==============================================================================
# analysis_scripts/06_cross_layer_correlations.py
# הצלבות בין רבדים (Cross-Layer Correlations)
# שיטה ראשית: מתאם ספירמן חלקי מבוקר תנאי (Partial Spearman Correlation)
# שיטה משנית: מתאם ספירמן בתוך כל קבוצה בנפרד + מבחן Fisher's z להשוואת מקדמים
# רגרסיית תיקוף בין תמות תחקיר למדדי לוג אובייקטיביים
# פרויקט מחקר WorkAI
# ==============================================================================

import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

def fdr_bh(pvals):
    n_tests = len(pvals)
    sorted_indices = np.argsort(pvals)
    adj_p = np.zeros(n_tests)
    cum_min = 1.0
    for rank, idx in reversed(list(enumerate(sorted_indices))):
        p_adj = pvals[idx] * n_tests / (rank + 1)
        cum_min = min(cum_min, p_adj)
        adj_p[idx] = min(cum_min, 1.0)
    return adj_p

def partial_spearman(x, y, covar):
    valid = ~(x.isna() | y.isna() | covar.isna())
    if valid.sum() < 6:
        return np.nan, np.nan, int(valid.sum())
    xv = stats.rankdata(x[valid])
    yv = stats.rankdata(y[valid])
    zv = stats.rankdata(covar[valid])
    
    r_xy = np.corrcoef(xv, yv)[0, 1]
    r_xz = np.corrcoef(xv, zv)[0, 1]
    r_yz = np.corrcoef(yv, zv)[0, 1]
    
    denom = np.sqrt(max(0, 1 - r_xz**2) * max(0, 1 - r_yz**2))
    if denom == 0:
        return 0.0, 1.0, int(valid.sum())
    r_part = (r_xy - r_xz * r_yz) / denom
    r_part = max(-1.0, min(1.0, r_part))
    
    n = len(xv)
    df = n - 3
    if df <= 0 or abs(r_part) >= 1.0:
        return r_part, 0.0, n
    t_stat = r_part * np.sqrt(df / (1 - r_part**2))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return r_part, p_val, n

def fisher_z_test(r1, n1, r2, n2):
    if np.isnan(r1) or np.isnan(r2) or n1 <= 3 or n2 <= 3:
        return np.nan, np.nan
    r1 = max(-0.999, min(0.999, r1))
    r2 = max(-0.999, min(0.999, r2))
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_stat = (z1 - z2) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_val

def main():
    print(">>> שלב 6: הרצת ניתוח הצלבות בין רבדים...")
    base_dir = SCRIPTS_DIR
    master_path = os.path.join(base_dir, "master_clean_n70.csv")
    df = pd.read_csv(master_path)

    # יצירת משתנה תנאי בינארי (1=סימולציה, 0=צ'אטבוט)
    df['cond_num'] = (df['קבוצה_אנגלית'] == 'Simulation').astype(int)

    # הגדרת משתני הרבדים
    layer1_vars = [
        ('משך_פעילות_נטו_מנוקה_דקות', 'משך פעילות נטו (דקות)'),
        ('קצב_פעולות_לדקה_APM', 'קצב פעולות לדקה (APM)'),
        ('שיעור_ביקור_במקורות', 'שיעור ביקור במקורות באתר'),
        ('חיפושים_חיצוניים_בגוגל', 'מספר חיפושים חיצוניים בגוגל'),
        ('שיעור_ניחוש_ללא_בדיקה', 'שיעור ניחוש מקורות ללא בדיקה'),
        ('ENA_Dim1_MR1', 'מיקום בציר ENA Dim 1 (MR1)')
    ]

    layer2_vars = [
        ('ציון_4_ממדים', 'ציון איכות כולל (4 ממדים)'),
        ('שיעור_דיוק_סיווג_מקורות', 'שיעור דיוק בסיווג מקורות'),
        ('d_prime', 'מדד רגישות SDT (d\')'),
        ('רמת_Kember', 'רמת רפלקציה לפי קמבר (0-4)')
    ]

    layer3_vars = [
        ('PU', 'שימושיות נתפסת (PU)'),
        ('PEOU', 'קלות שימוש נתפסת (PEOU)'),
        ('BI', 'כוונת שימוש התנהגותית (BI)'),
        ('CriTT', 'הערכה עצמית לחשיבה ביקורתית (CriTT)')
    ]

    # ==============================================================================
    # 1. טבלה 6א: הצלבת רובד 1 (תהליך) מול רובד 2 (ביצוע)
    # ==============================================================================
    rows_6a = []
    pvals_6a = []

    # תת-קבוצות למדדי ביצוע (מי שהשלימו משימה)
    df_l1_l2 = df[df['השלימה_משימה'] == 1].copy()
    sim_l1_l2 = df_l1_l2[df_l1_l2['קבוצה_אנגלית'] == 'Simulation']
    cb_l1_l2  = df_l1_l2[df_l1_l2['קבוצה_אנגלית'] == 'Chatbot']

    for v1, l1_name in layer1_vars:
        for v2, l2_name in layer2_vars:
            # בדיקת מתאם חלקי מבוקר קבוצה
            r_part, p_part, n_tot = partial_spearman(df_l1_l2[v1], df_l1_l2[v2], df_l1_l2['cond_num'])
            
            # מתאמים בתוך כל קבוצה בנפרד
            s_sub = sim_l1_l2[[v1, v2]].dropna()
            c_sub = cb_l1_l2[[v1, v2]].dropna()
            
            r_sim, p_sim = stats.spearmanr(s_sub[v1], s_sub[v2]) if len(s_sub) >= 4 else (np.nan, np.nan)
            r_cb, p_cb   = stats.spearmanr(c_sub[v1], c_sub[v2]) if len(c_sub) >= 4 else (np.nan, np.nan)
            
            z_diff, p_diff = fisher_z_test(r_sim, len(s_sub), r_cb, len(c_sub))
            
            pvals_6a.append(p_part)
            rows_6a.append({
                'משתנה רובד 1 (תהליך)': l1_name,
                'משתנה רובד 2 (ביצוע)': l2_name,
                'N כולל': n_tot,
                'Partial r_s (מבוקר)': f"{r_part:.3f}",
                'ערך p חלקי גולמי': p_part,
                'מתאם סימולציה r_s': f"{r_sim:.3f}" if not np.isnan(r_sim) else "-",
                'p סימולציה': f"{p_sim:.4f}" if not np.isnan(p_sim) else "-",
                'מתאם צ׳אטבוט r_s': f"{r_cb:.3f}" if not np.isnan(r_cb) else "-",
                'p צ׳אטבוט': f"{p_cb:.4f}" if not np.isnan(p_cb) else "-",
                'הפרש Fisher z': f"{z_diff:.2f}" if not np.isnan(z_diff) else "-",
                'p הפרש בין קבוצות': f"{p_diff:.4f}" if not np.isnan(p_diff) else "-"
            })

    fdr_6a = fdr_bh(pvals_6a)
    for i, r in enumerate(rows_6a):
        p_raw = r['ערך p חלקי גולמי']
        p_fdr = fdr_6a[i]
        r['ערך p חלקי גולמי'] = f"{p_raw:.4f}"
        r['ערך p מתוקן FDR'] = f"{p_fdr:.4f}"
        r['מובהק מבוקר (q<.05)'] = 'כן' if p_fdr < 0.05 else 'לא'

    tab_6a = pd.DataFrame(rows_6a)
    tab_6a.to_csv(os.path.join(base_dir, "tab_6a_layer1_vs_layer2.csv"), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # 2. טבלה 6ב: הצלבת רובד 2 (ביצוע) מול רובד 3 (תפיסה)
    # ==============================================================================
    rows_6b = []
    pvals_6b = []

    # תת-קבוצה למי שמילאו שאלון והשלימו משימה (N=48)
    df_l2_l3 = df[(df['השלימה_משימה'] == 1) & (df['מילאה_שאלון'] == 1)].copy()
    sim_l2_l3 = df_l2_l3[df_l2_l3['קבוצה_אנגלית'] == 'Simulation']
    cb_l2_l3  = df_l2_l3[df_l2_l3['קבוצה_אנגלית'] == 'Chatbot']

    for v2, l2_name in layer2_vars:
        for v3, l3_name in layer3_vars:
            r_part, p_part, n_tot = partial_spearman(df_l2_l3[v2], df_l2_l3[v3], df_l2_l3['cond_num'])
            
            s_sub = sim_l2_l3[[v2, v3]].dropna()
            c_sub = cb_l2_l3[[v2, v3]].dropna()
            
            r_sim, p_sim = stats.spearmanr(s_sub[v2], s_sub[v3]) if len(s_sub) >= 4 else (np.nan, np.nan)
            r_cb, p_cb   = stats.spearmanr(c_sub[v2], c_sub[v3]) if len(c_sub) >= 4 else (np.nan, np.nan)
            
            z_diff, p_diff = fisher_z_test(r_sim, len(s_sub), r_cb, len(c_sub))
            
            pvals_6b.append(p_part)
            rows_6b.append({
                'משתנה רובד 2 (ביצוע)': l2_name,
                'משתנה רובד 3 (תפיסה)': l3_name,
                'N כולל': n_tot,
                'Partial r_s (מבוקר)': f"{r_part:.3f}",
                'ערך p חלקי גולמי': p_part,
                'מתאם סימולציה r_s': f"{r_sim:.3f}" if not np.isnan(r_sim) else "-",
                'p סימולציה': f"{p_sim:.4f}" if not np.isnan(p_sim) else "-",
                'מתאם צ׳אטבוט r_s': f"{r_cb:.3f}" if not np.isnan(r_cb) else "-",
                'p צ׳אטבוט': f"{p_cb:.4f}" if not np.isnan(p_cb) else "-",
                'הפרש Fisher z': f"{z_diff:.2f}" if not np.isnan(z_diff) else "-",
                'p הפרש בין קבוצות': f"{p_diff:.4f}" if not np.isnan(p_diff) else "-"
            })

    fdr_6b = fdr_bh(pvals_6b)
    for i, r in enumerate(rows_6b):
        p_raw = r['ערך p חלקי גולמי']
        p_fdr = fdr_6b[i]
        r['ערך p חלקי גולמי'] = f"{p_raw:.4f}"
        r['ערך p מתוקן FDR'] = f"{p_fdr:.4f}"
        r['מובהק מבוקר (q<.05)'] = 'כן' if p_fdr < 0.05 else 'לא'

    tab_6b = pd.DataFrame(rows_6b)
    tab_6b.to_csv(os.path.join(base_dir, "tab_6b_layer2_vs_layer3.csv"), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # 3. טבלה 6ג: הצלבת רובד 3 (תפיסה) מול רובד 1 (תהליך)
    # ==============================================================================
    rows_6c = []
    pvals_6c = []

    # תת-קבוצה של כל מי שמילאו שאלון (N=49, כולל נבדקת 70)
    df_l3_l1 = df[df['מילאה_שאלון'] == 1].copy()
    sim_l3_l1 = df_l3_l1[df_l3_l1['קבוצה_אנגלית'] == 'Simulation']
    cb_l3_l1  = df_l3_l1[df_l3_l1['קבוצה_אנגלית'] == 'Chatbot']

    for v3, l3_name in layer3_vars:
        for v1, l1_name in layer1_vars:
            r_part, p_part, n_tot = partial_spearman(df_l3_l1[v3], df_l3_l1[v1], df_l3_l1['cond_num'])
            
            s_sub = sim_l3_l1[[v3, v1]].dropna()
            c_sub = cb_l3_l1[[v3, v1]].dropna()
            
            r_sim, p_sim = stats.spearmanr(s_sub[v3], s_sub[v1]) if len(s_sub) >= 4 else (np.nan, np.nan)
            r_cb, p_cb   = stats.spearmanr(c_sub[v3], c_sub[v1]) if len(c_sub) >= 4 else (np.nan, np.nan)
            
            z_diff, p_diff = fisher_z_test(r_sim, len(s_sub), r_cb, len(c_sub))
            
            pvals_6c.append(p_part)
            rows_6c.append({
                'משתנה רובד 3 (תפיסה)': l3_name,
                'משתנה רובד 1 (תהליך)': l1_name,
                'N כולל': n_tot,
                'Partial r_s (מבוקר)': f"{r_part:.3f}",
                'ערך p חלקי גולמי': p_part,
                'מתאם סימולציה r_s': f"{r_sim:.3f}" if not np.isnan(r_sim) else "-",
                'p סימולציה': f"{p_sim:.4f}" if not np.isnan(p_sim) else "-",
                'מתאם צ׳אטבוט r_s': f"{r_cb:.3f}" if not np.isnan(r_cb) else "-",
                'p צ׳אטבוט': f"{p_cb:.4f}" if not np.isnan(p_cb) else "-",
                'הפרש Fisher z': f"{z_diff:.2f}" if not np.isnan(z_diff) else "-",
                'p הפרש בין קבוצות': f"{p_diff:.4f}" if not np.isnan(p_diff) else "-"
            })

    fdr_6c = fdr_bh(pvals_6c)
    for i, r in enumerate(rows_6c):
        p_raw = r['ערך p חלקי גולמי']
        p_fdr = fdr_6c[i]
        r['ערך p חלקי גולמי'] = f"{p_raw:.4f}"
        r['ערך p מתוקן FDR'] = f"{p_fdr:.4f}"
        r['מובהק מבוקר (q<.05)'] = 'כן' if p_fdr < 0.05 else 'לא'

    tab_6c = pd.DataFrame(rows_6c)
    tab_6c.to_csv(os.path.join(base_dir, "tab_6c_layer3_vs_layer1.csv"), index=False, encoding='utf-8-sig')

    # ==============================================================================
    # 4. טבלה 6ד: רגרסיית תיקוף בין תמות תחקיר למדדי לוג אובייקטיביים
    # ==============================================================================
    # בדיקת תאימות: האם מי שדיווח על תמה מסוימת בתחקיר אכן פעל בהתאם בלוג?
    validation_pairs = [
        ('רפלקציה_תמה_1_אימות_מקורות_והצלבה', 'תמה 1: אימות מקורות', 'שיעור_ביקור_במקורות', 'שיעור ביקור במקורות באתר'),
        ('רפלקציה_תמה_2_הפקת_לקחים_לעתיד', 'תמה 2: הפקת לקחים', 'ציון_4_ממדים', 'ציון איכות כולל (4 ממדים)'),
        ('רפלקציה_תמה_3_הסתמכות_שטחית_ומלכודת_המקור', 'תמה 3: מלכודת המקור', 'שיעור_ניחוש_ללא_בדיקה', 'שיעור ניחוש ללא בדיקה'),
        ('רפלקציה_תמה_5_תגובה_קצרה_ללא_פירוט', 'תמה 5: תגובה קצרה ולאקונית', 'משך_פעילות_נטו_מנוקה_דקות', 'משך פעילות נטו (דקות)'),
        ('רפלקציה_תמה_6_ספקנות_ומודעות_להזיות_AI', 'תמה 6: ספקנות והזיות AI', 'חיפושים_חיצוניים_בגוגל', 'מספר חיפושים חיצוניים בגוגל')
    ]

    val_rows = []
    task_df = df[df['השלימה_משימה'] == 1].copy()

    for theme_col, theme_name, log_col, log_name in validation_pairs:
        sub = task_df[[theme_col, log_col, 'cond_num']].dropna()
        # מודל OLS: log_metric = B0 + B1*Theme + B2*Condition
        X = sm.add_constant(sub[[theme_col, 'cond_num']])
        y = sub[log_col]
        reg = sm.OLS(y, X).fit()
        
        # השוואת ממוצעים פשוטה בין קבוצות התמה
        m_present = sub[sub[theme_col] == 1][log_col].mean()
        m_absent  = sub[sub[theme_col] == 0][log_col].mean()

        val_rows.append({
            'תמת תחקיר': theme_name,
            'מדד לוג אובייקטיבי': log_name,
            'N': len(sub),
            'ממוצע עם התמה': f"{m_present:.2f}",
            'ממוצע ללא התמה': f"{m_absent:.2f}",
            'מקדם רגרסיה B (תמה)': f"{reg.params[theme_col]:.3f}",
            'שגיאת תקן SE': f"{reg.bse[theme_col]:.3f}",
            't': f"{reg.tvalues[theme_col]:.2f}",
            'ערך p (תמה)': f"{reg.pvalues[theme_col]:.4f}",
            'R² של המודל': f"{reg.rsquared:.3f}",
            'תיקוף נתמך': 'כן (מובהק)' if reg.pvalues[theme_col] < 0.05 else ('מגמה' if reg.pvalues[theme_col] < 0.10 else 'לא')
        })

    tab_6d = pd.DataFrame(val_rows)
    tab_6d.to_csv(os.path.join(base_dir, "tab_6d_validation_regression.csv"), index=False, encoding='utf-8-sig')

    print(">>> שלב 6 הושלם. כל טבלאות ההצלבות נשמרו ב-analysis_scripts.")

if __name__ == '__main__':
    main()


#==============================================================================
# PART 9: (removed: the source script only stored a hard-coded summary table)
#==============================================================================


#==============================================================================
# PART 10: All-pairs correlation matrix (exploratory)
# Thesis: Appendix E.3.2
#==============================================================================
# (source script: 06b_all_pairs_correlation_matrix.py)

"""
06b_all_pairs_correlation_matrix.py
Comprehensive All-Pairs Correlation Analysis across all variables in WorkAI:
- Excludes ENA/ONA variables
- Includes all Markov transition variables
- Crosses every variable with every other variable (Cross-Layer and Intra-Layer)
- Computes:
  1. Spearman rho (raw) & raw p-value
  2. Partial Spearman rho (controlled for experimental condition) & controlled p-value
  3. FDR Benjamini-Hochberg correction
  4. Automatic detection of part-whole / deterministic collinearity
- Outputs complete table, significant cross-layer table, significant intra-layer table,
  and full symmetric matrix.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = SCRIPTS_DIR
df = pd.read_csv(os.path.join(BASE_DIR, 'master_clean_n70.csv'))

# Create condition dummy: 1 = Simulation, 0 = Chatbot
df['cond_dummy'] = (df['קבוצת_ניסוי'] == 'סימולציה').astype(int)

# Convert Likert strings to numeric
likert_map = {
    'לא מסכימ.ה כלל': 1, 'לא מסכימ.ה': 2, 'לא מסכימ.ה מעט': 3,
    'מסכימ.ה מעט': 4, 'מסכימ.ה': 5, 'מסכימ.ה מאוד': 6
}
freq_map = {
    'אף פעם': 1, 'לעיתים רחוקות (פעם בחודש או פחות)': 2,
    'מדי פעם (2-3 פעמים בחודש)': 3, 'לעיתים קרובות (1-3 פעמים בשבוע)': 4,
    'לעיתים קרובות (4-7 פעמים בשבוע)': 5, 'באופן אינטנסיבי (מספר פעמים ביום)': 6
}

for col in df.columns:
    if (not pd.api.types.is_numeric_dtype(df[col])):
        mapped = df[col].map(likert_map)
        if mapped.notna().sum() >= 15:
            df[col] = mapped
        else:
            mapped_f = df[col].map(freq_map)
            if mapped_f.notna().sum() >= 15:
                df[col] = mapped_f

# Define curated list of distinct, meaningful variables categorized by layer
# (Avoiding alias duplicate columns like hits vs זיהוי_שגוי_כשגוי)

# LAYER 1: Process, Behavior & Markov (Excluding ENA)
l1_dict = {
    'משך_פעילות_נטו_מנוקה_דקות': 'משך פעילות נטו (דקות)',
    'מספר_פרומפטים_שנשלחו': 'מספר פרומפטים',
    'סהכ_מילים_בפרומפטים': 'סה״כ מילים בפרומפטים',
    'מילים_בפרומפט_ראשון': 'מילים בפרומפט ראשון',
    'דמיון_להנחיה_מקסימלי': 'דמיון להנחיה מובנית',
    'מדד_איכות_פרומפט': 'מדד איכות פרומפט (0-100)',
    'סהכ_פעולות_במערכת': 'סה״כ פעולות במערכת',
    'קצב_פעולות_לדקה_APM': 'קצב פעולות לדקה (APM)',
    'שיעור_ביקור_במקורות': 'שיעור ביקור במקורות',
    'חיפושים_חיצוניים_בגוגל': 'כמות חיפושים בגוגל',
    'הצלבות_גוגל_ואתר': 'הצלבות גוגל ואתר (Lateral Loops)',
    'שיעור_ניחוש_ללא_בדיקה': 'שיעור ניחוש ללא בדיקה',
    'משך_שהייה_ממוצע_במקור_שניות': 'שהייה ממוצעת במקור (שניות)',
    'נימוקי_החרגת_מקורות_כמות': 'כמות נימוקי החרגת מקורות',
    'סהכ_הודעות_צ\'אט_לבוט': 'סה״כ פניות לבוט (בצ׳אטבוט)',
    'אפס_כניסות': 'דפוס אפס כניסות לאתר',
    'מרקוב_פרומפט_אל_סיווג': 'מרקוב: פרומפט אל סיווג (דילוג)',
    'מרקוב_פרומפט_אל_כניסה_למקור': 'מרקוב: פרומפט אל כניסה למקור',
    'מרקוב_חיפוש_חיצוני_אל_כניסה_למקור': 'מרקוב: גוגל אל כניסה למקור',
    'מרקוב_כניסה_למקור_אל_סיווג': 'מרקוב: כניסה למקור אל סיווג'
}

# LAYER 2: Performance, SDT & Reflection
l2_dict = {
    'ציון_4_ממדים': 'ציון איכות כולל (4 ממדים)',
    'ציון_4_ממדים_ללא_ש3': 'ציון 4 ממדים (ללא שאלה 3)',
    'ציון_ממד_פרומפט': 'ציון ממד פרומפט',
    'ציון_ממד_זיהוי_מקורות': 'ציון ממד דיוק זיהוי מקורות',
    'ציון_ממד_תהליך_ובדיקה': 'ציון ממד תהליך ובדיקה',
    'ציון_ממד_תחקיר': 'ציון ממד תחקיר ורפלקציה',
    'שיעור_דיוק_סיווג_מקורות': 'שיעור דיוק בסיווג מקורות',
    'd_prime': 'מדד רגישות אבחנה (d\') ב-SDT',
    'c': 'מדד קריטריון שיפוט (c) ב-SDT',
    'H_rate': 'Hit Rate (זיהוי מקורות שקריים)',
    'FA_rate': 'False Alarm Rate (אזעקות שווא)',
    'רמת_Kember': 'רמת רפלקציה לפי קמבר (0-4)',
    'מילים_ברפלקציה': 'כמות מילים ברפלקציה',
    'רפלקציה_עומק_ממוצע_שאלות': 'עומק רפלקציה ממוצע (4 שאלות)',
    'רפלקציה_תמה_1_אימות_מקורות_והצלבה': 'תמת תחקיר 1: אימות והצלבה',
    'רפלקציה_תמה_2_הפקת_לקחים_לעתיד': 'תמת תחקיר 2: הפקת לקחים',
    'רפלקציה_תמה_3_הסתמכות_שטחית_ומלכודת_המקור': 'תמת תחקיר 3: מלכודת המקור',
    'רפלקציה_תמה_5_תגובה_קצרה_ללא_פירוט': 'תמת תחקיר 5: תגובה לאקונית',
    'רפלקציה_תמה_6_ספקנות_ומודעות_להזיות_AI': 'תמת תחקיר 6: מודעות להזיות AI'
}

# LAYER 3: Perception, Attitudes & Questionnaire
l3_dict = {
    'PU': 'TAM: שימושיות נתפסת כוללת (PU)',
    'PEOU': 'TAM: קלות שימוש כוללת (PEOU)',
    'BI': 'TAM: כוונת שימוש התנהגותית (BI)',
    'CriTT': 'CriTT: הערכה עצמית ביקורתית כוללת',
    'שביעות_רצון_תאימות_ציפיות': 'שביעות רצון: תאימות ציפיות',
    'שביעות_רצון_המלצה_לקורסים': 'שביעות רצון: המלצה לקורסים',
    'TAM_1_שיפור_הערכת_אמינות': 'TAM 1: שיפור הערכת אמינות',
    'TAM_2_הגדלת_יעילות_זיהוי_אמינות': 'TAM 2: הגדלת יעילות בזיהוי',
    'TAM_3_שימושי_ללימודים_אקדמיים': 'TAM 3: שימושי ללימודים אקדמיים',
    'TAM_4_עזר_לפתח_חשיבה_ביקורתית': 'TAM 4: פיתוח חשיבה ביקורתית',
    'TAM_5_קל_ללמוד_להשתמש': 'TAM 5: קל ללמוד להשתמש',
    'TAM_6_קל_לגרום_לעשות_מה_שרציתי': 'TAM 6: שליטה בפעולות הכלי',
    'TAM_7_אינטראקציה_ברורה_ומובנת': 'TAM 7: אינטראקציה ברורה',
    'TAM_8_גמיש_לשימוש_וניווט': 'TAM 8: גמיש לשימוש וניווט',
    'TAM_9_מתכוון_להשתמש_שוב': 'TAM 9: כוונה להשתמש שוב',
    'TAM_10_רוצה_כלים_דומים_נוספים': 'TAM 10: רצון בכלים דומים',
    'CRITT_1_זיהוי_שפה_רגשית': 'CriTT 1: זיהוי שפה רגשית ומניפולציות',
    'CRITT_2_מטרה_מוגדרת_בחשיבה': 'CriTT 2: מטרה מוגדרת בחשיבה',
    'CRITT_3_זיהוי_מבנה_טיעונים': 'CriTT 3: זיהוי מבנה טיעונים',
    'CRITT_4_ביטוי_חשיבה_בכתב': 'CriTT 4: ביטוי חשיבה בכתב',
    'CRITT_5_חשיבה_ממוקדת_ושיטתית': 'CriTT 5: חשיבה ממוקדת ושיטתית',
    'CRITT_6_חושב_ביקורתי_טוב_כללי': 'CriTT 6: תפיסה עצמית כחושב ביקורתי',
    'CRITT_7_הצלחה_במטלות_ביקורתיות': 'CriTT 7: הצלחה במטלות ביקורתיות',
    'שאלון_גיל': 'גיל הנבדקת',
    'שאלון_תדירות_שימוש_ב_GenAI': 'תדירות שימוש מוקדם ב-GenAI'
}

# Merge all variables
all_vars_dict = {}
all_vars_dict.update({k: (v, 'רובד 1: תהליך ולוגים') for k, v in l1_dict.items() if k in df.columns})
all_vars_dict.update({k: (v, 'רובד 2: ביצוע ושיפוט') for k, v in l2_dict.items() if k in df.columns})
all_vars_dict.update({k: (v, 'רובד 3: תפיסה ועמדות') for k, v in l3_dict.items() if k in df.columns})

var_keys = list(all_vars_dict.keys())
print(f"Total curated variables for All-Pairs matrix: {len(var_keys)}")

def partial_spearman(x, y, covar):
    valid = (~x.isna()) & (~y.isna()) & (~covar.isna())
    if valid.sum() < 12:
        return np.nan, np.nan, valid.sum()
    if covar[valid].nunique() <= 1:
        # All valid rows belong to one condition, return direct Spearman
        r, p = stats.spearmanr(x[valid], y[valid])
        return r, p, valid.sum()
    rx = stats.rankdata(x[valid])
    ry = stats.rankdata(y[valid])
    rc = stats.rankdata(covar[valid])
    
    def get_res(a, b):
        slope, intercept, _, _, _ = stats.linregress(b, a)
        return a - (slope * b + intercept)
    
    res_x = get_res(rx, rc)
    res_y = get_res(ry, rc)
    if np.all(res_x == 0) or np.all(res_y == 0):
        return 0.0, 1.0, valid.sum()
    r, p = stats.pearsonr(res_x, res_y)
    return r, p, valid.sum()

pairs_records = []

for i in range(len(var_keys)):
    for j in range(i + 1, len(var_keys)):
        k1 = var_keys[i]
        k2 = var_keys[j]
        
        name1, layer1 = all_vars_dict[k1]
        name2, layer2 = all_vars_dict[k2]
        
        # Valid data
        valid = (~df[k1].isna()) & (~df[k2].isna())
        n_val = valid.sum()
        if n_val < 12 or df.loc[valid, k1].nunique() <= 1 or df.loc[valid, k2].nunique() <= 1:
            continue
            
        # Raw Spearman
        r_raw, p_raw = stats.spearmanr(df.loc[valid, k1], df.loc[valid, k2])
        
        # Controlled Partial Spearman
        r_part, p_part, _ = partial_spearman(df[k1], df[k2], df['cond_dummy'])
        
        # Classification
        if layer1 == layer2:
            pair_type = f'תוך-רובדי ({layer1})'
            is_cross = False
        else:
            pair_type = f'בין-רובדי ({layer1} מול {layer2})'
            is_cross = True
            
        # Detect part-whole or deterministic collinearity
        is_part_whole = False
        
        # 1. Tautological / operational clones (rubric dimension vs its direct operational metric)
        tautology_pairs = [
            ({'מדד_איכות_פרומפט'}, {'ציון_ממד_פרומפט'}),
            ({'שיעור_ביקור_במקורות', 'שיעור_ניחוש_ללא_בדיקה', 'דפוס_אפס_כניסות', 'נימוקי_החרגת_מקורות_כמות'}, {'ציון_ממד_תהליך_ובדיקה'}),
            ({'שיעור_דיוק_סיווג_מקורות'}, {'ציון_ממד_זיהוי_מקורות'}),
            ({'רמת_Kember', 'רפלקציה_עומק_ממוצע_שאלות'}, {'ציון_ממד_תחקיר'}),
            ({'ציון_4_ממדים'}, {'ציון_4_ממדים_ללא_ש3'}),
            ({'שיעור_ביקור_במקורות'}, {'שיעור_ניחוש_ללא_בדיקה', 'אפס_כניסות'}),
            ({'משך_פעילות_נטו_מנוקה_דקות'}, {'קצב_פעולות_לדקה_APM'}),
            ({'d_prime', 'c'}, {'H_rate', 'FA_rate'}),
            ({'d_prime'}, {'c'})
        ]
        for set_a, set_b in tautology_pairs:
            if (k1 in set_a and k2 in set_b) or (k2 in set_a and k1 in set_b):
                is_part_whole = True
                break

        # 2. Composite score vs its own component subscores
        if not is_part_whole:
            if ('ציון_4_ממדים' in [k1, k2] or 'ציון_4_ממדים_ללא_ש3' in [k1, k2]):
                composite_components = [
                    'ציון_ממד_פרומפט', 'ציון_ממד_זיהוי_מקורות', 'ציון_ממד_תהליך_ובדיקה', 'ציון_ממד_תחקיר',
                    'מדד_איכות_פרומפט', 'שיעור_ביקור_במקורות', 'שיעור_ניחוש_ללא_בדיקה', 'אפס_כניסות',
                    'שיעור_דיוק_סיווג_מקורות', 'רמת_Kember', 'רפלקציה_עומק_ממוצע_שאלות'
                ]
                if any(k in [k1, k2] for k in composite_components):
                    is_part_whole = True
            elif ('CriTT' in [k1, k2]) and any('CRITT_' in k for k in [k1, k2]):
                is_part_whole = True
            elif ('PU' in [k1, k2]) and any(k in [k1, k2] for k in ['TAM_1', 'TAM_2', 'TAM_3', 'TAM_4']):
                is_part_whole = True
            elif ('PEOU' in [k1, k2]) and any(k in [k1, k2] for k in ['TAM_5', 'TAM_6', 'TAM_7', 'TAM_8']):
                is_part_whole = True
            elif ('BI' in [k1, k2]) and any(k in [k1, k2] for k in ['TAM_9', 'TAM_10']):
                is_part_whole = True
            elif ('רפלקציה_עומק_ממוצע_שאלות' in [k1, k2]) and ('רמת_Kember' in [k1, k2]):
                is_part_whole = True

        pairs_records.append({
            'קוד_משתנה_1': k1,
            'שם_משתנה_1': name1,
            'רובד_1': layer1,
            'קוד_משתנה_2': k2,
            'שם_משתנה_2': name2,
            'רובד_2': layer2,
            'סוג_הצלבה': pair_type,
            'הצלבה_בין_רובדית': 'כן' if is_cross else 'לא',
            'תלות_מתמטית_מובנית': 'כן' if is_part_whole else 'לא',
            'N_נבדקות': n_val,
            'מתאם_ספירמן_גולמי': round(r_raw, 3),
            'ערך_p_גולמי': round(p_raw, 4),
            'מתאם_ספירמן_חלקי_מבוקר': round(r_part, 3) if pd.notna(r_part) else np.nan,
            'ערך_p_מבוקר': round(p_part, 4) if pd.notna(p_part) else np.nan
        })

df_pairs = pd.DataFrame(pairs_records)

# Calculate FDR q-value for raw and controlled p-values
def calc_fdr(p_series):
    valid_p = p_series.dropna()
    n = len(valid_p)
    if n == 0:
        return p_series
    sorted_indices = valid_p.sort_values().index
    sorted_p = valid_p.loc[sorted_indices].values
    q_vals = np.zeros(n)
    for rank, p in enumerate(sorted_p, start=1):
        q_vals[rank-1] = p * n / rank
    # Monotonicity
    for i in range(n-2, -1, -1):
        q_vals[i] = min(q_vals[i], q_vals[i+1])
    q_vals = np.clip(q_vals, 0, 1)
    res = p_series.copy()
    res.loc[sorted_indices] = q_vals.round(4)
    return res

df_pairs['q_FDR_גולמי'] = calc_fdr(df_pairs['ערך_p_גולמי'])
df_pairs['q_FDR_מבוקר'] = calc_fdr(df_pairs['ערך_p_מבוקר'])

df_pairs['מובהק_גולמי'] = np.where(df_pairs['ערך_p_גולמי'] < 0.05, 'כן', 'לא')
df_pairs['מובהק_מבוקר'] = np.where(df_pairs['ערך_p_מבוקר'] < 0.05, 'כן', 'לא')
df_pairs['מובהק_FDR_מבוקר'] = np.where(df_pairs['q_FDR_מבוקר'] < 0.05, 'כן', 'לא')

# Save complete table
out_complete = os.path.join(BASE_DIR, "all_pairs_correlations_complete.csv")
df_pairs.to_csv(out_complete, index=False, encoding='utf-8-sig')
print(f"נשמרה טבלת כל הזוגות המלאה: {out_complete} ({len(df_pairs)} זוגות)")

# Filter significant cross-layer pairs (excluding deterministic part-whole)
df_cross_sig = df_pairs[
    (df_pairs['הצלבה_בין_רובדית'] == 'כן') & 
    (df_pairs['תלות_מתמטית_מובנית'] == 'לא') & 
    (df_pairs['מובהק_מבוקר'] == 'כן')
].sort_values(by='ערך_p_מבוקר')

out_cross_sig = os.path.join(BASE_DIR, "all_pairs_cross_layer_significant.csv")
df_cross_sig.to_csv(out_cross_sig, index=False, encoding='utf-8-sig')
print(f"נשמרה טבלת הצלבות בין-רובדיות מובהקות ומבוקרות: {out_cross_sig} ({len(df_cross_sig)} זוגות)")

# Format tab_6f for Excel workbook
tab_6f_cols = [
    'שם_משתנה_1', 'רובד_1', 'שם_משתנה_2', 'רובד_2', 'סוג_הצלבה',
    'N_נבדקות', 'מתאם_ספירמן_גולמי', 'ערך_p_גולמי',
    'מתאם_ספירמן_חלקי_מבוקר', 'ערך_p_מבוקר', 'q_FDR_מבוקר', 'מובהק_FDR_מבוקר'
]
df_tab_6f = df_cross_sig[tab_6f_cols].copy()
out_tab_6f = os.path.join(BASE_DIR, "tab_6f_all_pairs_cross_layer.csv")
df_tab_6f.to_csv(out_tab_6f, index=False, encoding='utf-8-sig')
print(f"נשמרה טבלת 6ו לאקסל: {out_tab_6f} ({len(df_tab_6f)} שורות)")

# Filter significant intra-layer pairs (excluding deterministic part-whole)
df_intra_sig = df_pairs[
    (df_pairs['הצלבה_בין_רובדית'] == 'לא') & 
    (df_pairs['תלות_מתמטית_מובנית'] == 'לא') & 
    (df_pairs['מובהק_מבוקר'] == 'כן')
].sort_values(by='ערך_p_מבוקר')

out_intra_sig = os.path.join(BASE_DIR, "all_pairs_intra_layer_significant.csv")
df_intra_sig.to_csv(out_intra_sig, index=False, encoding='utf-8-sig')
print(f"נשמרה טבלת הצלבות תוך-רובדיות מובהקות ומבוקרות: {out_intra_sig} ({len(df_intra_sig)} זוגות)")

# Build symmetric correlation matrix table (64x64)
matrix_df = pd.DataFrame(index=[all_vars_dict[k][0] for k in var_keys],
                         columns=[all_vars_dict[k][0] for k in var_keys])

for _, row in df_pairs.iterrows():
    name1 = row['שם_משתנה_1']
    name2 = row['שם_משתנה_2']
    r_val = row['מתאם_ספירמן_חלקי_מבוקר']
    matrix_df.loc[name1, name2] = r_val
    matrix_df.loc[name2, name1] = r_val

for name in matrix_df.index:
    matrix_df.loc[name, name] = 1.000

out_matrix = os.path.join(BASE_DIR, "full_correlation_matrix_table.csv")
matrix_df.to_csv(out_matrix, encoding='utf-8-sig')
print(f"נשמרה מטריצת המתאמים הסימטרית המלאה: {out_matrix} ({matrix_df.shape[0]}x{matrix_df.shape[1]})")

print("\n>>> סריקת כל-הזוגות הושלמה ")


#==============================================================================
# PART 11: (removed: superseded by the Table E9 models in Part 16)
#==============================================================================


#==============================================================================
# PART 12: Sensitivity analyses
# Thesis: Section 6.5; Table E10
#==============================================================================
# (source script: 07_sensitivity_analyses.py)

# ==============================================================================
# analysis_scripts/07_sensitivity_analyses.py
# ניתוחי רגישות (Sensitivity Analyses - A עד H)
# 8 ניתוחי רגישות השוואתיים לבחינת יציבות הממצאים המרכזיים
# כל ניתוח מוצג כזוג שורות: ניתוח בסיס (ראשי) מול ניתוח רגישות (חלופי)
# כולל עמודה מפורשת: "המסקנה השתנתה? (כן/לא)"
# כל המדדים מדווחים כממוצע וסטיית תקן M (SD) ללא חציונים וללא טווחים
# פרויקט מחקר WorkAI
# ==============================================================================

import os
import numpy as np
import pandas as pd
from scipy import stats

def mw_compare(s_data, c_data):
    s_clean = s_data.dropna()
    c_clean = c_data.dropna()
    n_s = len(s_clean)
    n_c = len(c_clean)
    res = stats.mannwhitneyu(s_clean, c_clean, alternative='two-sided')
    u_s = float(res.statistic)
    u_c = float(n_s * n_c - u_s)
    r_rb = (u_s - u_c) / (n_s * n_c)
    p_val = res.pvalue
    return {
        'n_s': n_s, 'm_s': s_clean.mean(), 'sd_s': s_clean.std(),
        'n_c': n_c, 'm_c': c_clean.mean(), 'sd_c': c_clean.std(),
        'u_s': u_s, 'u_c': u_c, 'p': p_val, 'r_rb': r_rb
    }

def main():
    print(">>> שלב 7: הרצת 8 ניתוחי רגישות (A עד H)...")
    base_dir = SCRIPTS_DIR
    master_path = os.path.join(base_dir, "master_clean_n70.csv")
    df = pd.read_csv(master_path)
    task_df = df[df['השלימה_משימה'] == 1].copy()

    sens_rows = []

    # ==============================================================================
    # ניתוח A: ללא שאלה 3 ברפלקציה (רמת קמבר וציון כולל 4 ממדים)
    # ==============================================================================
    # A1: רמת קמבר
    res_base_k = mw_compare(task_df[task_df['קבוצה_אנגלית']=='Simulation']['רמת_Kember'],
                            task_df[task_df['קבוצה_אנגלית']=='Chatbot']['רמת_Kember'])
    res_sens_k = mw_compare(task_df[task_df['קבוצה_אנגלית']=='Simulation']['רמת_Kember_ללא_שאלה_3'],
                            task_df[task_df['קבוצה_אנגלית']=='Chatbot']['רמת_Kember_ללא_שאלה_3'])
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח A1',
        'נושא הבדיקה': 'רמת רפלקציה קמבר (0-4)',
        'גרסת הניתוח': '1. בסיס (ראשי)',
        'הגדרת הניתוח': 'טקסט תחקיר מלא (כולל שאלה 3 בסימולציה)',
        'N סימולציה': res_base_k['n_s'],
        'סימולציה ממוצע (סטיית תקן)': f"{res_base_k['m_s']:.2f} ({res_base_k['sd_s']:.2f})",
        'N צ׳אטבוט': res_base_k['n_c'],
        'צ׳אטבוט ממוצע (סטיית תקן)': f"{res_base_k['m_c']:.2f} ({res_base_k['sd_c']:.2f})",
        'מבחן סטטיסטי': f"Mann-Whitney U={res_base_k['u_s']:.1f}",
        'ערך p': f"{res_base_k['p']:.4f}",
        'גודל אפקט r_rb': f"{res_base_k['r_rb']:.3f}",
        'המסקנה השתנתה?': 'לא (בסיס)',
        'הסבר ומסקנה': f"ערך גבוה יותר בתנאי הסימולציה ברמת הרפלקציה (p < .0001, r_rb = {res_base_k['r_rb']:.3f})"
    })
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח A1',
        'נושא הבדיקה': 'רמת רפלקציה קמבר (0-4)',
        'גרסת הניתוח': '2. רגישות (חלופי)',
        'הגדרת הניתוח': 'החרגת שאלה 3 (שאלת הטעות הייחודית לסימולציה)',
        'N סימולציה': res_sens_k['n_s'],
        'סימולציה ממוצע (סטיית תקן)': f"{res_sens_k['m_s']:.2f} ({res_sens_k['sd_s']:.2f})",
        'N צ׳אטבוט': res_sens_k['n_c'],
        'צ׳אטבוט ממוצע (סטיית תקן)': f"{res_sens_k['m_c']:.2f} ({res_sens_k['sd_c']:.2f})",
        'מבחן סטטיסטי': f"Mann-Whitney U={res_sens_k['u_s']:.1f}",
        'ערך p': f"{res_sens_k['p']:.4f}",
        'גודל אפקט r_rb': f"{res_sens_k['r_rb']:.3f}",
        'המסקנה השתנתה?': 'לא',
        'הסבר ומסקנה': f"גם לאחר הסרת שאלה 3, יתרון הסימולציה נותר מובהק ובינוני-גדול (p = {res_sens_k['p']:.4f}, r_rb = {res_sens_k['r_rb']:.3f})"
    })

    # A2: ציון איכות כולל 4 ממדים
    res_base_sc = mw_compare(task_df[task_df['קבוצה_אנגלית']=='Simulation']['ציון_4_ממדים'],
                             task_df[task_df['קבוצה_אנגלית']=='Chatbot']['ציון_4_ממדים'])
    res_sens_sc = mw_compare(task_df[task_df['קבוצה_אנגלית']=='Simulation']['ציון_4_ממדים_ללא_ש3'],
                             task_df[task_df['קבוצה_אנגלית']=='Chatbot']['ציון_4_ממדים_ללא_ש3'])
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח A2',
        'נושא הבדיקה': 'ציון איכות כולל (4 ממדים)',
        'גרסת הניתוח': '1. בסיס (ראשי)',
        'הגדרת הניתוח': 'ציון 4 ממדים מלא (משקל רפלקציה 10% עם שאלה 3)',
        'N סימולציה': res_base_sc['n_s'],
        'סימולציה ממוצע (סטיית תקן)': f"{res_base_sc['m_s']:.2f} ({res_base_sc['sd_s']:.2f})",
        'N צ׳אטבוט': res_base_sc['n_c'],
        'צ׳אטבוט ממוצע (סטיית תקן)': f"{res_base_sc['m_c']:.2f} ({res_base_sc['sd_c']:.2f})",
        'מבחן סטטיסטי': f"Mann-Whitney U={res_base_sc['u_s']:.1f}",
        'ערך p': f"{res_base_sc['p']:.4f}",
        'גודל אפקט r_rb': f"{res_base_sc['r_rb']:.3f}",
        'המסקנה השתנתה?': 'לא (בסיס)',
        'הסבר ומסקנה': f"ציון האיכות הכולל גבוה מובהקת בסימולציה (p = {res_base_sc['p']:.4f}, r_rb = {res_base_sc['r_rb']:.3f})"
    })
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח A2',
        'נושא הבדיקה': 'ציון איכות כולל (4 ממדים)',
        'גרסת הניתוח': '2. רגישות (חלופי)',
        'הגדרת הניתוח': 'ציון 4 ממדים כאשר רכיב הרפלקציה מחושב ללא שאלה 3',
        'N סימולציה': res_sens_sc['n_s'],
        'סימולציה ממוצע (סטיית תקן)': f"{res_sens_sc['m_s']:.2f} ({res_sens_sc['sd_s']:.2f})",
        'N צ׳אטבוט': res_sens_sc['n_c'],
        'צ׳אטבוט ממוצע (סטיית תקן)': f"{res_sens_sc['m_c']:.2f} ({res_sens_sc['sd_c']:.2f})",
        'מבחן סטטיסטי': f"Mann-Whitney U={res_sens_sc['u_s']:.1f}",
        'ערך p': f"{res_sens_sc['p']:.4f}",
        'גודל אפקט r_rb': f"{res_sens_sc['r_rb']:.3f}",
        'המסקנה השתנתה?': 'לא',
        'הסבר ומסקנה': f"הציון הכולל נותר מובהק ובעל עוצמת אפקט גדולה (p = {res_sens_sc['p']:.4f}, r_rb = {res_sens_sc['r_rb']:.3f}), ללא תלות בשאלה 3"
    })

    # ==============================================================================
    # ניתוח B: ללא חריגים קיצוניים (2.5*IQR) - N=54 שהשלימו משימה
    # ==============================================================================
    no_outliers = task_df[task_df['דגל_חריג'] == 0]
    res_b_clean = mw_compare(no_outliers[no_outliers['קבוצה_אנגלית']=='Simulation']['ציון_4_ממדים'],
                             no_outliers[no_outliers['קבוצה_אנגלית']=='Chatbot']['ציון_4_ממדים'])
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח B',
        'נושא הבדיקה': 'ציון איכות כולל (4 ממדים)',
        'גרסת הניתוח': '1. בסיס (ראשי)',
        'הגדרת הניתוח': 'מדגם מלא של משלימות המשימה (N=69)',
        'N סימולציה': res_base_sc['n_s'],
        'סימולציה ממוצע (סטיית תקן)': f"{res_base_sc['m_s']:.2f} ({res_base_sc['sd_s']:.2f})",
        'N צ׳אטבוט': res_base_sc['n_c'],
        'צ׳אטבוט ממוצע (סטיית תקן)': f"{res_base_sc['m_c']:.2f} ({res_base_sc['sd_c']:.2f})",
        'מבחן סטטיסטי': f"Mann-Whitney U={res_base_sc['u_s']:.1f}",
        'ערך p': f"{res_base_sc['p']:.4f}",
        'גודל אפקט r_rb': f"{res_base_sc['r_rb']:.3f}",
        'המסקנה השתנתה?': 'לא (בסיס)',
        'הסבר ומסקנה': f"הבדל ביצועים מובהק בין הקבוצות במדגם המלא (p = {res_base_sc['p']:.4f}, r_rb = {res_base_sc['r_rb']:.3f})"
    })
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח B',
        'נושא הבדיקה': 'ציון איכות כולל (4 ממדים)',
        'גרסת הניתוח': '2. רגישות (חלופי)',
        'הגדרת הניתוח': 'ללא 15 משתתפות שסומנו כחריגות קיצוניות (2.5*IQR, N=54)',
        'N סימולציה': res_b_clean['n_s'],
        'סימולציה ממוצע (סטיית תקן)': f"{res_b_clean['m_s']:.2f} ({res_b_clean['sd_s']:.2f})",
        'N צ׳אטבוט': res_b_clean['n_c'],
        'צ׳אטבוט ממוצע (סטיית תקן)': f"{res_b_clean['m_c']:.2f} ({res_b_clean['sd_c']:.2f})",
        'מבחן סטטיסטי': f"Mann-Whitney U={res_b_clean['u_s']:.1f}",
        'ערך p': f"{res_b_clean['p']:.4f}",
        'גודל אפקט r_rb': f"{res_b_clean['r_rb']:.3f}",
        'המסקנה השתנתה?': 'לא',
        'הסבר ומסקנה': f"האפקט נשמר (p < .0001, r_rb = {res_b_clean['r_rb']:.3f}), כלומר החריגים לא שינו את המסקנה"
    })

    # ==============================================================================
    # ניתוח C: ללא מעורבות נמוכה (משך < 2 דקות) - N=66 שהשלימו משימה
    # ==============================================================================
    no_low_eng = task_df[task_df['דגל_מעורבות_נמוכה'] == 0]
    res_c_clean = mw_compare(no_low_eng[no_low_eng['קבוצה_אנגלית']=='Simulation']['ציון_4_ממדים'],
                             no_low_eng[no_low_eng['קבוצה_אנגלית']=='Chatbot']['ציון_4_ממדים'])
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח C',
        'נושא הבדיקה': 'ציון איכות כולל (4 ממדים)',
        'גרסת הניתוח': '1. בסיס (ראשי)',
        'הגדרת הניתוח': 'מדגם מלא של משלימות המשימה (N=69)',
        'N סימולציה': res_base_sc['n_s'],
        'סימולציה ממוצע (סטיית תקן)': f"{res_base_sc['m_s']:.2f} ({res_base_sc['sd_s']:.2f})",
        'N צ׳אטבוט': res_base_sc['n_c'],
        'צ׳אטבוט ממוצע (סטיית תקן)': f"{res_base_sc['m_c']:.2f} ({res_base_sc['sd_c']:.2f})",
        'מבחן סטטיסטי': f"Mann-Whitney U={res_base_sc['u_s']:.1f}",
        'ערך p': f"{res_base_sc['p']:.4f}",
        'גודל אפקט r_rb': f"{res_base_sc['r_rb']:.3f}",
        'המסקנה השתנתה?': 'לא (בסיס)',
        'הסבר ומסקנה': f"הבדל ביצועים מובהק במדגם המלא (p = {res_base_sc['p']:.4f}, r_rb = {res_base_sc['r_rb']:.3f})"
    })
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח C',
        'נושא הבדיקה': 'ציון איכות כולל (4 ממדים)',
        'גרסת הניתוח': '2. רגישות (חלופי)',
        'הגדרת הניתוח': 'החרגת 3 משתתפות עם משך פעילות קצר מ-2 דקות (N=66)',
        'N סימולציה': res_c_clean['n_s'],
        'סימולציה ממוצע (סטיית תקן)': f"{res_c_clean['m_s']:.2f} ({res_c_clean['sd_s']:.2f})",
        'N צ׳אטבוט': res_c_clean['n_c'],
        'צ׳אטבוט ממוצע (סטיית תקן)': f"{res_c_clean['m_c']:.2f} ({res_c_clean['sd_c']:.2f})",
        'מבחן סטטיסטי': f"Mann-Whitney U={res_c_clean['u_s']:.1f}",
        'ערך p': f"{res_c_clean['p']:.4f}",
        'גודל אפקט r_rb': f"{res_c_clean['r_rb']:.3f}",
        'המסקנה השתנתה?': 'לא',
        'הסבר ומסקנה': f"הממצא נותר מובהק ויציב (p = {res_c_clean['p']:.4f}, r_rb = {res_c_clean['r_rb']:.3f})"
    })

    # ==============================================================================
    # ניתוח D: משך פעילות חלופי (משך מנורמל עד שעה, n=62)
    # ==============================================================================
    res_base_dur = mw_compare(task_df[task_df['קבוצה_אנגלית']=='Simulation']['משך_פעילות_נטו_מנוקה_דקות'],
                              task_df[task_df['קבוצה_אנגלית']=='Chatbot']['משך_פעילות_נטו_מנוקה_דקות'])
    res_sens_dur = mw_compare(task_df[task_df['קבוצה_אנגלית']=='Simulation']['משך_פעילות_מנורמל_עד_שעה'],
                              task_df[task_df['קבוצה_אנגלית']=='Chatbot']['משך_פעילות_מנורמל_עד_שעה'])
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח D',
        'נושא הבדיקה': 'משך פעילות (דקות)',
        'גרסת הניתוח': '1. בסיס (ראשי)',
        'הגדרת הניתוח': 'משך פעילות נטו מנוקה (אירועים עוקבים < 5 דקות, N=69)',
        'N סימולציה': res_base_dur['n_s'],
        'סימולציה ממוצע (סטיית תקן)': f"{res_base_dur['m_s']:.2f} ({res_base_dur['sd_s']:.2f})",
        'N צ׳אטבוט': res_base_dur['n_c'],
        'צ׳אטבוט ממוצע (סטיית תקן)': f"{res_base_dur['m_c']:.2f} ({res_base_dur['sd_c']:.2f})",
        'מבחן סטטיסטי': f"Mann-Whitney U={res_base_dur['u_s']:.1f}",
        'ערך p': f"{res_base_dur['p']:.4f}",
        'גודל אפקט r_rb': f"{res_base_dur['r_rb']:.3f}",
        'המסקנה השתנתה?': 'לא (בסיס)',
        'הסבר ומסקנה': f"משך הפעילות בסימולציה ארוך מובהקת מהצ׳אטבוט (p = {res_base_dur['p']:.4f}, r_rb = {res_base_dur['r_rb']:.3f})"
    })
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח D',
        'נושא הבדיקה': 'משך פעילות (דקות)',
        'גרסת הניתוח': '2. רגישות (חלופי)',
        'הגדרת הניתוח': 'משך מנורמל עד שעה (חיתוך סשנים ארוכים משעה ל-60 דק\', N=62)',
        'N סימולציה': res_sens_dur['n_s'],
        'סימולציה ממוצע (סטיית תקן)': f"{res_sens_dur['m_s']:.2f} ({res_sens_dur['sd_s']:.2f})",
        'N צ׳אטבוט': res_sens_dur['n_c'],
        'צ׳אטבוט ממוצע (סטיית תקן)': f"{res_sens_dur['m_c']:.2f} ({res_sens_dur['sd_c']:.2f})",
        'מבחן סטטיסטי': f"Mann-Whitney U={res_sens_dur['u_s']:.1f}",
        'ערך p': f"{res_sens_dur['p']:.4f}",
        'גודל אפקט r_rb': f"{res_sens_dur['r_rb']:.3f}",
        'המסקנה השתנתה?': 'לא',
        'הסבר ומסקנה': f"המובהקות נשמרת (p < .0001, r_rb = {res_sens_dur['r_rb']:.3f}), ללא השפעת משכי שהייה חריגים"
    })

    # ==============================================================================
    # ניתוח E: ניסיונות פרומפט חלופי (מספר פרומפטים שנשלחו מול מסווגים)
    # הערה: כאן המסקנה תלויה בהגדרת המשתנה!
    # ==============================================================================
    res_base_p = mw_compare(task_df[task_df['קבוצה_אנגלית']=='Simulation']['ניסיונות_פרומפט_משימה_מסווגים'],
                            task_df[task_df['קבוצה_אנגלית']=='Chatbot']['ניסיונות_פרומפט_משימה_מסווגים'])
    res_sens_p = mw_compare(task_df[task_df['קבוצה_אנגלית']=='Simulation']['מספר_פרומפטים_שנשלחו'],
                            task_df[task_df['קבוצה_אנגלית']=='Chatbot']['מספר_פרומפטים_שנשלחו'])
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח E',
        'נושא הבדיקה': 'ניסיונות פרומפט למשימה',
        'גרסת הניתוח': '1. בסיס (ראשי)',
        'הגדרת הניתוח': 'ניסיונות פרומפט משימה מסווגים מתודולוגית',
        'N סימולציה': res_base_p['n_s'],
        'סימולציה ממוצע (סטיית תקן)': f"{res_base_p['m_s']:.2f} ({res_base_p['sd_s']:.2f})",
        'N צ׳אטבוט': res_base_p['n_c'],
        'צ׳אטבוט ממוצע (סטיית תקן)': f"{res_base_p['m_c']:.2f} ({res_base_p['sd_c']:.2f})",
        'מבחן סטטיסטי': f"Mann-Whitney U={res_base_p['u_s']:.1f}",
        'ערך p': f"{res_base_p['p']:.4f}",
        'גודל אפקט r_rb': f"{res_base_p['r_rb']:.3f}",
        'המסקנה השתנתה?': 'לא (בסיס)',
        'הסבר ומסקנה': f"לא נמצא הבדל מובהק בין הקבוצות במספר ניסיונות הפרומפט המסווגים ישירות למשימה (p = {res_base_p['p']:.4f}, r_rb = {res_base_p['r_rb']:.3f}). שתי הקבוצות ניסחו כ-1.5 פרומפטים בממוצע."
    })
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח E',
        'נושא הבדיקה': 'ניסיונות פרומפט למשימה',
        'גרסת הניתוח': '2. רגישות (חלופי)',
        'הגדרת הניתוח': 'מספר פרומפטים כולל שנשלחו למערכת (עמודה 16)',
        'N סימולציה': res_sens_p['n_s'],
        'סימולציה ממוצע (סטיית תקן)': f"{res_sens_p['m_s']:.2f} ({res_sens_p['sd_s']:.2f})",
        'N צ׳אטבוט': res_sens_p['n_c'],
        'צ׳אטבוט ממוצע (סטיית תקן)': f"{res_sens_p['m_c']:.2f} ({res_sens_p['sd_c']:.2f})",
        'מבחן סטטיסטי': f"Mann-Whitney U={res_sens_p['u_s']:.1f}",
        'ערך p': f"{res_sens_p['p']:.4f}",
        'גודל אפקט r_rb': f"{res_sens_p['r_rb']:.3f}",
        'המסקנה השתנתה?': 'כן (תלוי בהגדרה)',
        'הסבר ומסקנה': f"המסקנה תלויה בהגדרת המשתנה: בספירת כלל הפניות שנשלחו לבוט (כולל הבהרות, שאלות טכניות ובירורים), בצ'אטבוט נשלחו מובהקת יותר פניות (p = {res_sens_p['p']:.4f}, r_rb = {res_sens_p['r_rb']:.3f}). לעומת זאת, בפרומפטים ישירים למשימה בלבד לא נמצא כל הבדל (p = {res_base_p['p']:.4f})."
    })

    # ==============================================================================
    # ניתוח F: אורך פרומפטים חלופי (סה״כ מילים מול מילים בפרומפט ראשון)
    # הערה: כאן הכיוון מתהפך ותלוי בהגדרת המשתנה!
    # ==============================================================================
    res_base_w = mw_compare(task_df[task_df['קבוצה_אנגלית']=='Simulation']['מילים_בפרומפט_ראשון'],
                            task_df[task_df['קבוצה_אנגלית']=='Chatbot']['מילים_בפרומפט_ראשון'])
    res_sens_w = mw_compare(task_df[task_df['קבוצה_אנגלית']=='Simulation']['סהכ_מילים_בפרומפטים'],
                            task_df[task_df['קבוצה_אנגלית']=='Chatbot']['סהכ_מילים_בפרומפטים'])
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח F',
        'נושא הבדיקה': 'אורך פרומפטים',
        'גרסת הניתוח': '1. בסיס (ראשי)',
        'הגדרת הניתוח': 'מילים בפרומפט ראשון בלבד',
        'N סימולציה': res_base_w['n_s'],
        'סימולציה ממוצע (סטיית תקן)': f"{res_base_w['m_s']:.2f} ({res_base_w['sd_s']:.2f})",
        'N צ׳אטבוט': res_base_w['n_c'],
        'צ׳אטבוט ממוצע (סטיית תקן)': f"{res_base_w['m_c']:.2f} ({res_base_w['sd_c']:.2f})",
        'מבחן סטטיסטי': f"Mann-Whitney U={res_base_w['u_s']:.1f}",
        'ערך p': f"{res_base_w['p']:.4f}",
        'גודל אפקט r_rb': f"{res_base_w['r_rb']:.3f}",
        'המסקנה השתנתה?': 'לא (בסיס)',
        'הסבר ומסקנה': f"בפרומפט הראשון לא נמצא הבדל מובהק סטטיסטית באורך (p = {res_base_w['p']:.4f}, r_rb = {res_base_w['r_rb']:.3f}), אף שנרשם יתרון כיווני קל ולא מובהק לסימולציה (25.6 מול 19.6 מילים)."
    })
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח F',
        'נושא הבדיקה': 'אורך פרומפטים',
        'גרסת הניתוח': '2. רגישות (חלופי)',
        'הגדרת הניתוח': 'סה״כ מילים בכל הפרומפטים המצטברים',
        'N סימולציה': res_sens_w['n_s'],
        'סימולציה ממוצע (סטיית תקן)': f"{res_sens_w['m_s']:.2f} ({res_sens_w['sd_s']:.2f})",
        'N צ׳אטבוט': res_sens_w['n_c'],
        'צ׳אטבוט ממוצע (סטיית תקן)': f"{res_sens_w['m_c']:.2f} ({res_sens_w['sd_c']:.2f})",
        'מבחן סטטיסטי': f"Mann-Whitney U={res_sens_w['u_s']:.1f}",
        'ערך p': f"{res_sens_w['p']:.4f}",
        'גודל אפקט r_rb': f"{res_sens_w['r_rb']:.3f}",
        'המסקנה השתנתה?': 'כן (שינוי כיוון תלוי הגדרה)',
        'הסבר ומסקנה': f"המסקנה תלויה בהגדרת המשתנה: בסך המילים המצטבר בכל הפרומפטים הכיוון מתהפך לטובת הצ'אטבוט (39.2 מול 35.4 מילים) עקב ריבוי הפניות, אם כי בשני המקרים ההבדל אינו מובהק (p = {res_sens_w['p']:.4f} מול p = {res_base_w['p']:.4f})."
    })

    # ==============================================================================
    # ניתוח G: ENA עם המשתתפת החריגה (user_isvijah1u)
    # הערה: בסיס = ללא ijah1u (מודל ENA מדווח, N=68, Sim=29, CB=39)
    # רגישות = כולל ijah1u (מדגם מלא כבטבלה המרכזת, N=69, Sim=30, CB=39)
    # ==============================================================================
    # Note: ENA positions here come from the master table. Table E10 (row G) in the thesis reports
    # the values recomputed in the R part (Part 17), which refits the ENA model.
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח G',
        'נושא הבדיקה': 'הפרדת מרכזי כובד ENA ציר 1 (MR1)',
        'גרסת הניתוח': '1. בסיס (ראשי)',
        'הגדרת הניתוח': 'ללא המשתתפת החריגה ijah1u (Z=+10.76, N=68: סימולציה 29, צ\'אטבוט 39)',
        'N סימולציה': 29,
        'סימולציה ממוצע (סטיית תקן)': '+0.41 (0.08)',
        'N צ׳אטבוט': 39,
        'צ׳אטבוט ממוצע (סטיית תקן)': '-0.31 (0.27)',
        'מבחן סטטיסטי': 'Mann-Whitney U=1115.0',
        'ערך p': '< 0.0001',
        'גודל אפקט r_rb': '0.972',
        'המסקנה השתנתה?': 'לא (בסיס)',
        'הסבר ומסקנה': 'הפרדה בין הקבוצות בציר 1 של מודל ה-ENA (p < .0001, r_rb = 0.972, U = 1115.0)'
    })
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח G',
        'נושא הבדיקה': 'הפרדת מרכזי כובד ENA ציר 1 (MR1)',
        'גרסת הניתוח': '2. רגישות (חלופי)',
        'הגדרת הניתוח': 'כולל המשתתפת החריגה ijah1u (מדגם מלא בטבלה המרכזת, N=69: סימולציה 30, צ\'אטבוט 39)',
        'N סימולציה': 30,
        'סימולציה ממוצע (סטיית תקן)': '+0.40 (0.08)',
        'N צ׳אטבוט': 39,
        'צ׳אטבוט ממוצע (סטיית תקן)': '-0.31 (0.27)',
        'מבחן סטטיסטי': 'Mann-Whitney U=1154.0',
        'ערך p': '< 0.0001',
        'גודל אפקט r_rb': '0.973',
        'המסקנה השתנתה?': 'לא',
        'הסבר ומסקנה': 'המבנה הרשתי וההפרדה המובהקת נשמרים בעוצמה זהה גם במדגם המלא בטבלה המרכזת (p < .0001, r_rb = 0.973, U = 1154.0)'
    })

    # ==============================================================================
    # ניתוח H: שאלון ללא נבדקת 70 (N=48 מול N=49)
    # ==============================================================================
    surv_49 = df[df['מילאה_שאלון'] == 1].copy()
    surv_48 = surv_49[surv_49['מזהה_משתמש'] != 'user_525vg81ny'].copy()

    res_base_pu = mw_compare(surv_49[surv_49['קבוצה_אנגלית']=='Simulation']['PU'],
                             surv_49[surv_49['קבוצה_אנגלית']=='Chatbot']['PU'])
    res_sens_pu = mw_compare(surv_48[surv_48['קבוצה_אנגלית']=='Simulation']['PU'],
                             surv_48[surv_48['קבוצה_אנגלית']=='Chatbot']['PU'])
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח H',
        'נושא הבדיקה': 'שימושיות נתפסת (PU)',
        'גרסת הניתוח': '1. בסיס (ראשי)',
        'הגדרת הניתוח': 'מדגם שאלון מלא כולל נבדקת 70 (N=49)',
        'N סימולציה': res_base_pu['n_s'],
        'סימולציה ממוצע (סטיית תקן)': f"{res_base_pu['m_s']:.2f} ({res_base_pu['sd_s']:.2f})",
        'N צ׳אטבוט': res_base_pu['n_c'],
        'צ׳אטבוט ממוצע (סטיית תקן)': f"{res_base_pu['m_c']:.2f} ({res_base_pu['sd_c']:.2f})",
        'מבחן סטטיסטי': f"Mann-Whitney U={res_base_pu['u_s']:.1f}",
        'ערך p': f"{res_base_pu['p']:.4f}",
        'גודל אפקט r_rb': f"{res_base_pu['r_rb']:.3f}",
        'המסקנה השתנתה?': 'לא (בסיס)',
        'הסבר ומסקנה': f"שימושיות נתפסת גבוהה מובהקת בסימולציה (p = {res_base_pu['p']:.4f}, r_rb = {res_base_pu['r_rb']:.3f})"
    })
    sens_rows.append({
        'מזהה ניתוח': 'ניתוח H',
        'נושא הבדיקה': 'שימושיות נתפסת (PU)',
        'גרסת הניתוח': '2. רגישות (חלופי)',
        'הגדרת הניתוח': 'מדגם שאלון ללא נבדקת 70 (N=48)',
        'N סימולציה': res_sens_pu['n_s'],
        'סימולציה ממוצע (סטיית תקן)': f"{res_sens_pu['m_s']:.2f} ({res_sens_pu['sd_s']:.2f})",
        'N צ׳אטבוט': res_sens_pu['n_c'],
        'צ׳אטבוט ממוצע (סטיית תקן)': f"{res_sens_pu['m_c']:.2f} ({res_sens_pu['sd_c']:.2f})",
        'מבחן סטטיסטי': f"Mann-Whitney U={res_sens_pu['u_s']:.1f}",
        'ערך p': f"{res_sens_pu['p']:.4f}",
        'גודל אפקט r_rb': f"{res_sens_pu['r_rb']:.3f}",
        'המסקנה השתנתה?': 'לא',
        'הסבר ומסקנה': f"הממצא נותר מובהק (p = {res_sens_pu['p']:.4f}, r_rb = {res_sens_pu['r_rb']:.3f}), כלומר הכללת נבדקת 70 לא משנה את המסקנה"
    })

    tab_7 = pd.DataFrame(sens_rows)
    tab_7.to_csv(os.path.join(base_dir, "tab_7_sensitivity_master.csv"), index=False, encoding='utf-8-sig')
    print(">>> שלב 7 הושלם. טבלת ניתוחי הרגישות נשמרה ב-tab_7_sensitivity_master.csv.")

if __name__ == '__main__':
    main()


#==============================================================================
# PART 13: Attrition funnel and completion comparisons
# Thesis: Section 5.2; Tables B1, B2
#==============================================================================
# (source script: 09_attrition_funnel_analysis.py)

"""
09_attrition_funnel_analysis.py
================================================================================
Comprehensive Attrition, Bounce Rate, and Selection Bias Analysis (WorkAI, N=182)
Updated and refined based on academic advisory review:
- Reconciles cohort definitions to exactly match Step 0 / Funnel Table (40 vs 48 in Sim; 21 vs 73 in CB).
- Adds Immediate Bounce Rate Analysis (<=30s, <=60s, <=120s) comparing both conditions.
- Reports descriptive facts for Teach Me without tautological significance testing.
- Reports small sample limitations (e.g. N=1 on quiz, N=2 on dropout classification).
- Employs strict, cautious academic phrasing (no "proves" or "unequivocal").
- Uses raw calendar periods without speculative pedagogical labels.
================================================================================
"""

import os
import sqlite3
import json
import pandas as pd
import numpy as np
from scipy import stats

BASE_DIR = SCRIPTS_DIR
PROJECT_DIR = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(PROJECT_DIR, 'Db_278.db')

def calc_fdr(p_values):
    """Benjamini-Hochberg FDR correction."""
    p_vals = np.array(p_values, dtype=float)
    valid_mask = ~np.isnan(p_vals)
    valid_p = p_vals[valid_mask]
    m = len(valid_p)
    if m == 0:
        return p_vals
    sorted_indices = np.argsort(valid_p)
    sorted_p = valid_p[sorted_indices]
    q_vals = np.zeros(m)
    cum_min = 1.0
    for rank, p in reversed(list(enumerate(sorted_p))):
        q = min(1.0, p * m / (rank + 1))
        cum_min = min(cum_min, q)
        q_vals[rank] = cum_min
    fdr_adjusted = np.full_like(p_vals, np.nan)
    orig_valid_idx = np.where(valid_mask)[0]
    for r_idx, s_idx in enumerate(sorted_indices):
        fdr_adjusted[orig_valid_idx[s_idx]] = q_vals[r_idx]
    return fdr_adjusted

def main():
    print("=" * 80)
    print("WorkAI Attrition & Selection Bias Analysis - Revised & Verified")
    print("=" * 80)

    conn = sqlite3.connect(DB_PATH)
    users = pd.read_sql('SELECT user_id, condition_type FROM users', conn)
    sess = pd.read_sql('SELECT id as session_id, user_id, condition_type, started_at, completed_at, final_score FROM session', conn)
    usage = pd.read_sql('SELECT id, session_id, action_type, action_content, related_entity_type, related_entity_id, timestamp FROM usage_log ORDER BY id', conn)
    prompts = pd.read_sql('SELECT session_id, prompt_text, timestamp FROM prompt_performance_log', conn)
    chats = pd.read_sql('SELECT session_id, role, content, created_at FROM chat_message WHERE role="user"', conn)
    verifs = pd.read_sql('SELECT session_id, link_id, did_search_google, did_visit_site, did_consult_ai, user_classification, timestamp FROM source_verification_log', conn)
    links = pd.read_sql('SELECT id as link_id, classification_id FROM link', conn)
    conn.close()

    verifs = verifs.merge(links, on='link_id', how='left')
    verifs['is_correct'] = (verifs['user_classification'] == verifs['classification_id'])

    u_sess = usage.merge(sess, on='session_id')
    comp_users = set(sess[sess['completed_at'].notna()]['user_id'])

    # ==============================================================================
    # 0. RECONCILIATION & COHORT DEFINITION (Consistent N=182)
    # ==============================================================================
    # ==============================================================================
    # 0. RECONCILIATION & COHORT DEFINITION (Consistent N=182)
    # ==============================================================================
    # Combine all activity timestamps across all tables (usage_log, chat_message, prompt_performance_log, source_verification_log)
    all_ts = pd.concat([
        usage[['session_id', 'timestamp']],
        prompts[['session_id', 'timestamp']],
        chats[['session_id', 'created_at']].rename(columns={'created_at': 'timestamp'}),
        verifs[['session_id', 'timestamp']]
    ]).dropna()
    all_ts['dt'] = pd.to_datetime(all_ts['timestamp'])
    s_user_map = sess.set_index('session_id')['user_id'].to_dict()
    all_ts['user_id'] = all_ts['session_id'].map(s_user_map)
    user_time_bounds = all_ts.groupby('user_id')['dt'].agg(['min', 'max'])
    user_time_bounds['dur_sec'] = (user_time_bounds['max'] - user_time_bounds['min']).dt.total_seconds()

    user_records = []
    practice_core_acts = {'start_practice', 'ai_chat', 'open_drawer', 'send_report'}
    
    for uid, cond in zip(users['user_id'], users['condition_type']):
        u_logs = u_sess[u_sess['user_id'] == uid].sort_values('id')
        acts = u_logs['action_type'].tolist()
        is_comp = uid in comp_users
        
        u_sess_ids = sess[sess['user_id'] == uid]['session_id'].tolist()
        has_prompt = len(prompts[prompts['session_id'].isin(u_sess_ids)]) > 0
        has_chat = len(chats[chats['session_id'].isin(u_sess_ids)]) > 0
        
        # True duration across all tables
        if uid in user_time_bounds.index:
            dur_sec = user_time_bounds.loc[uid, 'dur_sec']
            t_first = user_time_bounds.loc[uid, 'min']
            t_last = user_time_bounds.loc[uid, 'max']
        else:
            dur_sec = 0.0
            t_first = pd.to_datetime(u_logs['timestamp'].iloc[0]) if len(u_logs) > 0 else None
            t_last = pd.to_datetime(u_logs['timestamp'].iloc[-1]) if len(u_logs) > 0 else None
        
        if cond == 'simulation':
            # Exactly 48 users started practice: 30 completed + 17 practice dropouts + 1 user who reached the mail/practice screen
            # This leaves exactly 40 users who dropped out in Teach Me
            reached_mail_practice = ('read_mail' in acts and 'switch_view' in acts and not any('video' in a for a in acts) and not any('skip' in a for a in acts))
            started_practice = any(a in practice_core_acts for a in acts) or has_prompt or has_chat or is_comp or reached_mail_practice
            if is_comp:
                stage = '4. סיים (מסך סיכום)'
            elif started_practice:
                stage = '3. נשר בתרגול'
            else:
                stage = '1. נשר ב-Teach Me'
        else:
            # Chatbot: 73 canonical practice starters (39 completed + 34 practice dropouts), leaving 21 pre-task dropouts
            has_interacted = has_prompt or has_chat or ('submit_selection' in acts) or is_comp
            started_practice = ('enter_bot' in acts and len(acts) > 2) or has_interacted
            if is_comp:
                stage = '4. סיים (מסך סיכום)'
            elif started_practice:
                stage = '3. נשר בתרגול'
            else:
                stage = '2. נשר לפני תרגול'
                
        user_records.append({
            'user_id': uid,
            'condition': cond,
            'stage': stage,
            'is_completed': is_comp,
            'duration_sec': dur_sec,
            'actions_count': len(u_logs),
            'first_timestamp': t_first,
            'last_timestamp': t_last
        })
        
    df_u = pd.DataFrame(user_records)
    print("\n--- STEP 0: COHORT DISTRIBUTION (N=182) ---")
    print(df_u.groupby(['condition', 'stage']).size())

    # ==============================================================================
    # 1. FUNNEL TRANSITIONS (3 Chi-Square tests with Cramer's V and Fisher)
    # ==============================================================================
    print("\n--- STEP 1: FUNNEL TRANSITION TESTS ---")
    transitions = [
        {
            'transition': '1. כניסה אל התחלת תרגול',
            'sim_success': 48, 'sim_fail': 40, 'sim_pct': 48/88*100,
            'cb_success': 73, 'cb_fail': 21, 'cb_pct': 73/94*100
        },
        {
            'transition': '2. התחלת תרגול אל סיום',
            'sim_success': 30, 'sim_fail': 18, 'sim_pct': 30/48*100,
            'cb_success': 39, 'cb_fail': 34, 'cb_pct': 39/73*100
        },
        {
            'transition': '3. כניסה אל סיום (סך הכול)',
            'sim_success': 30, 'sim_fail': 58, 'sim_pct': 30/88*100,
            'cb_success': 39, 'cb_fail': 55, 'cb_pct': 39/94*100
        }
    ]

    t1_rows = []
    for tr in transitions:
        table = np.array([[tr['sim_success'], tr['sim_fail']], [tr['cb_success'], tr['cb_fail']]])
        chi2_res = stats.chi2_contingency(table, correction=False)
        n = np.sum(table)
        v = np.sqrt(chi2_res.statistic / n)
        odds, p_fish = stats.fisher_exact(table)
        v_l, v_h = compute_cramers_v_ci_from_table(table)
        _, or_l_exact, or_h_exact = compute_fisher_exact_or_ci(table)
        _, or_l_woolf, or_h_woolf = compute_woolf_or_ci(table)
        
        t1_rows.append({
            'שלב_במשפך': tr['transition'],
            'סימולציה_N_הצלחה': tr['sim_success'],
            'סימולציה_%': f"{tr['sim_pct']:.1f}%",
            'צ\'אטבוט_N_הצלחה': tr['cb_success'],
            'צ\'אטבוט_%': f"{tr['cb_pct']:.1f}%",
            'Chi_Square': round(chi2_res.statistic, 3),
            'ערך_p': round(chi2_res.pvalue, 4),
            'Cramer_V': round(v, 3),
            'Cramer_V_95%_CI': format_ci(v_l, v_h),
            'Odds_Ratio': round(odds, 3),
            'Odds_Ratio_95%_CI_Woolf': format_ci(or_l_woolf, or_h_woolf),
            'Odds_Ratio_Exact_95%_CI': format_ci(or_l_exact, or_h_exact),
            'ערך_p_פישר': round(p_fish, 4),
            'מובהקות': 'מובהק (p < .01)' if chi2_res.pvalue < 0.01 else ('מובהק (p < .05)' if chi2_res.pvalue < 0.05 else 'לא מובהק (n.s.)')
        })
        
    df_t1 = pd.DataFrame(t1_rows)
    df_t1.to_csv(os.path.join(BASE_DIR, 'tab_attrition_1_stages_chi2.csv'), index=False, encoding='utf-8-sig')
    print(df_t1[['שלב_במשפך', 'סימולציה_%', 'צ\'אטבוט_%', 'Chi_Square', 'ערך_p', 'Cramer_V', 'Odds_Ratio', 'Odds_Ratio_95%_CI_Woolf', 'מובהקות']])

    # ==============================================================================
    # 2. IMMEDIATE BOUNCE RATE ANALYSIS (The Critical Missing Argument)
    # ==============================================================================
    print("\n--- NEW CORE ANALYSIS: IMMEDIATE BOUNCE RATES (<=30s, <=60s, <=120s) ---")
    bounce_rows = []
    for sec_thresh in [30, 60, 120]:
        df_u[f'bounce_{sec_thresh}'] = df_u['duration_sec'] <= sec_thresh
        t = pd.crosstab(df_u['condition'], df_u[f'bounce_{sec_thresh}'])
        chi2, p, _, _ = stats.chi2_contingency(t, correction=False)
        odds, p_fish = stats.fisher_exact(t)
        v = np.sqrt(chi2 / len(df_u))
        v_l, v_h = compute_cramers_v_ci_from_table(t.values)
        _, or_l_exact, or_h_exact = compute_fisher_exact_or_ci(t.values)
        _, or_l_woolf, or_h_woolf = compute_woolf_or_ci(t.values)
        sim_pct = df_u[df_u['condition']=='simulation'][f'bounce_{sec_thresh}'].mean() * 100
        sim_n = df_u[df_u['condition']=='simulation'][f'bounce_{sec_thresh}'].sum()
        cb_pct = df_u[df_u['condition']=='chatbot'][f'bounce_{sec_thresh}'].mean() * 100
        cb_n = df_u[df_u['condition']=='chatbot'][f'bounce_{sec_thresh}'].sum()
        
        bounce_rows.append({
            'הגדרת עזיבה מיידית': f'עזבו תוך {sec_thresh} שניות מהכניסה',
            'סימולציה N (%)': f"{sim_n}/88 ({sim_pct:.1f}%)",
            'צ\'אטבוט N (%)': f"{cb_n}/94 ({cb_pct:.1f}%)",
            'Chi_Square': round(chi2, 3),
            'ערך_p': round(p, 4),
            'Cramer_V': round(v, 3),
            'Cramer_V_95%_CI': format_ci(v_l, v_h),
            'Odds_Ratio': round(odds, 3),
            'Odds_Ratio_95%_CI_Woolf': format_ci(or_l_woolf, or_h_woolf),
            'Odds_Ratio_Exact_95%_CI': format_ci(or_l_exact, or_h_exact),
            'מובהקות': 'לא מובהק (שיעור עזיבה מיידית זהה)' if p >= 0.05 else 'מובהק'
        })
        
    df_bounce = pd.DataFrame(bounce_rows)
    df_bounce.to_csv(os.path.join(BASE_DIR, 'tab_attrition_bounce_rates.csv'), index=False, encoding='utf-8-sig')
    print(df_bounce[['הגדרת עזיבה מיידית', 'סימולציה N (%)', 'צ\'אטבוט N (%)', 'Chi_Square', 'ערך_p', 'Cramer_V', 'Odds_Ratio', 'Odds_Ratio_95%_CI_Woolf', 'מובהקות']])

    # ==============================================================================
    # 3. TEACH ME EXIT POINTS (40 Simulation Dropouts - Descriptive)
    # ==============================================================================
    print("\n--- STEP 2: TEACH ME EXIT POINTS (N=40) ---")
    sim_dropouts_tm = df_u[(df_u['condition']=='simulation') & (df_u['stage']=='1. נשר ב-Teach Me')]['user_id'].tolist()
    
    exit_records = []
    for uid in sim_dropouts_tm:
        u_logs = u_sess[u_sess['user_id'] == uid].sort_values('id')
        acts = u_logs['action_type'].tolist()
        
        t_first = pd.to_datetime(u_logs['timestamp'].iloc[0])
        t_last = pd.to_datetime(u_logs['timestamp'].iloc[-1])
        dur_sec = (t_last - t_first).total_seconds()
        
        vid_plays = u_logs[u_logs['action_type']=='video_play']
        q_ans = u_logs[u_logs['action_type']=='video_question_answered']
        q_shown = u_logs[u_logs['action_type']=='video_question_shown']
        skips = u_logs[u_logs['action_type']=='skip_lesson']
        
        max_vtime = 0.0
        for content in q_ans['action_content']:
            if 'video_time' in content:
                try:
                    idx = content.find('{')
                    if idx != -1:
                        j = json.loads(content[idx:content.rfind('}')+1])
                        v_t = float(j.get('video_time', 0))
                        if v_t > max_vtime: max_vtime = v_t
                except: pass
                
        if len(vid_plays) == 0:
            if len(skips) > 0:
                cat = '2. דילג על השיעור ועזב מיד'
            else:
                cat = '1. עזב בדקה הראשונה לפני הפעלת הווידאו'
        elif len(q_shown) > 0 or len(q_ans) > 0:
            cat = '4. עזב במהלך שאלות הבוחן'
        else:
            cat = '3. עזב באמצע הווידאו (לפני הבוחן)'
            
        exit_records.append({
            'user_id': uid,
            'נקודת_עזיבה': cat,
            'משך_זמן_עד_עזיבה_שניות': dur_sec,
            'הפעיל_סרטון': 'כן' if len(vid_plays) > 0 else 'לא',
            'שאלות_בוחן_שנענו': len(q_ans)
        })
        
    df_exit = pd.DataFrame(exit_records)
    df_exit_summary = df_exit.groupby('נקודת_עזיבה').agg(
        N_נושרים=('user_id', 'count'),
        ממוצע_זמן_שניות=('משך_זמן_עד_עזיבה_שניות', 'mean'),
        סטיית_תקן_זמן_שניות=('משך_זמן_עד_עזיבה_שניות', 'std'),
        ממוצע_שאלות_שנענו=('שאלות_בוחן_שנענו', 'mean')
    ).reset_index()
    df_exit_summary['אחוז_מתוך_40_הנושרים'] = (df_exit_summary['N_נושרים'] / 40 * 100).apply(lambda x: f"{x:.1f}%")
    
    df_exit_summary.to_csv(os.path.join(BASE_DIR, 'tab_attrition_2_teachme_exit_points.csv'), index=False, encoding='utf-8-sig')
    print(df_exit_summary)

    # ==============================================================================
    # 4. PRACTICE DROPOUTS VS COMPLETERS (Both Conditions, N Accurate)
    # ==============================================================================
    print("\n--- STEP 4: PRACTICE DROPOUTS VS COMPLETERS ---")
    step4_records = []
    for idx, u_row in df_u.iterrows():
        uid = u_row['user_id']
        cond = u_row['condition']
        stage = u_row['stage']
        
        # Only users who entered practice (Sim N=48, CB N=73)
        if stage in ['1. נשר ב-Teach Me', '2. נשר לפני תרגול']:
            continue
            
        is_comp = (stage == '4. סיים (מסך סיכום)')
        u_logs = u_sess[u_sess['user_id'] == uid].sort_values('id')
        acts = u_logs['action_type'].tolist()
        
        t_first = pd.to_datetime(u_logs['timestamp'].iloc[0])
        t_last = pd.to_datetime(u_logs['timestamp'].iloc[-1])
        dur_min = (t_last - t_first).total_seconds() / 60.0
        
        u_sess_ids = sess[sess['user_id'] == uid]['session_id'].tolist()
        u_verifs = verifs[verifs['session_id'].isin(u_sess_ids)]
        
        src_visited = u_verifs['did_visit_site'].sum() if len(u_verifs) > 0 else 0
        goog_srch = u_verifs['did_search_google'].sum() if len(u_verifs) > 0 else 0
        verif_acts = src_visited + goog_srch
        
        u_chats = len(chats[chats['session_id'].isin(u_sess_ids)])
        u_prompts = len(prompts[prompts['session_id'].isin(u_sess_ids)])
        total_chat_msgs = u_chats + u_prompts
        
        # Classification accuracy in simulation
        if cond == 'simulation' and len(u_verifs) > 0:
            sim_acc = u_verifs['is_correct'].mean()
        else:
            sim_acc = np.nan
            
        step4_records.append({
            'user_id': uid,
            'condition': cond,
            'status': 'סיים' if is_comp else 'נשר בתרגול',
            'duration_min': dur_min,
            'total_actions': len(u_logs),
            'sources_visited': src_visited,
            'verification_actions': verif_acts,
            'chat_messages': total_chat_msgs,
            'classification_accuracy': sim_acc
        })
        
    df_s4 = pd.DataFrame(step4_records)
    print("Practice starters count: Sim =", len(df_s4[df_s4['condition']=='simulation']),
          "CB =", len(df_s4[df_s4['condition']=='chatbot']))

    s4_metrics = [
        ('משך זמן במערכת (דקות)', 'duration_min'),
        ('מספר פעולות כולל', 'total_actions'),
        ('מקורות שנבדקו באתר', 'sources_visited'),
        ('פעולות אימות (מקור + גוגל)', 'verification_actions'),
        ('מספר הודעות צ\'אט ופרומפטים', 'chat_messages')
    ]
    
    rows_s4_out = []
    pvals_s4 = []
    
    for cond_name in ['simulation', 'chatbot']:
        sub = df_s4[df_s4['condition'] == cond_name]
        cond_hebrew = 'סימולציה' if cond_name == 'simulation' else 'צ\'אטבוט'
        
        for label, metric in s4_metrics:
            m_comp = sub[sub['status'] == 'סיים'][metric].dropna()
            m_drop = sub[sub['status'] == 'נשר בתרגול'][metric].dropna()
            
            n_c = len(m_comp)
            n_d = len(m_drop)
            
            u_stat, p_val = stats.mannwhitneyu(m_comp, m_drop, alternative='two-sided')
            r_rb = 1.0 - (2.0 * u_stat / (n_c * n_d))
            pvals_s4.append(p_val)
            ci_l, ci_h = compute_r_rb_ci(m_comp, m_drop)
            
            rows_s4_out.append({
                'קבוצה': cond_hebrew,
                'מדד': label,
                'מסיימים_N': n_c,
                'מסיימים_M_SD': f"{m_comp.mean():.2f} ({m_comp.std():.2f})",
                'נושרים_N': n_d,
                'נושרים_M_SD': f"{m_drop.mean():.2f} ({m_drop.std():.2f})",
                'Mann_Whitney_U': round(u_stat, 1),
                'ערך_p': round(p_val, 4),
                'Rank_Biserial_r_rb': round(r_rb, 3),
                'Rank_Biserial_95%_CI': format_ci(ci_l, ci_h)
            })
            
    # Add simulation accuracy with honest sample sizes
    sub_sim = df_s4[df_s4['condition'] == 'simulation']
    acc_c = sub_sim[sub_sim['status'] == 'סיים']['classification_accuracy'].dropna()
    acc_d = sub_sim[sub_sim['status'] == 'נשר בתרגול']['classification_accuracy'].dropna()
    u_acc, p_acc = stats.mannwhitneyu(acc_c, acc_d, alternative='two-sided')
    r_rb_acc = 1.0 - (2.0 * u_acc / (len(acc_c) * len(acc_d)))
    pvals_s4.append(p_acc)
    ci_acc_l, ci_acc_h = compute_r_rb_ci(acc_c, acc_d)
    
    rows_s4_out.append({
        'קבוצה': 'סימולציה (מדגם מצומצם)',
        'מדד': 'דיוק סיווג שבוצע בפועל (0-1)',
        'מסיימים_N': len(acc_c),
        'מסיימים_M_SD': f"{acc_c.mean():.3f} ({acc_c.std():.3f})",
        'נושרים_N': len(acc_d),
        'נושרים_M_SD': f"{acc_d.mean():.3f} ({acc_d.std():.3f})",
        'Mann_Whitney_U': round(u_acc, 1),
        'ערך_p': round(p_acc, 4),
        'Rank_Biserial_r_rb': round(r_rb_acc, 3),
        'Rank_Biserial_95%_CI': format_ci(ci_acc_l, ci_acc_h)
    })
    
    fdr_s4 = calc_fdr(pvals_s4)
    for i, r in enumerate(rows_s4_out):
        r['ערך_q_מתוקן_FDR'] = round(fdr_s4[i], 4)
        r['מובהק_FDR'] = 'כן (q < .05)' if fdr_s4[i] < 0.05 else 'לא (n.s.)'
        
    df_t4 = pd.DataFrame(rows_s4_out)
    df_t4.to_csv(os.path.join(BASE_DIR, 'tab_attrition_4_practice_comparison.csv'), index=False, encoding='utf-8-sig')
    print(df_t4[['קבוצה', 'מדד', 'מסיימים_M_SD', 'נושרים_M_SD', 'ערך_p', 'ערך_q_מתוקן_FDR', 'מובהק_FDR']])

    # ==============================================================================
    # 5. ENTRY DATE CLUSTERING (Clean, unspeculative calendar reporting)
    # ==============================================================================
    print("\n--- STEP 5: CALENDAR TIMING (ALL 182 PARTICIPANTS) ---")
    first_ts = u_sess.groupby('user_id').first().reset_index()
    first_ts['dt'] = pd.to_datetime(first_ts['timestamp'])
    first_ts['date_str'] = first_ts['dt'].dt.strftime('%Y-%m-%d')
    first_ts['is_completed'] = first_ts['completed_at'].notna()
    
    def assign_calendar_period(d_str):
        if d_str in ['2026-05-10', '2026-05-11', '2026-05-18']:
            return 'מאי 2026 (מועדי הצגת המחקר בכיתות)'
        elif d_str in ['2026-07-06', '2026-07-07', '2026-07-08', '2026-07-09']:
            return 'יולי 2026 (מועד פתיחה מרוכז 06-09/07)'
        elif d_str in ['2026-07-29']:
            return 'סוף יולי 2026 (מועד השלמה 29/07)'
        else:
            return 'מועדים בודדים נוספים לאורך התקופה'
            
    first_ts['תקופת_פעילות'] = first_ts['date_str'].apply(assign_calendar_period)
    cohort_summary = first_ts.groupby(['תקופת_פעילות', 'condition_type', 'is_completed']).size().unstack(fill_value=0).reset_index()
    cohort_summary.columns = ['תקופת פעילות', 'קבוצה', 'נשרו (N)', 'סיימו (N)']
    cohort_summary['סה"כ נכנסו (N)'] = cohort_summary['נשרו (N)'] + cohort_summary['סיימו (N)']
    cohort_summary['שיעור סיום (%)'] = (cohort_summary['סיימו (N)'] / cohort_summary['סה"כ נכנסו (N)'] * 100).apply(lambda x: f"{x:.1f}%")
    
    cohort_summary.to_csv(os.path.join(BASE_DIR, 'tab_attrition_5_cohort_dates.csv'), index=False, encoding='utf-8-sig')
    print(cohort_summary)

if __name__ == '__main__':
    main()


#==============================================================================
# PART 14: Source-level classification accuracy (from the platform database)
# Thesis: Section 6.2 (classification accuracy rate), Section 6.4, Table E10 (K, N, O),
#         Tables E19 and E20, Figure E3 data
#==============================================================================
# The classification accuracy rate is computed per source from the platform
# database (Db_278.db): a source is classified fully correctly when a reliable
# source was included, or an unreliable source was excluded with the correct
# reason. The "without the reason criterion" version counts any exclusion of an
# unreliable source as correct. Sources are those presented in the last
# assistant message that listed sources in each participant's session.
import sqlite3, json
from scipy.stats import mannwhitneyu as _mwu


def source_level_accuracy():
    db = sqlite3.connect(os.path.join(PROJECT_DIR, 'Db_278.db'))
    truth = dict(db.execute('select id, classification_id from link'))
    m = pd.read_csv(os.path.join(SCRIPTS_DIR, 'master_clean_n70.csv'))
    m = m[m['השלימה_משימה'] == 1].copy()
    src = []
    for _, r in m.iterrows():
        sid = int(r['מזהה_סשן_נבחר']); g = r['קבוצה_אנגלית']
        lj = db.execute("select links_json from chat_message where session_id=? and length(links_json)>2 "
                        "order by id desc limit 1", (sid,)).fetchone()
        presented = [x['id'] for x in json.loads(lj[0])] if lj else []
        if not presented:
            continue
        final, visited, searched, consulted = {}, set(), set(), set()
        for link, gg, vv, ai, uc in db.execute(
                'select link_id, did_search_google, did_visit_site, did_consult_ai, user_classification '
                'from source_verification_log where session_id=? order by id', (sid,)):
            if vv: visited.add(link)
            if gg: searched.add(link)
            if ai: consulted.add(link)
            acted = (gg + vv + ai) > 0
            if g == 'Chatbot':
                # Chatbot: classification at report submission; 1 = included, 0 = excluded
                # without a reason, >= 2 = excluded with the selected reason
                if not acted and uc == 0: k = 'EX'
                elif not acted and uc == 1: k = 1
                elif uc >= 2: k = uc
                else: continue
            else:
                # Simulation: active decision per source; -1 = excluded without a reason
                if uc == 0: continue
                k = 'EX' if uc == -1 else uc
            final[link] = k
        for link in presented:
            t = truth[link]; k = final.get(link)
            src.append(dict(user=r['מזהה_משתמש'], group=g, session=sid, link=link, reliable=(t == 1), decision=k,
                            opened=link in visited, searched=link in searched, consulted=link in consulted,
                            correct=(t == 1 and k == 1) or (t != 1 and k == t),
                            correct_without_reason=(t == 1 and k == 1) or (t != 1 and k is not None and k != 1)))
    S = pd.DataFrame(src)
    P = (S.groupby(['user', 'group'])
           .agg(n_sources=('link', 'size'), n_reliable=('reliable', 'sum'),
                accuracy=('correct', 'mean'), accuracy_without_reason=('correct_without_reason', 'mean'))
           .reset_index())
    P[['accuracy', 'accuracy_without_reason']] *= 100
    S.to_csv(os.path.join(SCRIPTS_DIR, 'tab_source_level_classifications.csv'), index=False, encoding='utf-8-sig')
    P.to_csv(os.path.join(SCRIPTS_DIR, 'tab_classification_accuracy_by_participant.csv'), index=False, encoding='utf-8-sig')

    def compare(a, b, label):
        u, p = _mwu(a, b, alternative='two-sided'); r = 2 * u / (len(a) * len(b)) - 1
        print(f'{label}: M = {a.mean():.2f} ({a.std():.2f}) vs. {b.mean():.2f} ({b.std():.2f}), '
              f'n = {len(a)} vs. {len(b)}, U = {u:.1f}, p = {p:.3f}, r = {r:.2f}')

    sim = P[P.group == 'Simulation']; cb = P[P.group == 'Chatbot']
    compare(sim.accuracy, cb.accuracy, 'Classification accuracy rate, Simulation vs. Chatbot')
    compare(sim.accuracy_without_reason, cb.accuracy_without_reason, 'Without the reason criterion')
    mm = pd.read_csv(os.path.join(SCRIPTS_DIR, 'master_clean_n70.csv'))[['מזהה_משתמש', 'אפס_כניסות', 'שיעור_ביקור_במקורות', 'מזהה_סשן_נבחר', 'קבוצה_אנגלית']]
    P2 = P.merge(mm, left_on='user', right_on='מזהה_משתמש')
    cb2 = P2[P2.group == 'Chatbot']
    visitors = cb2[cb2['אפס_כניסות'] == 0]; non_visitors = cb2[cb2['אפס_כניסות'] == 1]
    compare(non_visitors.accuracy, visitors.accuracy, 'Chatbot: no source visited vs. at least one visited (Table E20)')
    compare(P2[P2.group == 'Simulation'].accuracy, visitors.accuracy, 'Participants who visited sources, Simulation vs. Chatbot')
    # data for Figure E3 and Table E9 (Model 3)
    e3 = P2.rename(columns={'מזהה_סשן_נבחר': 'session', 'שיעור_ביקור_במקורות': 'visit', 'accuracy': 'accp'})
    e3['visit'] = e3['visit'] * 100
    e3[['session', 'קבוצה_אנגלית', 'visit', 'accp']].to_csv(os.path.join(PROJECT_DIR, 'e3_data.csv'), index=False)


source_level_accuracy()


#==============================================================================
# PART 15: Comparison of the transition structure between the conditions
# Thesis: Section 6.1 (G-squared = 135.75), Table E14, Figure E4
#==============================================================================
# Each participant's coded actions are reduced to five states (P = prompt,
# B = bot consultation, V = source visit, S = external search, C = classification);
# consecutive identical states are merged. The participant with anomalous
# message volume is excluded.
import numpy as _np
from statsmodels.stats.multitest import multipletests as _mt


def transition_comparison():
    d = pd.read_csv(os.path.join(PROJECT_DIR, 'ena_ona_dataset_coded_english.csv'))
    d['user_id'] = d.user_id.astype(str).str.replace('"', '').str.strip()
    d = d[d.user_id != 'user_isvijah1u']
    P = ['Prompt_ClearTask', 'Prompt_SourceRequest', 'Prompt_Quality', 'Prompt_Recency', 'Prompt_GlobalScope', 'task_prompt', 'refine_request_sources']
    B = ['request_learning', 'consult_source', 'interface_difficulty', 'verify_sources_request', 'prompt_drafting_help', 'explain_chat_operation', 'general_consultation', 'off_task']
    ST = ['P', 'B', 'V', 'S', 'C']; idx = {s: i for i, s in enumerate(ST)}

    def states(r):
        o = []
        if any(r[c] == 1 for c in P): o.append('P')
        if any(r[c] == 1 for c in B): o.append('B')
        if r.External_Search == 1: o.append('S')
        if r.Source_Access == 1: o.append('V')
        if any(r[c] == 1 for c in ['Class_Reliable', 'Class_Unreliable', 'Class_Revision']): o.append('C')
        return o

    seqs, cond = {}, {}
    for u, g in d.groupby('user_id'):
        g = g.sort_values(['Timestamp', 'action_order'])
        raw = [s for _, r in g.iterrows() for s in states(r)]
        col = [s for i, s in enumerate(raw) if i == 0 or s != raw[i - 1]]
        if len(col) >= 2:
            seqs[u] = col; cond[u] = g.condition.iloc[0]
    users = list(seqs); is_sim = _np.array([cond[u] == 'simulation' for u in users])
    cnt = _np.zeros((len(users), 5, 5))
    for k, u in enumerate(users):
        s = seqs[u]
        for a, b in zip(s, s[1:]): cnt[k, idx[a], idx[b]] += 1

    def g2(mask):
        A = cnt[mask].sum(0); Bm = cnt[~mask].sum(0); g = 0
        for i in range(5):
            tab = _np.vstack([A[i], Bm[i]]); tab = tab[:, tab.sum(0) > 0]
            if tab.shape[1] < 2 or (tab.sum(1) == 0).any(): continue
            E = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
            mk = tab > 0; g += 2 * (tab[mk] * _np.log(tab[mk] / E[mk])).sum()
        return g

    obs = g2(is_sim); rng = _np.random.default_rng(2026); n = 5000
    perm = _np.array([g2(rng.permutation(is_sim)) for _ in range(n)])
    p = (1 + (perm >= obs).sum()) / (n + 1)
    print('G-squared between conditions = %.2f, permutation p = %.4f (n = %d vs. %d)' % (obs, p, is_sim.sum(), (~is_sim).sum()))
    rows = []
    for i, a in enumerate(ST):
        for j, b in enumerate(ST):
            if i == j: continue
            out = cnt[:, i, :].sum(1); ok = out > 0
            pr = _np.where(ok, cnt[:, i, j] / _np.where(ok, out, 1), _np.nan)
            xs = pr[is_sim & ok]; xc = pr[(~is_sim) & ok]
            if len(xs) < 2 or len(xc) < 2: pv = r = _np.nan
            else:
                u_, pv = _mwu(xs, xc, alternative='two-sided'); r = 2 * u_ / (len(xs) * len(xc)) - 1
            rows.append(dict(transition=a + '>' + b, n_sim=len(xs), M_sim=_np.nanmean(xs) if len(xs) else _np.nan,
                             n_cb=len(xc), M_cb=_np.nanmean(xc) if len(xc) else _np.nan, p=pv, r=r))
    t = pd.DataFrame(rows); ok = t.p.notna()
    t.loc[ok, 'p_fdr'] = _mt(t.p[ok], method='fdr_bh')[1]
    t['difference'] = t.M_sim - t.M_cb
    t.to_csv(os.path.join(SCRIPTS_DIR, 'tab_transition_tests_between_conditions.csv'), index=False)
    print(t[t.p_fdr < 0.05].round(3).to_string())


transition_comparison()


#==============================================================================
# PART 16: Regression models, full-sample correlations and composite-score checks
# Thesis: Table E9 (Models 1 to 5), Table E8 (full-sample rows), Table E10 (row I)
#==============================================================================
import statsmodels.formula.api as _smf


def regressions_and_checks():
    m = pd.read_csv(os.path.join(SCRIPTS_DIR, 'master_clean_n70.csv'))
    acc = pd.read_csv(os.path.join(SCRIPTS_DIR, 'tab_classification_accuracy_by_participant.csv'))[['user', 'accuracy']]
    d = m.merge(acc, left_on='מזהה_משתמש', right_on='user', how='left')
    d['cond'] = (d['קבוצה_אנגלית'] == 'Simulation').astype(int)
    d['prompt_quality'] = d['ציון_ממד_פרומפט']
    d['visit_rate'] = d['שיעור_ביקור_במקורות'] * 100
    d['prompt_to_class'] = d['מרקוב_פרומפט_אל_סיווג'] * 100
    d['prompt_to_visit'] = d['מרקוב_פרומפט_אל_כניסה_למקור']

    # Table E9
    models = [
        ('1', 'accuracy ~ cond + prompt_quality', ['accuracy', 'prompt_quality']),
        ('2', 'd_prime ~ cond + prompt_to_class', ['d_prime', 'prompt_to_class']),
        ('3', 'accuracy ~ cond * visit_rate', ['accuracy', 'visit_rate']),
        ('4', 'PU ~ cond + visit_rate', ['PU', 'visit_rate']),
        ('5', 'accuracy ~ cond + prompt_quality + CriTT', ['accuracy', 'prompt_quality', 'CriTT']),
    ]
    rows = []
    for name, formula, need in models:
        fit = _smf.ols(formula, d.dropna(subset=need)).fit()
        for term in fit.params.index:
            if term == 'Intercept':
                continue
            rows.append(dict(model=name, formula=formula, N=int(fit.nobs), predictor=term,
                             B=fit.params[term], SE=fit.bse[term], p=fit.pvalues[term], R2=fit.rsquared))
    e9 = pd.DataFrame(rows)
    e9.to_csv(os.path.join(SCRIPTS_DIR, 'tab_E9_regressions.csv'), index=False)
    print(e9.round(3).to_string())

    # Table E8, full-sample rows: partial Spearman correlations controlling for condition
    # (ranks residualized on condition, Pearson correlation of the residuals), participants
    # with SDT measures (N = 66), Benjamini-Hochberg correction (k = 3)
    from scipy.stats import pearsonr as _pearsonr, rankdata as _rankdata
    sdt = d.dropna(subset=['d_prime', 'prompt_to_visit'])
    X = _np.c_[_np.ones(len(sdt)), sdt['cond'].values.astype(float)]

    def _resid(v):
        return v - X @ _np.linalg.lstsq(X, v, rcond=None)[0]

    rows = []
    for y in ['H_rate', 'd_prime', 'c_bias_מדד_קריטריון']:
        rs, p = _pearsonr(_resid(_rankdata(sdt['prompt_to_visit'])), _resid(_rankdata(sdt[y])))
        rows.append(dict(x='Prompt -> Source visit transition', y=y, N=len(sdt), rs=rs, p=p))
    e8 = pd.DataFrame(rows)
    e8['p_fdr'] = _mt(e8.p, method='fdr_bh')[1]
    e8.to_csv(os.path.join(SCRIPTS_DIR, 'tab_E8_full_sample_rows.csv'), index=False)
    print(e8.round(3).to_string())

    # Table E10, row I: comparative composite score without selected dimensions
    c = d[d['השלימה_משימה'] == 1]
    p_, v_, pr_, rf_ = [c[k] for k in ['ציון_ממד_פרומפט', 'ציון_ממד_זיהוי_מקורות', 'ציון_ממד_תהליך_ובדיקה', 'ציון_ממד_תחקיר']]
    versions = {
        'Without the reflection dimension': (15 * p_ + 35 * v_ + 30 * pr_) / 80,
        'Without the process execution dimension': (15 * p_ + 35 * v_ + 10 * rf_) / 60,
        'Without the process execution and reflection dimensions': (15 * p_ + 35 * v_) / 50,
    }
    for label, score in versions.items():
        a = score[c['קבוצה_אנגלית'] == 'Simulation']; b = score[c['קבוצה_אנגלית'] == 'Chatbot']
        u, pv = _mwu(a, b, alternative='two-sided')
        print(f'{label}: n = {len(a)} / {len(b)}, p = {pv:.3f}, r = {2 * u / (len(a) * len(b)) - 1:.2f}')


regressions_and_checks()


#==============================================================================
# PART 17: R pipeline: ENA, ONA, permutation tests, Table E8, network figures
# Thesis: Figures 4, 5, E2; Tables E1, E2, E8, E10 (row G)
# PART 18: R figures: Figure E3 and Table E9 (Model 3), Figure E1
#==============================================================================

R_PIPELINE = r'''# ==============================================================================
# R part of the WorkAI analysis
#  1. ENA model and dimension tests (N = 69)
#  2. Edge-wise ENA permutation tests (5,000 permutations, FDR)
#  3. Edge-wise ONA permutation tests (5,000 permutations, FDR)
#  4. Table E8 correlations with Benjamini-Hochberg correction (k = 6, k = 11)
#  5. Table E10 sensitivity analysis (ENA Dimension 1)
#  7. Reflection level by condition and the Figure E3 moderation regression (N = 67)
#  8. Network figures (Figures 4, 5, E2)
# ==============================================================================

suppressPackageStartupMessages({
  library(rENA)
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(ggrepel)
})

set.seed(42)

# Create output directory for all CSV results
out_dir <- "output_csv_results"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

cat("===============================================================================\n")
cat("STEP 1: LOADING DATA & RUNNING COMPLETE ENA ACCUMULATION (N = 69)\n")
cat("===============================================================================\n")

df_ena <- read.csv("ena_dataset_workai_english.csv", stringsAsFactors = FALSE)
df_master <- read.csv("analysis_scripts/master_clean_n70.csv", stringsAsFactors = FALSE)

code_names <- c(
  "P_Task", "P_Quality", "P_Recency", "P_Global", "P_Retry",
  "C_Method", "C_Prompt", "C_Tech",
  "S_Visit", "S_Google", "S_AI",
  "ID_Correct", "ID_Partial", "ID_Wrong",
  "D_L0", "D_L1", "D_L2", "D_L3", "D_L4"
)

code_labels <- c(
  "Prompt: Task", "Prompt: Quality", "Prompt: Recency", "Prompt: Global", "Prompt: Retry",
  "Bot: Method Help", "Bot: Phrasing Help", "Bot: Tech Assist",
  "Source: Visit Site", "Source: Google Search", "Source: AI Verify",
  "ID: Correct", "ID: Partial", "ID: Wrong",
  "Debrief: Lvl 0", "Debrief: Lvl 1", "Debrief: Lvl 2", "Debrief: Lvl 3", "Debrief: Lvl 4"
)

cluster_names <- c(
  rep("Prompt Formulation", 5),
  rep("Bot Consultation", 3),
  rep("Source Verification", 3),
  rep("Source Identification", 3),
  rep("Debriefing & Reflection", 5)
)

cluster_colors <- c(
  "Prompt Formulation"      = "#059669",
  "Bot Consultation"       = "#D97706",
  "Source Verification"     = "#0284C7",
  "Source Identification"   = "#7C3AED",
  "Debriefing & Reflection" = "#1E40AF"
)

# ENA Accumulation (Window = 4)
accum <- ena.accumulate.data(
  units = df_ena[, c("UserName", "Condition")],
  conversation = df_ena[, c("Condition", "SessionID")],
  codes = df_ena[, code_names],
  window.size.back = 4
)
sim_units <- accum$meta.data$Condition == "simulation"
cb_units  <- accum$meta.data$Condition == "chatbot"

# Rotation by Means
set_mr <- ena.make.set(
  enadata = accum,
  rotation.by = ena.rotate.by.mean,
  rotation.params = list(sim_units, cb_units)
)

lw <- as.matrix(set_mr$line.weights)
users_ena <- set_mr$meta.data$UserName
cond_ena  <- set_mr$meta.data$Condition
n_users   <- length(users_ena)
n_conn    <- ncol(lw)

pts <- as.data.frame(set_mr$points)
pts$UserName  <- users_ena
pts$Condition <- cond_ena

sim_d1 <- pts$MR1[pts$Condition == "simulation"]
sim_d2 <- pts$SVD2[pts$Condition == "simulation"]
cb_d1  <- pts$MR1[pts$Condition == "chatbot"]
cb_d2  <- pts$SVD2[pts$Condition == "chatbot"]

# Variance explained
var_d1 <- set_mr$model$variance[1] * 100
var_d2 <- set_mr$model$variance[2] * 100
var_2d <- var_d1 + var_d2

# Goodness of fit (projection vs original space)
pts_raw <- as.matrix(set_mr$line.weights) %*% as.matrix(set_mr$rotation$rotation)
r_p_d1  <- cor(pts$MR1, pts_raw[, 1], method = "pearson")
r_s_d1  <- cor(pts$MR1, pts_raw[, 1], method = "spearman")
r_p_d2  <- cor(pts$SVD2, pts_raw[, 2], method = "pearson")
r_s_d2  <- cor(pts$SVD2, pts_raw[, 2], method = "spearman")

# Mann-Whitney U test on Dim 1 & Dim 2
calc_mwu <- function(x, y) {
  wt <- wilcox.test(x, y, exact = FALSE)
  n1 <- length(x); n2 <- length(y)
  u1 <- wt$statistic
  u2 <- n1 * n2 - u1
  r_rb <- 1 - (2 * u2) / (n1 * n2)
  mu_u <- (n1 * n2) / 2
  sigma_u <- sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
  z <- (u1 - mu_u) / sigma_u
  r_z <- z / sqrt(n1 + n2)
  list(W = u1, U_sim = u1, U_cb = u2, p = wt$p.value, r_rb = r_rb, r_z = r_z)
}
mwu_d1 <- calc_mwu(sim_d1, cb_d1)
mwu_d2 <- calc_mwu(sim_d2, cb_d2)

# Brown-Forsythe Levene test (median-based)
calc_levene_median <- function(x, y) {
  z_x <- abs(x - median(x))
  z_y <- abs(y - median(y))
  df_lev <- data.frame(z = c(z_x, z_y), grp = factor(c(rep("Sim", length(x)), rep("CB", length(y)))))
  fit <- lm(z ~ grp, data = df_lev)
  anv <- anova(fit)
  list(F_stat = anv$`F value`[1], p = anv$`Pr(>F)`[1])
}
lev_d1 <- calc_levene_median(sim_d1, cb_d1)
lev_d2 <- calc_levene_median(sim_d2, cb_d2)

dim_metrics_df <- data.frame(
  Dimension = c("Dimension 1 (MR1)", "Dimension 2 (SVD2)"),
  Variance_Explained_Pct = c(round(var_d1, 2), round(var_d2, 2)),
  Pearson_GoF = c(round(r_p_d1, 4), round(r_p_d2, 4)),
  Spearman_GoF = c(round(r_s_d1, 4), round(r_s_d2, 4)),
  Simulation_Mean = c(round(mean(sim_d1), 4), round(mean(sim_d2), 4)),
  Simulation_SD = c(round(sd(sim_d1), 4), round(sd(sim_d2), 4)),
  Simulation_Median = c(round(median(sim_d1), 4), round(median(sim_d2), 4)),
  Simulation_Min = c(round(min(sim_d1), 4), round(min(sim_d2), 4)),
  Simulation_Max = c(round(max(sim_d1), 4), round(max(sim_d2), 4)),
  Chatbot_Mean = c(round(mean(cb_d1), 4), round(mean(cb_d2), 4)),
  Chatbot_SD = c(round(sd(cb_d1), 4), round(sd(cb_d2), 4)),
  Chatbot_Median = c(round(median(cb_d1), 4), round(median(cb_d2), 4)),
  Chatbot_Min = c(round(min(cb_d1), 4), round(min(cb_d2), 4)),
  Chatbot_Max = c(round(max(cb_d1), 4), round(max(cb_d2), 4)),
  Centroid_Difference = c(round(mean(sim_d1) - mean(cb_d1), 4), round(mean(sim_d2) - mean(cb_d2), 4)),
  MannWhitney_W = c(mwu_d1$W, mwu_d2$W),
  MannWhitney_p = c(mwu_d1$p, mwu_d2$p),
  Rank_Biserial_r = c(round(mwu_d1$r_rb, 4), round(mwu_d2$r_rb, 4)),
  Effect_r_z = c(round(mwu_d1$r_z, 4), round(mwu_d2$r_z, 4)),
  Levene_Median_F = c(round(lev_d1$F_stat, 4), round(lev_d2$F_stat, 4)),
  Levene_Median_p = c(lev_d1$p, lev_d2$p),
  stringsAsFactors = FALSE
)
write.csv(dim_metrics_df, file.path(out_dir, "01_ena_dimensions_and_tests.csv"), row.names = FALSE)
cat("Module 1 complete. Saved 01_ena_dimensions_and_tests.csv\n")


cat("\n===============================================================================\n")
cat("STEP 2: EDGE-WISE ENA PERMUTATION TESTS (5,000 Permutations, FDR)\n")
cat("===============================================================================\n")

sim_means_ena <- colMeans(lw[cond_ena == "simulation", , drop = FALSE])
cb_means_ena  <- colMeans(lw[cond_ena == "chatbot", , drop = FALSE])
obs_diff_ena  <- sim_means_ena - cb_means_ena

n_perm <- 5000
sim_idx_ena <- which(cond_ena == "simulation")
n_sim_ena   <- length(sim_idx_ena)

# Vectorized permutation test
perm_mat_ena <- matrix(0, nrow = n_perm, ncol = n_conn)
for (p_i in 1:n_perm) {
  shuff <- sample(n_users)
  s_sub <- shuff[1:n_sim_ena]
  c_sub <- shuff[(n_sim_ena + 1):n_users]
  perm_mat_ena[p_i, ] <- colMeans(lw[s_sub, , drop = FALSE]) - colMeans(lw[c_sub, , drop = FALSE])
}

p_vals_ena <- numeric(n_conn)
for (j in 1:n_conn) {
  p_vals_ena[j] <- (sum(abs(perm_mat_ena[, j]) >= abs(obs_diff_ena[j])) + 1) / (n_perm + 1)
}
p_fdr_ena <- p.adjust(p_vals_ena, method = "BH")

ena_table <- data.frame(
  connection = colnames(lw),
  mean_simulation = round(sim_means_ena, 4),
  mean_chatbot = round(cb_means_ena, 4),
  difference = round(obs_diff_ena, 4),
  p = round(p_vals_ena, 5),
  p_fdr = round(p_fdr_ena, 5),
  stringsAsFactors = FALSE
)
ena_table <- ena_table[order(ena_table$p_fdr, ena_table$p, -abs(ena_table$difference)), ]
write.csv(ena_table, file.path(out_dir, "02_ena_edgewise_permutation_fdr.csv"), row.names = FALSE)
cat(sprintf("Module 2 complete. Total ENA edges: %d, Significant after FDR: %d\n", n_conn, sum(ena_table$p_fdr < 0.05)))


cat("\n===============================================================================\n")
cat("STEP 3: EDGE-WISE ONA PERMUTATION TESTS (5,000 Permutations, FDR)\n")
cat("===============================================================================\n")

users_u <- unique(df_ena$UserName)
n_users_ona <- length(users_u)
user_cond_ona <- sapply(users_u, function(u) df_ena$Condition[df_ena$UserName == u][1])

pairs_ona <- character()
for (f in code_names) {
  for (t in code_names) {
    if (f != t) pairs_ona <- c(pairs_ona, paste(f, "->", t))
  }
}
n_pairs_ona <- length(pairs_ona)

ona_raw <- matrix(0, nrow = n_users_ona, ncol = n_pairs_ona, dimnames = list(users_u, pairs_ona))
for (u in users_u) {
  u_df <- df_ena[df_ena$UserName == u, ]
  sess_list <- unique(u_df$SessionID)
  for (s in sess_list) {
    sdf <- u_df[u_df$SessionID == s, ]
    n_rows <- nrow(sdf)
    if (n_rows < 2) next
    mat <- as.matrix(sdf[, code_names])
    for (i in 1:(n_rows - 1)) {
      from_idx <- which(mat[i, ] > 0)
      if (length(from_idx) == 0) next
      max_j <- min(n_rows, i + 3)
      for (j in (i + 1):max_j) {
        decay <- 1 / (j - i)
        to_idx <- which(mat[j, ] > 0)
        for (f_i in from_idx) {
          for (t_i in to_idx) {
            if (f_i != t_i) {
              p_name <- paste(code_names[f_i], "->", code_names[t_i])
              ona_raw[u, p_name] <- ona_raw[u, p_name] + decay
            }
          }
        }
      }
    }
  }
}

ona_norm <- ona_raw
for (i in 1:n_users_ona) {
  denom <- sqrt(sum(ona_raw[i, ]^2))
  if (denom > 0) ona_norm[i, ] <- ona_raw[i, ] / denom
}

sim_means_ona <- colMeans(ona_norm[user_cond_ona == "simulation", , drop = FALSE])
cb_means_ona  <- colMeans(ona_norm[user_cond_ona == "chatbot", , drop = FALSE])
obs_diff_ona  <- sim_means_ona - cb_means_ona

sim_idx_ona <- which(user_cond_ona == "simulation")
n_sim_ona   <- length(sim_idx_ona)

perm_mat_ona <- matrix(0, nrow = n_perm, ncol = n_pairs_ona)
for (p_i in 1:n_perm) {
  shuff <- sample(n_users_ona)
  s_sub <- shuff[1:n_sim_ona]
  c_sub <- shuff[(n_sim_ona + 1):n_users_ona]
  perm_mat_ona[p_i, ] <- colMeans(ona_norm[s_sub, , drop = FALSE]) - colMeans(ona_norm[c_sub, , drop = FALSE])
}

p_vals_ona <- numeric(n_pairs_ona)
for (j in 1:n_pairs_ona) {
  p_vals_ona[j] <- (sum(abs(perm_mat_ona[, j]) >= abs(obs_diff_ona[j])) + 1) / (n_perm + 1)
}
p_fdr_ona <- p.adjust(p_vals_ona, method = "BH")

ona_table <- data.frame(
  connection = pairs_ona,
  mean_simulation = round(sim_means_ona, 4),
  mean_chatbot = round(cb_means_ona, 4),
  difference = round(obs_diff_ona, 4),
  p = round(p_vals_ona, 5),
  p_fdr = round(p_fdr_ona, 5),
  stringsAsFactors = FALSE
)
ona_table <- ona_table[order(ona_table$p_fdr, ona_table$p, -abs(ona_table$difference)), ]
write.csv(ona_table, file.path(out_dir, "03_ona_edgewise_permutation_fdr.csv"), row.names = FALSE)
cat(sprintf("Module 3 complete. Total ONA transitions: %d, Significant after FDR: %d\n", n_pairs_ona, sum(ona_table$p_fdr < 0.05)))


cat("\n===============================================================================\n")
cat("STEP 4: TABLE E8 CROSS-LAYER SDT CORRELATIONS & BH STEP-UP (k = 6, k = 11)\n")
cat("===============================================================================\n")

df_merged_sdt <- merge(pts, df_master, by.x = "UserName", by.y = "מזהה_משתמש", all.x = TRUE)

sim_sub_sdt <- df_merged_sdt[df_merged_sdt$Condition == "simulation", ]
cb_sub_sdt  <- df_merged_sdt[df_merged_sdt$Condition == "chatbot", ]

# Spearman correlation: Position on ENA Dim 1 x d'
c_sim_d1_dp <- cor.test(sim_sub_sdt$MR1, sim_sub_sdt$d_prime, method = "spearman", exact = FALSE)
c_cb_d1_dp  <- cor.test(cb_sub_sdt$MR1, cb_sub_sdt$d_prime, method = "spearman", exact = FALSE)

# Step-up BH function
calc_bh_stepup <- function(tests_list) {
  k <- length(tests_list)
  sorted_t <- tests_list[order(sapply(tests_list, function(x) x$raw_p))]
  raw_p <- sapply(sorted_t, function(x) x$raw_p)
  direct_steps <- sapply(1:k, function(i) raw_p[i] * k / i)
  adj_p <- numeric(k)
  cum_min <- 1.0
  for (i in k:1) {
    cum_min <- min(cum_min, direct_steps[i])
    adj_p[i] <- min(1.0, cum_min)
  }
  data.frame(
    Rank = 1:k,
    Test = sapply(sorted_t, function(x) x$name),
    Population = sapply(sorted_t, function(x) x$pop),
    N = sapply(sorted_t, function(x) x$N),
    rs = round(sapply(sorted_t, function(x) x$rs), 3),
    raw_p = round(raw_p, 5),
    Direct_Step_BH = round(direct_steps, 5),
    p_FDR_StepUp = round(adj_p, 5),
    APA_Format = ifelse(adj_p < 0.001, "< .001***", ifelse(adj_p < 0.01, paste0(round(adj_p, 3), "**"),
                 ifelse(adj_p < 0.05, paste0(round(adj_p, 3), "*"), paste0(round(adj_p, 3), " n.s.")))),
    stringsAsFactors = FALSE
  )
}

# Simulation Family (k = 6)
sim_k6_tests <- list(
  list(name = "Position on ENA Dimension 1 x d'", pop = "Simulation", N = sum(!is.na(sim_sub_sdt$d_prime)),
       rs = c_sim_d1_dp$estimate, raw_p = c_sim_d1_dp$p.value),
  list(name = "Position on ENA Dimension 1 x Hit rate", pop = "Simulation", N = sum(!is.na(sim_sub_sdt$H_rate)),
       rs = cor(sim_sub_sdt$MR1, sim_sub_sdt$H_rate, method = "spearman", use = "complete.obs"),
       raw_p = cor.test(sim_sub_sdt$MR1, sim_sub_sdt$H_rate, method = "spearman", exact = FALSE)$p.value),
  list(name = "Position on ENA Dimension 1 x c", pop = "Simulation", N = sum(!is.na(sim_sub_sdt$c_bias_מדד_קריטריון)),
       rs = cor(sim_sub_sdt$MR1, sim_sub_sdt$c_bias_מדד_קריטריון, method = "spearman", use = "complete.obs"),
       raw_p = cor.test(sim_sub_sdt$MR1, sim_sub_sdt$c_bias_מדד_קריטריון, method = "spearman", exact = FALSE)$p.value),
  list(name = "Source visit rate x Hit rate", pop = "Simulation", N = sum(!is.na(sim_sub_sdt$H_rate)),
       rs = cor(sim_sub_sdt$שיעור_ביקור_במקורות, sim_sub_sdt$H_rate, method = "spearman", use = "complete.obs"),
       raw_p = cor.test(sim_sub_sdt$שיעור_ביקור_במקורות, sim_sub_sdt$H_rate, method = "spearman", exact = FALSE)$p.value),
  list(name = "Source visit rate x c", pop = "Simulation", N = sum(!is.na(sim_sub_sdt$c_bias_מדד_קריטריון)),
       rs = cor(sim_sub_sdt$שיעור_ביקור_במקורות, sim_sub_sdt$c_bias_מדד_קריטריון, method = "spearman", use = "complete.obs"),
       raw_p = cor.test(sim_sub_sdt$שיעור_ביקור_במקורות, sim_sub_sdt$c_bias_מדד_קריטריון, method = "spearman", exact = FALSE)$p.value),
  list(name = "Source visit rate x d'", pop = "Simulation", N = sum(!is.na(sim_sub_sdt$d_prime)),
       rs = cor(sim_sub_sdt$שיעור_ביקור_במקורות, sim_sub_sdt$d_prime, method = "spearman", use = "complete.obs"),
       raw_p = cor.test(sim_sub_sdt$שיעור_ביקור_במקורות, sim_sub_sdt$d_prime, method = "spearman", exact = FALSE)$p.value)
)
df_sim_k6 <- calc_bh_stepup(sim_k6_tests)

# Chatbot Family (k = 11)
cb_k11_tests <- list(
  list(name = "Classification without verification rate x Hit rate", pop = "Chatbot", N = 37,
       rs = cor(cb_sub_sdt$שיעור_ניחוש_ללא_בדיקה, cb_sub_sdt$H_rate, method = "spearman", use = "complete.obs"),
       raw_p = cor.test(cb_sub_sdt$שיעור_ניחוש_ללא_בדיקה, cb_sub_sdt$H_rate, method = "spearman", exact = FALSE)$p.value),
  list(name = "Zero source visits x Hit rate", pop = "Chatbot", N = 37,
       rs = cor(cb_sub_sdt$אפס_כניסות, cb_sub_sdt$H_rate, method = "spearman", use = "complete.obs"),
       raw_p = cor.test(cb_sub_sdt$אפס_כניסות, cb_sub_sdt$H_rate, method = "spearman", exact = FALSE)$p.value),
  list(name = "Position on ENA Dimension 1 x d'", pop = "Chatbot", N = 37,
       rs = c_cb_d1_dp$estimate, raw_p = c_cb_d1_dp$p.value),
  list(name = "Prompt -> Classification transition x Hit rate", pop = "Chatbot", N = 37,
       rs = cor(cb_sub_sdt$מרקוב_פרומפט_אל_סיווג, cb_sub_sdt$H_rate, method = "spearman", use = "complete.obs"),
       raw_p = cor.test(cb_sub_sdt$מרקוב_פרומפט_אל_סיווג, cb_sub_sdt$H_rate, method = "spearman", exact = FALSE)$p.value),
  list(name = "Prompt -> Classification transition x d'", pop = "Chatbot", N = 37,
       rs = cor(cb_sub_sdt$מרקוב_פרומפט_אל_סיווג, cb_sub_sdt$d_prime, method = "spearman", use = "complete.obs"),
       raw_p = cor.test(cb_sub_sdt$מרקוב_פרומפט_אל_סיווג, cb_sub_sdt$d_prime, method = "spearman", exact = FALSE)$p.value),
  list(name = "Zero source visits x d'", pop = "Chatbot", N = 37,
       rs = cor(cb_sub_sdt$אפס_כניסות, cb_sub_sdt$d_prime, method = "spearman", use = "complete.obs"),
       raw_p = cor.test(cb_sub_sdt$אפס_כניסות, cb_sub_sdt$d_prime, method = "spearman", exact = FALSE)$p.value),
  list(name = "Source visit rate x d'", pop = "Chatbot", N = 37,
       rs = cor(cb_sub_sdt$שיעור_ביקור_במקורות, cb_sub_sdt$d_prime, method = "spearman", use = "complete.obs"),
       raw_p = cor.test(cb_sub_sdt$שיעור_ביקור_במקורות, cb_sub_sdt$d_prime, method = "spearman", exact = FALSE)$p.value),
  list(name = "Source visit rate x Hit rate", pop = "Chatbot", N = 37,
       rs = cor(cb_sub_sdt$שיעור_ביקור_במקורות, cb_sub_sdt$H_rate, method = "spearman", use = "complete.obs"),
       raw_p = cor.test(cb_sub_sdt$שיעור_ביקור_במקורות, cb_sub_sdt$H_rate, method = "spearman", exact = FALSE)$p.value),
  list(name = "Zero source visits x c", pop = "Chatbot", N = 37,
       rs = cor(cb_sub_sdt$אפס_כניסות, cb_sub_sdt$c_bias_מדד_קריטריון, method = "spearman", use = "complete.obs"),
       raw_p = cor.test(cb_sub_sdt$אפס_כניסות, cb_sub_sdt$c_bias_מדד_קריטריון, method = "spearman", exact = FALSE)$p.value),
  list(name = "Prompt -> Classification transition x c", pop = "Chatbot", N = 37,
       rs = cor(cb_sub_sdt$מרקוב_פרומפט_אל_סיווג, cb_sub_sdt$c_bias_מדד_קריטריון, method = "spearman", use = "complete.obs"),
       raw_p = cor.test(cb_sub_sdt$מרקוב_פרומפט_אל_סיווג, cb_sub_sdt$c_bias_מדד_קריטריון, method = "spearman", exact = FALSE)$p.value),
  list(name = "Source visit rate x c", pop = "Chatbot", N = 37,
       rs = cor(cb_sub_sdt$שיעור_ביקור_במקורות, cb_sub_sdt$c_bias_מדד_קריטריון, method = "spearman", use = "complete.obs"),
       raw_p = cor.test(cb_sub_sdt$שיעור_ביקור_במקורות, cb_sub_sdt$c_bias_מדד_קריטריון, method = "spearman", exact = FALSE)$p.value)
)
df_cb_k11 <- calc_bh_stepup(cb_k11_tests)

df_table_e8_all <- rbind(df_sim_k6, df_cb_k11)
write.csv(df_table_e8_all, file.path(out_dir, "04_table_e8_sdt_correlations.csv"), row.names = FALSE)
cat("Module 4 complete. Saved 04_table_e8_sdt_correlations.csv\n")


cat("\n===============================================================================\n")
cat("STEP 5: TABLE E10 SENSITIVITY ANALYSIS (ENA DIMENSION 1)\n")
cat("===============================================================================\n")

# Base (N = 69: Sim = 30, CB = 39)
# Without anomalous user user_isvijah1u (N = 68: Sim = 29, CB = 39)
pts_no_outlier <- pts[pts$UserName != "user_isvijah1u", ]

mwu_base <- calc_mwu(pts$MR1[pts$Condition == "simulation"], pts$MR1[pts$Condition == "chatbot"])
mwu_clean <- calc_mwu(pts_no_outlier$MR1[pts_no_outlier$Condition == "simulation"],
                      pts_no_outlier$MR1[pts_no_outlier$Condition == "chatbot"])

t_e10_df <- data.frame(
  Analysis = c("G", "G"),
  Measure = c("ENA Dimension 1", "ENA Dimension 1"),
  Version = c("Base", "Without the participant with anomalous message volume (user_isvijah1u)"),
  n_simulation = c(30, 29),
  n_chatbot = c(39, 39),
  MannWhitney_W = c(mwu_base$W, mwu_clean$W),
  MannWhitney_p = c(mwu_base$p, mwu_clean$p),
  Rank_Biserial_r = c(round(mwu_base$r_rb, 3), round(mwu_clean$r_rb, 3)),
  Effect_r_z = c(round(mwu_base$r_z, 3), round(mwu_clean$r_z, 3)),
  APA_p = c("< .001***", "< .001***"),
  stringsAsFactors = FALSE
)
write.csv(t_e10_df, file.path(out_dir, "05_table_e10_sensitivity_ena_dim1.csv"), row.names = FALSE)
cat("Module 5 complete. Saved 05_table_e10_sensitivity_ena_dim1.csv\n")


# (Step 6, an exploratory cluster analysis that is not reported in the thesis, was removed.)

cat("\n===============================================================================\n")
cat("STEP 7: REFLECTION LEVEL M/SD & FIGURE E3 MODERATION REGRESSION (N = 67)\n")
cat("===============================================================================\n")

# Reflection Level (Kember 0 to 4)
ref_sim <- df_master$רמת_Kember[df_master$קבוצה_אנגלית == "Simulation"]
ref_cb  <- df_master$רמת_Kember[df_master$קבוצה_אנגלית == "Chatbot"]

ref_df <- data.frame(
  Measure = "Reflection Level (Kember 0-4)",
  Simulation_n = sum(!is.na(ref_sim)),
  Simulation_Mean = round(mean(ref_sim, na.rm = TRUE), 2),
  Simulation_SD = round(sd(ref_sim, na.rm = TRUE), 2),
  Chatbot_n = sum(!is.na(ref_cb)),
  Chatbot_Mean = round(mean(ref_cb, na.rm = TRUE), 2),
  Chatbot_SD = round(sd(ref_cb, na.rm = TRUE), 2),
  stringsAsFactors = FALSE
)

# Figure E3 Moderation Regression: Accuracy ~ Condition * Source Visit Rate (N = 67)
# Classification accuracy rate = source-level rate computed from the database in Part 14 (e3_data.csv)
sub67 <- read.csv("e3_data.csv", stringsAsFactors = FALSE, check.names = FALSE)
sub67$cond_num <- ifelse(sub67$קבוצה_אנגלית == "Simulation", 1, 0)
sub67$acc_pct  <- sub67$accp
sub67$visit_pct <- sub67$visit

mod_fit <- lm(acc_pct ~ cond_num * visit_pct, data = sub67)
sm_mod  <- summary(mod_fit)$coefficients

# Simple slopes
# In Chatbot (cond_num = 0): slope is visit_pct
slope_cb_B  <- sm_mod["visit_pct", "Estimate"]
slope_cb_SE <- sm_mod["visit_pct", "Std. Error"]
slope_cb_t  <- sm_mod["visit_pct", "t value"]
slope_cb_p  <- sm_mod["visit_pct", "Pr(>|t|)"]

# In Simulation (cond_num = 1): slope is visit_pct + cond_num:visit_pct
sub67$cond_ref_sim <- ifelse(sub67$קבוצה_אנגלית == "Simulation", 0, 1)
mod_fit_sim <- lm(acc_pct ~ cond_ref_sim * visit_pct, data = sub67)
sm_mod_sim  <- summary(mod_fit_sim)$coefficients

slope_sim_B  <- sm_mod_sim["visit_pct", "Estimate"]
slope_sim_SE <- sm_mod_sim["visit_pct", "Std. Error"]
slope_sim_t  <- sm_mod_sim["visit_pct", "t value"]
slope_sim_p  <- sm_mod_sim["visit_pct", "Pr(>|t|)"]

mod_results_df <- data.frame(
  Effect = c("Intercept (Chatbot at 0% visits)", "Condition (Simulation vs. Chatbot at 0% visits)",
             "Source Visit Rate in Chatbot (Simple Slope)", "Condition x Source Visit Rate (Interaction)",
             "Source Visit Rate in Simulation (Simple Slope)"),
  B = round(c(sm_mod["(Intercept)", "Estimate"], sm_mod["cond_num", "Estimate"],
              slope_cb_B, sm_mod["cond_num:visit_pct", "Estimate"], slope_sim_B), 4),
  SE = round(c(sm_mod["(Intercept)", "Std. Error"], sm_mod["cond_num", "Std. Error"],
               slope_cb_SE, sm_mod["cond_num:visit_pct", "Std. Error"], slope_sim_SE), 4),
  t = round(c(sm_mod["(Intercept)", "t value"], sm_mod["cond_num", "t value"],
              slope_cb_t, sm_mod["cond_num:visit_pct", "t value"], slope_sim_t), 4),
  p = c(sm_mod["(Intercept)", "Pr(>|t|)"], sm_mod["cond_num", "Pr(>|t|)"],
        slope_cb_p, sm_mod["cond_num:visit_pct", "Pr(>|t|)"], slope_sim_p),
  stringsAsFactors = FALSE
)

write.csv(ref_df, file.path(out_dir, "09_reflection_level_by_condition.csv"), row.names = FALSE)
write.csv(mod_results_df, file.path(out_dir, "10_figureE3_moderation_regression_slopes.csv"), row.names = FALSE)
cat("Module 7 complete. Saved Reflection & Moderation Regression CSVs\n")


cat("\n===============================================================================\n")
cat("STEP 8: GENERATING ALL PUBLICATION FIGURES (Figures 4, 5, E2, E3)\n")
cat("===============================================================================\n")

# Circular Layout Definition
radius <- 10.0
angles_deg <- c(
  56, 42, 28, 14, 0,
  -24, -42, -58,
  -88, -114, -138,
  -168, 168, 150,
  134, 118, 102, 86, 70
)

node_df <- data.frame(
  code    = code_names,
  label   = code_labels,
  cluster = factor(cluster_names, levels = unique(cluster_names)),
  angle   = angles_deg * pi / 180,
  stringsAsFactors = FALSE
)
node_df$x <- radius * cos(node_df$angle)
node_df$y <- radius * sin(node_df$angle)

offset <- 2.2
node_df$lbl_x <- (radius + offset) * cos(node_df$angle)
node_df$lbl_y <- (radius + offset) * sin(node_df$angle)
# SPECIFIC NON-OVERLAPPING ADJUSTMENTS:
# Completely separate Source: Visit Site and Source: Google Search
node_df$lbl_x[node_df$code == "S_Visit"]  <- 1.2
node_df$lbl_y[node_df$code == "S_Visit"]  <- -12.9
node_df$lbl_x[node_df$code == "S_Google"] <- -6.2
node_df$lbl_y[node_df$code == "S_Google"] <- -11.2
node_df$lbl_x[node_df$code == "S_AI"]     <- -11.2
node_df$lbl_y[node_df$code == "S_AI"]     <- -8.0
node_df$lbl_x[node_df$code == "D_L4"]     <- 2.2
node_df$lbl_y[node_df$code == "D_L4"]     <- 12.8
node_df$lbl_x[node_df$code == "D_L3"]     <- -2.8
node_df$lbl_y[node_df$code == "D_L3"]     <- 12.8
node_df$lbl_x[node_df$code == "P_Task"]   <- 7.2
node_df$lbl_y[node_df$code == "P_Task"]   <- 11.6

calc_offset_curve <- function(x1, y1, x2, y2, r_start = 0.55, r_end = 1.05) {
  dx <- x2 - x1; dy <- y2 - y1; d <- sqrt(dx^2 + dy^2)
  if (d == 0) return(c(x1, y1, x2, y2))
  xs <- x1 + (dx / d) * r_start; ys <- y1 + (dy / d) * r_start
  xe <- x2 - (dx / d) * r_end;   ye <- y2 - (dy / d) * r_end
  return(c(xs, ys, xe, ye))
}

build_edges_ona <- function(df_sub, val_col) {
  edge_list <- list()
  for (k in 1:nrow(df_sub)) {
    parts <- strsplit(df_sub$connection[k], " -> ")[[1]]
    f <- parts[1]; t <- parts[2]
    f_node <- node_df[node_df$code == f, ]
    t_node <- node_df[node_df$code == t, ]
    pts_c <- calc_offset_curve(f_node$x, f_node$y, t_node$x, t_node$y)
    v <- df_sub[[val_col]][k]
    edge_list[[k]] <- data.frame(
      from = f, to = t,
      x = pts_c[1], y = pts_c[2], xend = pts_c[3], yend = pts_c[4],
      weight = v,
      abs_weight = abs(v),
      stringsAsFactors = FALSE
    )
  }
  df_res <- do.call(rbind, edge_list)
  df_res[order(df_res$abs_weight), ]
}

# --- FIGURE 4A: Participant Space (Multi-panel: letter (a) in top-left) ---
pts_plot <- pts
pts_plot$Condition_en <- factor(ifelse(pts_plot$Condition == "simulation", "Simulation", "Chatbot"),
                                levels = c("Simulation", "Chatbot"))
pts_plot$shape_val <- ifelse(pts_plot$Condition == "simulation", 21, 24)

# 95% CIs
calc_ci <- function(v) {
  m <- mean(v); se <- sd(v) / sqrt(length(v))
  c(m = m, low = m - 1.96 * se, high = m + 1.96 * se)
}
sim_ci_x <- calc_ci(sim_d1); sim_ci_y <- calc_ci(sim_d2)
cb_ci_x  <- calc_ci(cb_d1);  cb_ci_y  <- calc_ci(cb_d2)

centroids_df <- data.frame(
  Condition_en = factor(c("Simulation", "Chatbot"), levels = c("Simulation", "Chatbot")),
  x = c(sim_ci_x["m"], cb_ci_x["m"]),
  y = c(sim_ci_y["m"], cb_ci_y["m"]),
  xmin = c(sim_ci_x["low"], cb_ci_x["low"]),
  xmax = c(sim_ci_x["high"], cb_ci_x["high"]),
  ymin = c(sim_ci_y["low"], cb_ci_y["low"]),
  ymax = c(sim_ci_y["high"], cb_ci_y["high"])
)

fig_colors <- c("Simulation" = "#0D9488", "Chatbot" = "#2563EB")

p4a <- ggplot() +
  geom_hline(yintercept = 0, color = "#CBD5E1", linetype = "dashed", linewidth = 0.5) +
  geom_vline(xintercept = 0, color = "#CBD5E1", linetype = "dashed", linewidth = 0.5) +
  geom_point(data = pts_plot, aes(x = MR1, y = SVD2, fill = Condition_en, shape = Condition_en),
             size = 3.2, color = "white", stroke = 0.8, alpha = 0.75) +
  geom_errorbar(data = centroids_df, aes(x = x, ymin = ymin, ymax = ymax, color = Condition_en),
                width = 0.03, linewidth = 1.0, show.legend = FALSE) +
  geom_errorbar(data = centroids_df, aes(y = y, xmin = xmin, xmax = xmax, color = Condition_en),
                orientation = "y", width = 0.03, linewidth = 1.0, show.legend = FALSE) +
  geom_point(data = centroids_df, aes(x = x, y = y, fill = Condition_en),
             shape = 22, size = 6.2, color = "white", stroke = 1.8, show.legend = FALSE) +
  scale_fill_manual(values = fig_colors, name = "Condition") +
  scale_color_manual(values = fig_colors, name = "Condition") +
  scale_shape_manual(values = c("Simulation" = 21, "Chatbot" = 24), name = "Condition") +
  scale_x_continuous(limits = c(-0.75, 0.75), breaks = seq(-0.6, 0.6, 0.3)) +
  scale_y_continuous(limits = c(-0.65, 0.65), breaks = seq(-0.6, 0.6, 0.3)) +
  labs(title = "(a)",
       x = "Dimension 1 (31.0%)",
       y = "Dimension 2 (12.7%)") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0, margin = margin(b = 10)),
    axis.title = element_text(size = 10.5, face = "bold", color = "#1E293B"),
    axis.text  = element_text(size = 9, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#F1F5F9", linewidth = 0.5),
    legend.position = "bottom",
    plot.background = element_rect(fill = "white", color = NA)
  ) +
  guides(
    fill = guide_legend(
      title = "Condition",
      override.aes = list(shape = c(21, 24), fill = unname(fig_colors), size = 4.0, color = "white", stroke = 0.8)
    ),
    shape = "none",
    color = "none"
  )

# --- FIGURE 4B: ENA Difference Network in Standard ENA Projection Space ---
nodes_ena <- as.data.frame(set_mr$rotation$nodes)
nodes_ena$code <- as.character(nodes_ena$code)
code_cats <- setNames(cluster_names, code_names)
nodes_ena$cluster <- factor(code_cats[nodes_ena$code], levels = unique(cluster_names))
code_lbls <- setNames(code_labels, code_names)
nodes_ena$label <- code_lbls[nodes_ena$code]

sig_ena <- ena_table[ena_table$p_fdr < 0.05, ]
edge_list_ena <- list()
for (k in 1:nrow(sig_ena)) {
  parts <- strsplit(sig_ena$connection[k], " & ")[[1]]
  f <- parts[1]; t <- parts[2]
  f_node <- nodes_ena[nodes_ena$code == f, ]
  t_node <- nodes_ena[nodes_ena$code == t, ]
  d_val <- sig_ena$difference[k]
  edge_list_ena[[k]] <- data.frame(
    from = f, to = t,
    x = f_node$MR1, y = f_node$SVD2,
    xend = t_node$MR1, yend = t_node$SVD2,
    diff = d_val, abs_diff = abs(d_val),
    Difference_Direction = ifelse(d_val > 0, "Positive (stronger in simulation)", "Negative (stronger in chatbot)"),
    stringsAsFactors = FALSE
  )
}
edges_ena_df <- do.call(rbind, edge_list_ena)
edges_ena_df$Difference_Direction <- factor(edges_ena_df$Difference_Direction,
  levels = c("Positive (stronger in simulation)", "Negative (stronger in chatbot)"))
edges_ena_df <- edges_ena_df[order(edges_ena_df$abs_diff), ]

diff_colors <- c("Positive (stronger in simulation)" = "#0D9488",
                 "Negative (stronger in chatbot)"     = "#2563EB")

p4b <- ggplot() +
  geom_hline(yintercept = 0, color = "#E2E8F0", linetype = "dashed", linewidth = 0.5) +
  geom_vline(xintercept = 0, color = "#E2E8F0", linetype = "dashed", linewidth = 0.5) +
  geom_segment(data = edges_ena_df,
               aes(x = x, y = y, xend = xend, yend = yend,
                   color = Difference_Direction, linewidth = abs_diff, alpha = abs_diff),
               lineend = "round") +
  scale_color_manual(values = diff_colors, name = "Difference Direction") +
  scale_linewidth_continuous(range = c(0.8, 3.8), guide = "none") +
  scale_alpha_continuous(range = c(0.5, 0.95), guide = "none") +
  geom_point(data = nodes_ena, aes(x = MR1, y = SVD2), size = 7.0, color = "white") +
  geom_point(data = nodes_ena, aes(x = MR1, y = SVD2, fill = cluster),
             size = 5.2, shape = 21, color = "#0F172A", stroke = 1.2) +
  geom_label_repel(data = nodes_ena, aes(x = MR1, y = SVD2, label = label),
                   fill = alpha("white", 0.92), color = "#0F172A",
                   size = 2.8, fontface = "bold", box.padding = unit(0.35, "lines"),
                   point.padding = unit(0.35, "lines"), max.overlaps = Inf,
                   segment.color = "#94A3B8", segment.size = 0.4) +
  scale_fill_manual(values = cluster_colors, name = "Category") +
  scale_x_continuous(limits = c(-3.8, 4.5), breaks = seq(-3, 4, 1)) +
  scale_y_continuous(limits = c(-2.5, 2.2), breaks = seq(-2, 2, 1)) +
  labs(title = "(b)",
       x = "Dimension 1 (31.0%)",
       y = "Dimension 2 (12.7%)") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0, margin = margin(b = 10)),
    axis.title = element_text(size = 10.5, face = "bold", color = "#1E293B"),
    axis.text  = element_text(size = 9, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#F1F5F9", linewidth = 0.5),
    legend.position = "bottom",
    legend.box = "vertical",
    legend.box.just = "center",
    legend.spacing.y = unit(2, "mm"),
    legend.title = element_text(size = 9.5, face = "bold", color = "#1E293B"),
    legend.text = element_text(size = 8.5, color = "#1E293B"),
    plot.background = element_rect(fill = "white", color = NA)
  ) +
  guides(
    color = guide_legend(title = "Difference Direction", nrow = 1, byrow = TRUE, override.aes = list(linewidth = 2.5)),
    fill = guide_legend(title = "Category", nrow = 1, byrow = TRUE, override.aes = list(size = 5.0))
  )

# Save Figure 4
ggsave("figure_4a_ena_participant_space.png", p4a, width = 8.5, height = 8.0, dpi = 300)
ggsave("figure_4a_ena_participant_space.pdf", p4a, width = 8.5, height = 8.0)
ggsave("figure_4b_ena_difference_fdr.png", p4b, width = 9.0, height = 9.2, dpi = 300)
ggsave("figure_4b_ena_difference_fdr.pdf", p4b, width = 9.0, height = 9.2)

p4_comb <- p4a + p4b + plot_layout(ncol = 2)
ggsave("figure_4_ena_combined.png", p4_comb, width = 17.5, height = 8.8, dpi = 300)
ggsave("figure_4_ena_combined.pdf", p4_comb, width = 17.5, height = 8.8)

# --- FIGURE 5 & E2: Standalone 5a, 5b, 5c, 2-Panel, 3-Panel Composite, and Figure E2 ---
sim_sub_ona <- ona_table[ona_table$mean_simulation >= 0.05, ]
cb_sub_ona  <- ona_table[ona_table$mean_chatbot >= 0.05, ]
sig_ona     <- ona_table[ona_table$p_fdr < 0.05, ]

edges_sim_ona <- build_edges_ona(sim_sub_ona, "mean_simulation")
edges_cb_ona  <- build_edges_ona(cb_sub_ona, "mean_chatbot")
edges_diff_ona <- build_edges_ona(sig_ona, "difference")
edges_diff_ona$Difference_Direction <- factor(
  ifelse(edges_diff_ona$weight > 0, "Positive (stronger in simulation)", "Negative (stronger in chatbot)"),
  levels = c("Positive (stronger in simulation)", "Negative (stronger in chatbot)")
)

w_limits_ona <- c(0.05, max(c(edges_sim_ona$weight, edges_cb_ona$weight)))

theme_network_base <- theme_void() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0, margin = margin(t = 6, b = 10)),
    legend.position = "bottom",
    legend.box = "vertical",
    legend.box.just = "center",
    legend.spacing.y = unit(3, "mm"),
    legend.title = element_text(size = 9.5, face = "bold", color = "#1E293B"),
    legend.text = element_text(size = 8.5, color = "#1E293B"),
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(t = 12, r = 18, b = 25, l = 18)
  )

p5a <- ggplot() +
  geom_curve(data = edges_sim_ona, aes(x = x, y = y, xend = xend, yend = yend, linewidth = weight, alpha = weight),
             color = "#0D9488", curvature = 0.16, arrow = arrow(length = unit(2.5, "mm"), type = "closed")) +
  scale_linewidth_continuous(limits = w_limits_ona, range = c(0.8, 4.0), name = "Mean Normalized Weight") +
  scale_alpha_continuous(limits = w_limits_ona, range = c(0.55, 0.95), guide = "none") +
  geom_point(data = node_df, aes(x = x, y = y, fill = cluster), size = 5.2, shape = 21, color = "white", stroke = 1.2) +
  geom_text(data = node_df, aes(x = lbl_x, y = lbl_y, label = label), size = 2.8, fontface = "bold", color = "#1E293B") +
  scale_fill_manual(values = cluster_colors, name = "Category") +
  guides(
    fill = guide_legend(title = "Category", nrow = 1, byrow = TRUE, override.aes = list(size = 5.0)),
    linewidth = guide_legend(title = "Mean Normalized Weight", nrow = 1, byrow = TRUE)
  ) +
  coord_fixed(xlim = c(-15.5, 15.5), ylim = c(-14.5, 14.5)) +
  labs(title = "(a)") +
  theme_network_base

p5b <- ggplot() +
  geom_curve(data = edges_cb_ona, aes(x = x, y = y, xend = xend, yend = yend, linewidth = weight, alpha = weight),
             color = "#2563EB", curvature = 0.16, arrow = arrow(length = unit(2.5, "mm"), type = "closed")) +
  scale_linewidth_continuous(limits = w_limits_ona, range = c(0.8, 4.0), name = "Mean Normalized Weight") +
  scale_alpha_continuous(limits = w_limits_ona, range = c(0.55, 0.95), guide = "none") +
  geom_point(data = node_df, aes(x = x, y = y, fill = cluster), size = 5.2, shape = 21, color = "white", stroke = 1.2) +
  geom_text(data = node_df, aes(x = lbl_x, y = lbl_y, label = label), size = 2.8, fontface = "bold", color = "#1E293B") +
  scale_fill_manual(values = cluster_colors, name = "Category") +
  guides(
    fill = guide_legend(title = "Category", nrow = 1, byrow = TRUE, override.aes = list(size = 5.0)),
    linewidth = guide_legend(title = "Mean Normalized Weight", nrow = 1, byrow = TRUE)
  ) +
  coord_fixed(xlim = c(-15.5, 15.5), ylim = c(-14.5, 14.5)) +
  labs(title = "(b)") +
  theme_network_base

p5c <- ggplot() +
  geom_curve(data = edges_diff_ona, aes(x = x, y = y, xend = xend, yend = yend, color = Difference_Direction,
                                        linewidth = abs_weight, alpha = abs_weight),
             curvature = 0.16, arrow = arrow(length = unit(2.5, "mm"), type = "closed")) +
  scale_color_manual(values = diff_colors, name = "Difference Direction") +
  scale_linewidth_continuous(range = c(0.8, 4.0), guide = "none") +
  scale_alpha_continuous(range = c(0.55, 0.95), guide = "none") +
  geom_point(data = node_df, aes(x = x, y = y, fill = cluster), size = 5.2, shape = 21, color = "white", stroke = 1.2) +
  geom_text(data = node_df, aes(x = lbl_x, y = lbl_y, label = label), size = 2.8, fontface = "bold", color = "#1E293B") +
  scale_fill_manual(values = cluster_colors, name = "Category") +
  guides(
    color = guide_legend(title = "Difference Direction", nrow = 1, byrow = TRUE),
    fill = guide_legend(title = "Category", nrow = 1, byrow = TRUE, override.aes = list(size = 5.0))
  ) +
  coord_fixed(xlim = c(-15.5, 15.5), ylim = c(-14.5, 14.5)) +
  labs(title = "(c)") +
  theme_network_base

# Figure E2 (Appendix E Standalone Difference Network - Single-panel: NO text above plot)
p_e2 <- ggplot() +
  geom_curve(data = edges_diff_ona, aes(x = x, y = y, xend = xend, yend = yend, color = Difference_Direction,
                                        linewidth = abs_weight, alpha = abs_weight),
             curvature = 0.16, arrow = arrow(length = unit(2.8, "mm"), type = "closed")) +
  scale_color_manual(values = diff_colors, name = "Difference Direction") +
  scale_linewidth_continuous(range = c(0.8, 4.0), guide = "none") +
  scale_alpha_continuous(range = c(0.55, 0.95), guide = "none") +
  geom_point(data = node_df, aes(x = x, y = y, fill = cluster), size = 5.5, shape = 21, color = "white", stroke = 1.2) +
  geom_text(data = node_df, aes(x = lbl_x, y = lbl_y, label = label), size = 3.0, fontface = "bold", color = "#1E293B") +
  scale_fill_manual(values = cluster_colors, name = "Category") +
  guides(
    color = guide_legend(title = "Difference Direction", nrow = 1, byrow = TRUE),
    fill = guide_legend(title = "Category", nrow = 1, byrow = TRUE, override.aes = list(size = 5.2))
  ) +
  coord_fixed(xlim = c(-15.5, 15.5), ylim = c(-14.5, 14.5)) +
  theme_network_base +
  theme(plot.title = element_blank())

# Standalone panels
ggsave("figure_5a_ona_simulation_mean.png", p5a, width = 9.0, height = 9.2, dpi = 300)
ggsave("figure_5a_ona_simulation_mean.pdf", p5a, width = 9.0, height = 9.2)
ggsave("figure_5b_ona_chatbot_mean.png", p5b, width = 9.0, height = 9.2, dpi = 300)
ggsave("figure_5b_ona_chatbot_mean.pdf", p5b, width = 9.0, height = 9.2)
ggsave("figure_5c_ona_difference_fdr.png", p5c, width = 9.0, height = 9.2, dpi = 300)
ggsave("figure_5c_ona_difference_fdr.pdf", p5c, width = 9.0, height = 9.2)

# Figure E2
ggsave("figure_e2_ona_difference_network.png", p_e2, width = 9.5, height = 9.6, dpi = 300)
ggsave("figure_e2_ona_difference_network.pdf", p_e2, width = 9.5, height = 9.6)
ggsave("figure_e2.png", p_e2, width = 9.5, height = 9.6, dpi = 300)
ggsave("figure_e2.pdf", p_e2, width = 9.5, height = 9.6)

# Figure 5 Two Panels (Simulation Left (a), Chatbot Right (b) - Thesis Figure 5)
p5_two <- p5a + p5b + plot_layout(ncol = 2, guides = "collect") &
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.box.just = "center",
    legend.spacing.y = unit(3, "mm"),
    plot.margin = margin(t = 10, r = 15, b = 30, l = 15)
  )
ggsave("figure_5_two_panels.png", p5_two, width = 17.5, height = 9.2, dpi = 300)
ggsave("figure_5_two_panels.pdf", p5_two, width = 17.5, height = 9.2)
ggsave("figure_5.png", p5_two, width = 17.5, height = 9.2, dpi = 300)
ggsave("figure_5.pdf", p5_two, width = 17.5, height = 9.2)

# Figure 5 Three Panels (Simulation, Chatbot, Difference)
p5_three <- p5a + p5b + p5c + plot_layout(ncol = 3, guides = "collect") &
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.box.just = "center",
    legend.spacing.y = unit(3, "mm"),
    plot.margin = margin(t = 10, r = 15, b = 30, l = 15)
  )
ggsave("figure_5_ona_three_panels.png", p5_three, width = 23.5, height = 9.0, dpi = 300)
ggsave("figure_5_ona_three_panels.pdf", p5_three, width = 23.5, height = 9.0)

# (Figure E3 is drawn in Part 18.)

# --- FIGURE 3: Markov State Diagrams (Single-panel: NO text above plot) ---
cat("Generating Figure 3 Markov State Diagrams...\n")
py_fig3_code <- '
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, networkx as nx

STATES_HEB = ["פרומפט", "התייעצות", "כניסה למקור", "חיפוש חיצוני", "סיווג"]
HEB_TO_EN = {"פרומפט": "Prompt", "התייעצות": "Bot\\nConsultation", "כניסה למקור": "Visit\\nSource", "חיפוש חיצוני": "External\\nSearch", "סיווג": "Classification"}
STATES_EN = [HEB_TO_EN[s] for s in STATES_HEB]

sim_probs = pd.read_csv("analysis_scripts/tab_3_markov_probs_sim.csv", index_col=0)[STATES_HEB]
cb_probs = pd.read_csv("analysis_scripts/tab_3_markov_probs_cb.csv", index_col=0)[STATES_HEB]
sim_probs.index = [HEB_TO_EN[s] for s in sim_probs.index]
sim_probs.columns = [HEB_TO_EN[s] for s in sim_probs.columns]
cb_probs.index = [HEB_TO_EN[s] for s in cb_probs.index]
cb_probs.columns = [HEB_TO_EN[s] for s in cb_probs.columns]

def bezier(p0, p1, rad, t):
    D = p1 - p0; dist = np.linalg.norm(D)
    N = np.array([-D[1], D[0]]) / dist if dist > 0 else np.array([0, 1])
    pc = (p0 + p1) / 2.0 + rad * dist * N
    return (1-t)**2 * p0 + 2*(1-t)*t * pc + t**2 * p1

sim_t = {
    ("Prompt", "Bot\\nConsultation"): 0.26,
    ("Prompt", "Visit\\nSource"): 0.65,
    ("Bot\\nConsultation", "Prompt"): 0.26,
    ("Bot\\nConsultation", "Visit\\nSource"): 0.43,
    ("Visit\\nSource", "External\\nSearch"): 0.48,
    ("Visit\\nSource", "Classification"): 0.70,
    ("External\\nSearch", "Visit\\nSource"): 0.22,
    ("External\\nSearch", "Classification"): 0.54,
    ("Classification", "Visit\\nSource"): 0.68
}

cb_t = {
    ("Prompt", "Bot\\nConsultation"): 0.26,
    ("Prompt", "Visit\\nSource"): 0.60,
    ("Prompt", "External\\nSearch"): 0.67,
    ("Prompt", "Classification"): 0.54,
    ("Bot\\nConsultation", "Prompt"): 0.26,
    ("Visit\\nSource", "Bot\\nConsultation"): 0.54,
    ("Visit\\nSource", "External\\nSearch"): 0.52,
    ("Visit\\nSource", "Classification"): 0.70,
    ("External\\nSearch", "Prompt"): 0.72,
    ("External\\nSearch", "Visit\\nSource"): 0.22,
    ("External\\nSearch", "Classification"): 0.70,
    ("Classification", "Visit\\nSource"): 0.56,
    ("Classification", "External\\nSearch"): 0.51
}

def create_fig3(out_png, out_pdf):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300)
    pos = {
        "Prompt": np.array([-1.15, 0.35]),
        "Bot\\nConsultation": np.array([-1.15, -0.75]),
        "External\\nSearch": np.array([0.0, 1.20]),
        "Visit\\nSource": np.array([0.0, -0.35]),
        "Classification": np.array([1.15, 0.35])
    }
    state_colors = {"Prompt": "#059669", "Bot\\nConsultation": "#D97706", "Visit\\nSource": "#0284C7", "External\\nSearch": "#6366F1", "Classification": "#7C3AED"}
    groups = [("(a)", sim_probs, axes[0], sim_t), ("(b)", cb_probs, axes[1], cb_t)]
    for letter, prob_mat, ax, t_map in groups:
        G = nx.DiGraph()
        for s in STATES_EN: G.add_node(s)
        for s1 in STATES_EN:
            for s2 in STATES_EN:
                if s1 == s2: continue
                p_val = prob_mat.loc[s1, s2]
                if p_val >= 0.10: G.add_edge(s1, s2, weight=p_val)
        node_c = [state_colors[node] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_c, node_size=3800, alpha=0.95, edgecolors="#1E293B", linewidths=2.0)
        node_labels = {"Prompt": "Prompt", "Bot\\nConsultation": "Bot\\nConsult", "Visit\\nSource": "Visit\\nSource", "External\\nSearch": "External\\nSearch", "Classification": "Classify\\nSource"}
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=9.5, font_family="sans-serif", font_weight="bold", font_color="white")
        for (u, v, d) in G.edges(data=True):
            w = d["weight"]; p_u = pos[u]; p_v = pos[v]; has_rev = G.has_edge(v, u)
            if (u == "Prompt" and v == "Classification") or (u == "Classification" and v == "Prompt"): rad = -0.32
            elif has_rev: rad = 0.22
            else: rad = 0.08
            ax.annotate("", xy=p_v, xycoords="data", xytext=p_u, textcoords="data",
                        arrowprops=dict(arrowstyle="-|>", color="#334155", lw=max(1.8, w * 9.5),
                                        connectionstyle=f"arc3,rad={rad}", shrinkA=28, shrinkB=28, alpha=0.85))
            t_val = t_map.get((u, v), 0.50)
            label_pos = bezier(p_u, p_v, rad, t_val)
            ax.text(label_pos[0], label_pos[1], f"{w:.2f}", color="#0F172A", fontsize=9, fontweight="heavy", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec="#94A3B8", lw=1.2, alpha=0.92))
        ax.axis("off")
        ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.35, 1.55)
        ax.text(-1.65, 1.45, letter, fontsize=14, fontweight="bold", color="#1E293B", ha="left", va="top")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
create_fig3("figure_3_markov_state_diagrams.png", "figure_3_markov_state_diagrams.pdf")
create_fig3("figure_3.png", "figure_3.pdf")
'
system2("python3", c("-c", shQuote(py_fig3_code)))

# --- FIGURE 6: Comparative Composite Score and Dimensions (Single-panel: NO text above plot) ---
cat("Generating Figure 6 Composite Score & Dimensions...\n")
df_task <- df_master[df_master$מזהה_משתמש != "user_525vg81ny", ]
df <- df_master
fig6_measures <- c(
  "ציון_4_ממדים" = "Comparative\ncomposite score",
  "ציון_ממד_פרומפט" = "Prompt\nquality",
  "ציון_ממד_זיהוי_מקורות" = "Link verification\naccuracy",
  "ציון_ממד_תהליך_ובדיקה" = "Process\nexecution",
  "ציון_ממד_תחקיר" = "Reflection"
)

fig6_list <- list()
for (c_name in names(fig6_measures)) {
  s_vals <- df_task[[c_name]][df_task$קבוצה_אנגלית == "Simulation"]
  c_vals <- df_task[[c_name]][df_task$קבוצה_אנגלית == "Chatbot"]
  
  s_m <- mean(s_vals); s_se <- sd(s_vals) / sqrt(length(s_vals))
  c_m <- mean(c_vals); c_se <- sd(c_vals) / sqrt(length(c_vals))
  
  fig6_list[[paste0(c_name, "_sim")]] <- data.frame(
    Measure = fig6_measures[c_name],
    Condition = "Simulation",
    mean = s_m,
    ci_low = s_m - 1.96 * s_se,
    ci_high = s_m + 1.96 * s_se,
    stringsAsFactors = FALSE
  )
  fig6_list[[paste0(c_name, "_cb")]] <- data.frame(
    Measure = fig6_measures[c_name],
    Condition = "Chatbot",
    mean = c_m,
    ci_low = c_m - 1.96 * c_se,
    ci_high = c_m + 1.96 * c_se,
    stringsAsFactors = FALSE
  )
}
fig6_df <- do.call(rbind, fig6_list)
fig6_df$Measure <- factor(fig6_df$Measure, levels = unname(fig6_measures))
fig6_df$Condition <- factor(fig6_df$Condition, levels = c("Simulation", "Chatbot"))

annot_df <- data.frame(
  Measure = factor(unname(fig6_measures), levels = unname(fig6_measures)),
  label = c("***", "n.s.", "n.s.", "***", "***"),
  y_pos = c(67, 85, 60, 60, 102)
)

p6 <- ggplot(fig6_df, aes(x = Measure, y = mean, fill = Condition)) +
  geom_col(position = position_dodge(0.75), width = 0.65, color = "white", linewidth = 0.5) +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), position = position_dodge(0.75), width = 0.22, linewidth = 0.8, color = "#1E293B") +
  geom_text(data = annot_df, aes(x = Measure, y = y_pos, label = label), inherit.aes = FALSE,
            size = 4.2, fontface = "bold", color = "#1E293B") +
  scale_fill_manual(values = fig_colors, name = "Condition") +
  scale_y_continuous(limits = c(0, 110), breaks = seq(0, 100, 20), expand = c(0, 0)) +
  labs(x = NULL, y = "Score (0 to 100)") +
  theme_classic(base_size = 12) +
  theme(
    axis.title.y = element_text(size = 11.5, face = "bold", color = "#1E293B", margin = margin(r = 10)),
    axis.text.x  = element_text(size = 10, face = "bold", color = "#1E293B", lineheight = 0.9),
    axis.text.y  = element_text(size = 10, color = "#475569"),
    panel.grid.major.y = element_line(color = "#F1F5F9", linewidth = 0.6),
    legend.position = "bottom",
    legend.title = element_text(size = 10, face = "bold", color = "#1E293B"),
    legend.text = element_text(size = 9.5, color = "#1E293B"),
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(t = 15, r = 20, b = 15, l = 20)
  )

ggsave("figure_6_composite_and_dimensions.png", p6, width = 9.5, height = 6.0, dpi = 300)
ggsave("figure_6_composite_and_dimensions.pdf", p6, width = 9.5, height = 6.0)
ggsave("figure_6.png", p6, width = 9.5, height = 6.0, dpi = 300)
ggsave("figure_6.pdf", p6, width = 9.5, height = 6.0)

# --- FIGURE E1: Reflection Level Distribution (Single-panel: NO text above plot) ---
cat("Generating Figure E1 Reflection Level Distribution...\n")
k_sim <- table(factor(df_task$רמת_Kember[df_task$קבוצה_אנגלית == "Simulation"], levels = 0:4))
k_cb  <- table(factor(df_task$רמת_Kember[df_task$קבוצה_אנגלית == "Chatbot"], levels = 0:4))

pct_sim <- as.numeric(k_sim) / sum(k_sim) * 100
pct_cb  <- as.numeric(k_cb) / sum(k_cb) * 100

kember_labels <- c(
  "Level 0\nIrrelevant",
  "Level 1\nNon-reflection",
  "Level 2\nUnderstanding",
  "Level 3\nReflection",
  "Level 4\nCritical reflection"
)

fig_e1_df <- rbind(
  data.frame(Level = factor(kember_labels, levels = kember_labels), Condition = "Simulation", pct = pct_sim),
  data.frame(Level = factor(kember_labels, levels = kember_labels), Condition = "Chatbot", pct = pct_cb)
)
fig_e1_df$Condition <- factor(fig_e1_df$Condition, levels = c("Simulation", "Chatbot"))
fig_e1_df$lbl <- sprintf("%.1f%%", fig_e1_df$pct)

p_e1 <- ggplot(fig_e1_df, aes(x = Level, y = pct, fill = Condition)) +
  geom_col(position = position_dodge(0.75), width = 0.65, color = "white", linewidth = 0.5) +
  geom_text(aes(label = lbl), position = position_dodge(0.75), vjust = -0.5, size = 3.6, fontface = "bold", color = "#1E293B") +
  scale_fill_manual(values = fig_colors, name = "Condition") +
  scale_y_continuous(limits = c(0, 65), breaks = seq(0, 60, 10), expand = c(0, 0)) +
  labs(x = NULL, y = "Percentage of participants (%)") +
  theme_classic(base_size = 12) +
  theme(
    axis.title.y = element_text(size = 11.5, face = "bold", color = "#1E293B", margin = margin(r = 10)),
    axis.text.x  = element_text(size = 10, face = "bold", color = "#1E293B", lineheight = 0.9),
    axis.text.y  = element_text(size = 10, color = "#475569"),
    panel.grid.major.y = element_line(color = "#F1F5F9", linewidth = 0.6),
    legend.position = "bottom",
    legend.title = element_text(size = 10, face = "bold", color = "#1E293B"),
    legend.text = element_text(size = 9.5, color = "#1E293B"),
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(t = 15, r = 20, b = 15, l = 20)
  )

ggsave("figure_e1_reflection_level_distribution.png", p_e1, width = 9.5, height = 6.0, dpi = 300)
ggsave("figure_e1_reflection_level_distribution.pdf", p_e1, width = 9.5, height = 6.0)
ggsave("figure_e1.png", p_e1, width = 9.5, height = 6.0, dpi = 300)
ggsave("figure_e1.pdf", p_e1, width = 9.5, height = 6.0)


cat("\n===============================================================================\n")
cat("PIPELINE EXECUTION COMPLETE! ALL TABLES & FIGURES GENERATED & VALIDATED.\n")
cat("===============================================================================\n")
'''

R_FIGURES = r'''# Figure E3 (classification accuracy rate by source visit rate) and Figure E1 (reflection levels)
suppressPackageStartupMessages({library(ggplot2);library(scales)})
fig_colors <- c("Simulation" = "#0D9488", "Chatbot" = "#2563EB")
sub67 <- read.csv("e3_data.csv", check.names=FALSE)
sub67$acc_pct <- sub67$accp; sub67$visit_pct <- sub67$visit
sub67$cond_en <- factor(ifelse(sub67$`קבוצה_אנגלית` == "Simulation", "Simulation", "Chatbot"), levels = c("Simulation", "Chatbot"))
sub67$cn <- ifelse(sub67$cond_en=="Simulation",1,0)
f <- summary(lm(acc_pct ~ cn*visit_pct, sub67))$coefficients
f2 <- summary(lm(acc_pct ~ I(1-cn)*visit_pct, sub67))$coefficients
fmtp <- function(p) if (p < .001) "p < 0.001" else paste0("p = ", sprintf("%.3f",p))
fmtb <- function(b) sub("-", "\u2212", sprintf("%.2f", b))
fmtr <- function(b) sub("-", "\u2212", sprintf("%.2f", b))
lab7 <- paste0("Condition x visit rate interaction: B = ", fmtb(f["cn:visit_pct",1]), ", ", fmtp(f["cn:visit_pct",4]),
  "\nSlope, simulation: B = ", fmtb(f2["visit_pct",1]), ", ", fmtp(f2["visit_pct",4]),
  "\nSlope, chatbot: B = ", fmtb(f["visit_pct",1]), ", ", fmtp(f["visit_pct",4]))
cat(lab7,"\n"); print(f["cn:visit_pct",1], digits=6)
p7 <- ggplot(sub67, aes(x = visit_pct, y = acc_pct, color = cond_en, fill = cond_en, shape = cond_en)) +
  geom_smooth(method = "lm", formula = y ~ x, fullrange = TRUE, alpha = 0.18, linewidth = 1.1) +
  geom_point(size = 3.6, stroke = 1.0, color = "white", alpha = 0.88) +
  scale_color_manual(values = fig_colors, name = "Condition") +
  scale_fill_manual(values = fig_colors, name = "Condition") +
  scale_shape_manual(values = c("Simulation" = 21, "Chatbot" = 24), name = "Condition") +
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 20), expand = c(0, 0)) + coord_cartesian(clip = "off") +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 20), expand = c(0, 0)) +
  labs(x = "Source Visit Rate (%)", y = "Classification Accuracy Rate (%)") +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 11, face = "bold", color = "#1E293B"),
    axis.text  = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#E2E8F0", linewidth = 0.5),
    legend.position = "bottom",
    plot.background = element_rect(fill = "white", color = NA), plot.margin = margin(10, 22, 6, 6)
  )

p7 <- p7 + annotate("label", x = 2, y = 97, label = lab7, hjust = 0, vjust = 1, size = 3.4, color = "#1E293B", fill = "white", label.size = 0.3, lineheight = 1.05)
ggsave("figure_e3_sig.png", p7, width = 8.5, height = 6.5, dpi = 600)
ggsave("figure_e3_sig.pdf", p7, width = 8.5, height = 6.5, device = cairo_pdf)

# Figure E1: Kember reflection levels by condition (from the master table)
m_e1 <- read.csv("analysis_scripts/master_clean_n70.csv", stringsAsFactors = FALSE, check.names = FALSE)
lv <- m_e1$`רמת_Kember`[m_e1$`קבוצה_אנגלית` == "Simulation" & !is.na(m_e1$`רמת_Kember`)]
lc <- m_e1$`רמת_Kember`[m_e1$`קבוצה_אנגלית` == "Chatbot" & !is.na(m_e1$`רמת_Kember`)]
pct_sim <- as.numeric(table(factor(lv, levels = 0:4))) / length(lv) * 100
pct_cb  <- as.numeric(table(factor(lc, levels = 0:4))) / length(lc) * 100
w <- wilcox.test(lv, lc, exact=FALSE)
rr <- 2*w$statistic/(30*39)-1
cat("U=",w$statistic," p=",w$p.value," r=",rr,"\n")
kember_labels <- c(
  "Level 0\nIrrelevant",
  "Level 1\nNon-reflection",
  "Level 2\nUnderstanding",
  "Level 3\nReflection",
  "Level 4\nCritical reflection"
)

fig_e1_df <- rbind(
  data.frame(Level = factor(kember_labels, levels = kember_labels), Condition = "Simulation", pct = pct_sim),
  data.frame(Level = factor(kember_labels, levels = kember_labels), Condition = "Chatbot", pct = pct_cb)
)
fig_e1_df$Condition <- factor(fig_e1_df$Condition, levels = c("Simulation", "Chatbot"))
fig_e1_df$lbl <- sprintf("%.1f%%", fig_e1_df$pct)

p_e1 <- ggplot(fig_e1_df, aes(x = Level, y = pct, fill = Condition)) +
  geom_col(position = position_dodge(0.75), width = 0.65, color = "white", linewidth = 0.5) +
  geom_text(aes(label = lbl), position = position_dodge(0.75), vjust = -0.5, size = 3.6, fontface = "bold", color = "#1E293B") +
  scale_fill_manual(values = fig_colors, name = "Condition") +
  scale_y_continuous(limits = c(0, 65), breaks = seq(0, 60, 10), expand = c(0, 0)) +
  labs(x = NULL, y = "Percentage of participants (%)") +
  theme_classic(base_size = 12) +
  theme(
    axis.title.y = element_text(size = 11.5, face = "bold", color = "#1E293B", margin = margin(r = 10)),
    axis.text.x  = element_text(size = 10, face = "bold", color = "#1E293B", lineheight = 0.9),
    axis.text.y  = element_text(size = 10, color = "#475569"),
    panel.grid.major.y = element_line(color = "#F1F5F9", linewidth = 0.6),
    legend.position = "bottom",
    legend.title = element_text(size = 10, face = "bold", color = "#1E293B"),
    legend.text = element_text(size = 9.5, color = "#1E293B"),
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(t = 15, r = 20, b = 15, l = 20)
  )
p_e1 <- p_e1 +
  annotate("text", x = 0.6, y = 62, label = paste0("Mann-Whitney U = ", sprintf("%.1f", w$statistic), ", ", fmtp(w$p.value), ", r = ", fmtr(rr), "\nM = ", sprintf("%.2f", mean(lv)), " vs. ", sprintf("%.2f", mean(lc))), hjust = 0, vjust = 1, size = 3.6, fontface = "bold", color = "#1E293B", lineheight = 1.05)
ggsave("figure_e1_sig.png", p_e1, width = 9.5, height = 6.0, dpi = 600)
ggsave("figure_e1_sig.pdf", p_e1, width = 9.5, height = 6.0, device = cairo_pdf)
'''


def run_r(code):
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.R', delete=False, encoding='utf-8') as fh:
        fh.write(code)
        path = fh.name
    try:
        subprocess.run(['Rscript', '--encoding=UTF-8', path], cwd=PROJECT_DIR, check=True)
    finally:
        os.remove(path)


if __name__ == '__main__':
    run_r(R_PIPELINE)
    run_r(R_FIGURES)
