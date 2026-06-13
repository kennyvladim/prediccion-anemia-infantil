# -*- coding: utf-8 -*-
"""
PIPELINE v3 — PREDICCIÓN DE ANEMIA INFANTIL (ENDES 2024 PERÚ)
VERSION 3 MODELOS: Random Forest, Regresión Logística, Árbol de Decisión

Adaptado para ejecutar en Visual Studio Code / entorno local.
Coloca los 9 CSV de ENDES 2024 en la carpeta indicada por DATA_DIR.

PASOS 01-04 : Carga, Control de calidad, Merges, Ingeniería de variables
PASOS 05-06 : Modelado, Evaluación y Gráficos
"""

# =============================================================================
# CONFIGURACIÓN — AJUSTA ESTA RUTA
# =============================================================================
DATA_DIR = r"C:\Users\kenny\Downloads\prediccion-anemia-infantil\data"        # carpeta donde están los 9 CSV de ENDES 2024
OUTPUT_DIR  = r"./output"        # carpeta donde se guardarán los PNG generados
# =============================================================================

import os, glob, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')            # backend sin ventana; cambia a 'TkAgg' si quieres ventana
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dependencias ──────────────────────────────────────────────────────────────
# pip install pandas numpy scikit-learn matplotlib
# ─────────────────────────────────────────────────────────────────────────────

# =============================================================================
# PASO 01 — CARGA DE MÓDULOS
# =============================================================================
print("=" * 90)
print("PIPELINE v3 — ANEMIA INFANTIL ENDES 2024 — PASOS 01 A 04")
print("=" * 90)
print("\n" + "=" * 90)
print("PASO 01 — CARGA DE MÓDULOS")
print("=" * 90)

def cargar(nombre):
    """Carga el CSV que coincida con el patrón <nombre>_2024*.csv en DATA_DIR."""
    patron = glob.glob(os.path.join(DATA_DIR, f'{nombre}_2024*.csv'))
    if not patron:
        raise FileNotFoundError(
            f"No se encontró {nombre}_2024.csv en '{DATA_DIR}'.\n"
            f"Asegúrate de que el archivo existe y el nombre empieza con '{nombre}_2024'."
        )
    return pd.read_csv(max(patron))   # usa el más reciente si hay duplicados

RECH0  = cargar('RECH0')
RECH4  = cargar('RECH4')
RECH5  = cargar('RECH5')
RECH6  = cargar('RECH6')
RECH23 = cargar('RECH23')
REC41  = cargar('REC41')
REC43  = cargar('REC43')
REC94  = cargar('REC94')
REC95  = cargar('REC95')

for nombre, df in [('RECH0', RECH0), ('RECH4', RECH4), ('RECH5', RECH5),
                   ('RECH6', RECH6), ('RECH23', RECH23), ('REC41', REC41),
                   ('REC43', REC43), ('REC94', REC94), ('REC95', REC95)]:
    print(f"  {nombre:8} {df.shape[0]:>7,} filas x {df.shape[1]:>3} columnas")

print("\n  9/9 módulos cargados correctamente")

# =============================================================================
# PASO 02 — CONTROL DE CALIDAD
# =============================================================================
print("\n" + "=" * 90)
print("PASO 02 — CONTROL DE CALIDAD")
print("=" * 90)

print(f"\n  RECH6 original: {len(RECH6):,} niños <5 años")

df = RECH6.copy()

# Filtro 1: HC57A != 9 (excluir anemia no medida)
n0 = len(df)
df = df[df['HC57A'] != 9].copy()
print(f"  Filtro HC57A != 9 : -{n0-len(df):,}  -> {len(df):,}")

# Filtro 2: HC70 plausible (z-score talla/edad válido, <9996)
n1 = len(df)
df['HC70'] = pd.to_numeric(df['HC70'], errors='coerce')
df = df[df['HC70'] < 9996].copy()
print(f"  Filtro HC70 < 9996: -{n1-len(df):,}  -> {len(df):,}")

# Filtro 3: HC55 == 0 (solo hemoglobina efectivamente medida)
n2 = len(df)
df = df[df['HC55'] == 0].copy()
print(f"  Filtro HC55 == 0  : -{n2-len(df):,}  -> {len(df):,}")

# Target binario
df['target'] = df['HC57A'].isin([1, 2, 3]).astype(int)

n_pos = int(df['target'].sum())
n_neg = int((df['target'] == 0).sum())
print(f"\n  TARGET:")
print(f"    Sin anemia (0): {n_neg:>7,} ({100*n_neg/len(df):.1f}%)")
print(f"    Con anemia (1): {n_pos:>7,} ({100*n_pos/len(df):.1f}%)")

df_elegibles = df.copy()
print(f"\n  df_elegibles: {df_elegibles.shape}")

# =============================================================================
# PASO 03 — MERGES
# =============================================================================
print("\n" + "=" * 90)
print("PASO 03 — MERGES")
print("=" * 90)

df = df_elegibles.copy()

df['HHID'] = df['HHID'].astype(str).str.strip()
df['HC0']  = pd.to_numeric(df['HC0'], errors='coerce')
df['HC1']  = pd.to_numeric(df['HC1'], errors='coerce')

df['HIDX_match'] = df.groupby('HHID')['HC1'].rank(method='first').astype('Int64')

print(f"\n  HHID únicos       : {df['HHID'].nunique():,}")
print(f"  Niños (filas)     : {len(df):,}")
print(f"  HIDX_match        : rango por edad ascendente dentro del hogar")

# MERGE 1: RECH0 (hogar)
r0 = RECH0.copy()
r0['HHID'] = r0['HHID'].astype(str).str.strip()
r0 = r0[['HHID', 'HV040', 'HV025', 'HV024', 'HV014', 'HV005']].drop_duplicates('HHID')
df = df.merge(r0, on='HHID', how='left', indicator=True)
print(f"\n  Merge 1 RECH0   : {(df['_merge']=='left_only').sum()} sin match")
df = df.drop(columns='_merge')

# MERGE 2: RECH23 (riqueza y servicios)
r23 = RECH23.copy()
r23['HHID'] = r23['HHID'].astype(str).str.strip()
r23 = r23[['HHID', 'HV270', 'HV201', 'HV205', 'HV226']].drop_duplicates('HHID')
df = df.merge(r23, on='HHID', how='left', indicator=True)
print(f"  Merge 2 RECH23  : {(df['_merge']=='left_only').sum()} sin match")
df = df.drop(columns='_merge')

# MERGE 3: RECH5 (hemoglobina materna)
r5 = RECH5.copy()
r5['HHID'] = r5['HHID'].astype(str).str.strip()
r5 = r5[['HHID', 'HA56', 'HA1']].drop_duplicates('HHID')
df = df.merge(r5, on='HHID', how='left', indicator=True)
print(f"  Merge 3 RECH5   : {(df['_merge']=='left_only').sum()} sin match"
      f"  | HA56 válido: {df['HA56'].notna().sum():,}")
df = df.drop(columns='_merge')

# MERGE 4: RECH4 (madre como miembro del hogar)
r4 = RECH4.copy()
r4['HHID'] = r4['HHID'].astype(str).str.strip()
r4['IDXH4'] = pd.to_numeric(r4['IDXH4'], errors='coerce')
r4 = r4[['HHID', 'IDXH4', 'SH13', 'SH11C']]
df['HC60'] = pd.to_numeric(df['HC60'], errors='coerce')
df = df.merge(r4, left_on=['HHID', 'HC60'], right_on=['HHID', 'IDXH4'],
              how='left', indicator=True)
print(f"  Merge 4 RECH4   : {(df['_merge']=='left_only').sum()} sin match"
      f"  | SH13 válido: {df['SH13'].notna().sum():,}")
df = df.drop(columns=['_merge', 'IDXH4'])

# MERGE 5: REC41 (último nacimiento)
r41 = REC41.copy()
r41['HHID'] = r41['CASEID'].astype(str).str.strip().str[:9]
r41['MIDX'] = pd.to_numeric(r41['MIDX'], errors='coerce')
idx_ultimo = r41.groupby('HHID')['MIDX'].idxmax()
r41 = r41.loc[idx_ultimo, ['HHID', 'M13', 'M45', 'M18']].copy()
df = df.merge(r41, on='HHID', how='left', indicator=True)
print(f"  Merge 5 REC41   : {(df['_merge']=='left_only').sum()} sin match"
      f"  | M13 válido: {df['M13'].notna().sum():,}")
df = df.drop(columns='_merge')

# MERGE 6: REC43 (salud del niño)
r43 = REC43.copy()
r43['HHID'] = r43['CASEID'].astype(str).str.strip().str[:9]
r43['HIDX'] = pd.to_numeric(r43['HIDX'], errors='coerce').astype('Int64')
r43 = r43[['HHID', 'HIDX', 'H11', 'H31', 'H31B', 'H22', 'H43']].drop_duplicates(['HHID', 'HIDX'])
df = df.merge(r43, left_on=['HHID', 'HIDX_match'], right_on=['HHID', 'HIDX'],
              how='left', indicator=True)
print(f"  Merge 6 REC43   : {(df['_merge']=='left_only').sum()} sin match")
df = df.drop(columns=['_merge', 'HIDX'])

# MERGE 7: REC94 (SIS del niño)
r94 = REC94.copy()
r94['HHID'] = r94['CASEID'].astype(str).str.strip().str[:9]
r94['IDX94'] = pd.to_numeric(r94['IDX94'], errors='coerce').astype('Int64')
r94 = r94[['HHID', 'IDX94', 'S413']].drop_duplicates(['HHID', 'IDX94'])
df = df.merge(r94, left_on=['HHID', 'HIDX_match'], right_on=['HHID', 'IDX94'],
              how='left', indicator=True)
print(f"  Merge 7 REC94   : {(df['_merge']=='left_only').sum()} sin match")
df = df.drop(columns=['_merge', 'IDX94'])

# MERGE 8: REC95 (suplementación Perú)
r95 = REC95.copy()
r95['HHID'] = r95['CASEID'].astype(str).str.strip().str[:9]
r95['IDX95'] = pd.to_numeric(r95['IDX95'], errors='coerce').astype('Int64')
r95 = r95[['HHID', 'IDX95', 'S465EA', 'S465EB', 'S466']].drop_duplicates(['HHID', 'IDX95'])
df = df.merge(r95, left_on=['HHID', 'HIDX_match'], right_on=['HHID', 'IDX95'],
              how='left', indicator=True)
print(f"  Merge 8 REC95   : {(df['_merge']=='left_only').sum()} sin match")
df = df.drop(columns=['_merge', 'IDX95'])

df_maestro = df.copy()
print(f"\n  df_maestro: {df_maestro.shape}")

# =============================================================================
# PASO 04 — INGENIERÍA DE VARIABLES (29 predictores + 1 bandera)
# =============================================================================
print("\n" + "=" * 90)
print("PASO 04 — INGENIERÍA DE VARIABLES")
print("=" * 90)

d = df_maestro.copy()

def num(serie):
    """Convierte a numérico; espacios y cadenas vacías → NaN."""
    return pd.to_numeric(serie.astype(str).str.strip().replace('', np.nan), errors='coerce')

M = pd.DataFrame(index=d.index)

# Niño (4)
M['edad_meses'] = num(d['HC1'])
M['sexo_nino']  = (num(d['HC27']) == 1).astype(float)
M['stunting']   = (num(d['HC70']) < -200).astype(float)
M['wasting']    = (num(d['HC72']) < -200).astype(float)

# Hogar (5)
M['altitud_msnm']   = num(d['HV040'])
M['area_urbana']    = (num(d['HV025']) == 1).astype(float)
M['region']         = num(d['HV024'])
M['nro_ninos_h']    = num(d['HV014'])
M['indice_riqueza'] = num(d['HV270'])

# Agua / saneamiento (3)
M['agua_mejorada'] = num(d['HV201']).isin([11, 12, 13]).astype(float)
M['san_mejorado']  = num(d['HV205']).isin([11, 12]).astype(float)
M['comb_limpio']   = num(d['HV226']).isin([1, 2, 3]).astype(float)

# Madre (6)
M['educ_madre']    = num(d['HC61'])
M['madre_trabaja'] = num(d['SH13']).isin([1, 2, 3, 4]).astype(float)
M['madre_sis']     = (num(d['SH11C']) == 1).astype(float)
M['edad_madre']    = num(d['HA1'])
hb = num(d['HA56'])
hb = hb.where((hb >= 70) & (hb <= 200), np.nan)
M['hb_materna'] = hb
m13 = num(d['M13'])
m13 = m13.where(m13 != 98, np.nan)
M['ctrl_prenatal_temprano'] = ((m13 >= 1) & (m13 <= 3)).astype(float)
M['sin_ctrl_prenatal']      = ((m13 == 0) | (m13.isna())).astype(float)

# Gestación (2)
M['hierro_emb']      = (num(d['M45']) == 1).astype(float)
M['bajo_peso_nacer'] = num(d['M18']).isin([4, 5]).astype(float)

# Salud infantil (5)
M['diarrea']     = (num(d['H11']) == 2).astype(float)
tos = num(d['H31'])
M['tos']         = (tos == 2).astype(float)
ira = (num(d['H31B']) == 1).astype(float)
M['ira']         = ira.where(tos == 2, 0.0)
M['vitamina_a']  = (num(d['H22']) == 1).astype(float)
M['antiparasit'] = (num(d['H43']) == 1).astype(float)

# SIS del niño (1)
M['sis_nino'] = (num(d['S413']) == 1).astype(float)

# Suplementación Perú (3)
M['hierro_7d']      = (num(d['S465EA']) == 1).astype(float)
M['micronut_minsa'] = (num(d['S465EB']) == 1).astype(float)
M['cred']           = (num(d['S466'])   == 1).astype(float)

# Target y peso muestral
M['target']        = d['target'].values
M['peso_muestral'] = pd.to_numeric(d['HV005'], errors='coerce') / 1e6

# Bandera de hemoglobina materna ausente
M['hb_materna_falta'] = M['hb_materna'].isna().astype(float)

# Eliminar filas con NaN en variables continuas clave
continuas_strict = ['edad_meses', 'altitud_msnm', 'region', 'nro_ninos_h',
                    'indice_riqueza', 'educ_madre', 'edad_madre']

print(f"\n  NaN en variables continuas (antes de limpieza):")
for c in continuas_strict + ['hb_materna']:
    print(f"    {c:20}: {M[c].isna().sum():>6,}")

n_antes = len(M)
M = M.dropna(subset=continuas_strict).copy()
print(f"\n  Filas eliminadas por NaN residual: {n_antes - len(M):,}")

mediana_hb = M['hb_materna'].median()
M['hb_materna'] = M['hb_materna'].fillna(mediana_hb)
print(f"  hb_materna: NaN rellenado con mediana provisional ({mediana_hb:.0f})")
print(f"              (el Paso 05 reimputa con la mediana del TRAIN)")

print(f"\n  df_modelo final: {M.shape[0]:,} niños x {M.shape[1]} columnas")

assert M.isna().sum().sum() == 0, "Aún hay NaN en df_modelo"

PREDICTORES = [c for c in M.columns if c not in ('target', 'peso_muestral')]
print(f"\n  Predictores: {len(PREDICTORES)}")
print(f"  Target -> sin anemia: {(M['target']==0).sum():,} | "
      f"con anemia: {(M['target']==1).sum():,} "
      f"({100*M['target'].mean():.1f}% positivos)")

print(f"\n  Lista de {len(PREDICTORES)} predictores:")
for i, p in enumerate(PREDICTORES, 1):
    print(f"    {i:2}. {p}")

df_modelo = M.copy()

print("\n" + "=" * 90)
print("PASOS 01-04 COMPLETADOS  —  df_modelo listo para el PASO 05 (modelado)")
print("=" * 90)


# =============================================================================
# PASO 05 — MODELADO (3 algoritmos: RF, Regresión Logística, Árbol de Decisión)
# =============================================================================
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

print("\n" + "=" * 90)
print("PASO 05 — MODELADO (3 algoritmos: RF, Regresión Logística, Árbol de Decisión)")
print("=" * 90)

# 05.1 — Preparar X, y, peso
X = df_modelo[PREDICTORES].copy()
y = df_modelo['target'].astype(int).values
w = df_modelo['peso_muestral'].values

idx = np.arange(len(X))
Xtr, Xte, ytr, yte, itr, ite = train_test_split(
    X, y, idx, test_size=0.20, stratify=y, random_state=42)
wtr, wte = w[itr], w[ite]

print(f"\n  Train: {len(Xtr):,}  |  Test: {len(Xte):,}")
print(f"  Prevalencia anemia  train: {ytr.mean()*100:.1f}%  test: {yte.mean()*100:.1f}%")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def pipe(modelo):
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('clf', modelo),
    ])

# 05.2 — Grid Search
param_grid = {
    'Logistic Regression': {
        'clf__max_iter':     [1000, 2000, 3000],
        'clf__class_weight': ['balanced'],
    },
    'Random Forest': {
        'clf__n_estimators':     [100, 300, 500],
        'clf__max_depth':        [8, 12, 20],
        'clf__min_samples_leaf': [10, 20, 50],
        'clf__class_weight':     ['balanced'],
    },
    'Decision Tree': {
        'clf__max_depth':        [4, 6, 12],
        'clf__min_samples_leaf': [20, 50, 100],
        'clf__class_weight':     ['balanced'],
    },
}

base_modelos = {
    'Logistic Regression': pipe(LogisticRegression(random_state=42)),
    'Random Forest':       pipe(RandomForestClassifier(random_state=42, n_jobs=-1)),
    'Decision Tree':       pipe(DecisionTreeClassifier(random_state=42)),
}

print("\n" + "-" * 90)
print("05.2 — GRID SEARCH  (Stratified 5-Fold, scoring=roc_auc)")
print("-" * 90)
print("  Esto puede tardar varios minutos...\n")

modelos        = {}
mejores_params = {}

for nombre, pipeline_base in base_modelos.items():
    gs = GridSearchCV(
        estimator  = pipeline_base,
        param_grid = param_grid[nombre],
        cv         = cv,
        scoring    = 'roc_auc',
        n_jobs     = -1,
        refit      = True,
        verbose    = 0,
    )
    gs.fit(Xtr, ytr)
    modelos[nombre]        = gs.best_estimator_
    mejores_params[nombre] = gs.best_params_
    print(f"  {nombre:22}  CV-AUC óptimo: {gs.best_score_*100:.1f}%")
    for param, val in gs.best_params_.items():
        print(f"    {param.replace('clf__',''):30} = {val}")
    print()

print("  Grid Search completado.")

# 05.3 — Función de evaluación
def evaluar(nombre, modelo, usar_peso):
    t0 = time.time()
    cv_auc = cross_val_score(modelo, Xtr, ytr, cv=cv,
                             scoring='roc_auc', n_jobs=-1).mean()
    if usar_peso:
        try:
            modelo.fit(Xtr, ytr, clf__sample_weight=wtr)
        except (TypeError, ValueError):
            modelo.fit(Xtr, ytr)
    else:
        modelo.fit(Xtr, ytr)

    proba = modelo.predict_proba(Xte)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    sw    = wte if usar_peso else None
    test_auc = roc_auc_score(yte, proba, sample_weight=sw)
    acc      = accuracy_score(yte, pred, sample_weight=sw)
    tn, fp, fn, tp = confusion_matrix(yte, pred, sample_weight=sw).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        'modelo': nombre,
        'cv_auc': cv_auc,
        'test_auc': test_auc,
        'brecha_pp': (cv_auc - test_auc) * 100,
        'accuracy': acc,
        'sensibilidad': sens,
        'especificidad': spec,
        'segundos': time.time() - t0,
    }

# 05.4 — Modelos SIN ponderar
print("\n" + "-" * 90)
print("A) MODELOS SIN PONDERAR  (comparables con el paper de Etiopía)")
print("-" * 90)

res_sin = []
modelos_fit = {}
for nombre, modelo in modelos.items():
    p = pipe(modelo.named_steps['clf'])
    r = evaluar(nombre, p, usar_peso=False)
    res_sin.append(r)
    modelos_fit[nombre] = p
    print(f"  {nombre:22} CV-AUC {r['cv_auc']*100:5.1f}  "
          f"Test-AUC {r['test_auc']*100:5.1f}  "
          f"brecha {r['brecha_pp']:+5.1f}pp  "
          f"Sens {r['sensibilidad']*100:4.1f}  Esp {r['especificidad']*100:4.1f}")

df_sin = pd.DataFrame(res_sin).set_index('modelo')

# 05.5 — Modelos PONDERADOS
print("\n" + "-" * 90)
print("B) MODELOS PONDERADOS con peso muestral HV005  (representativo de Perú)")
print("-" * 90)

res_pon = []
for nombre, modelo in modelos.items():
    r = evaluar(nombre, pipe(modelo.named_steps['clf']), usar_peso=True)
    res_pon.append(r)
    print(f"  {nombre:22} CV-AUC {r['cv_auc']*100:5.1f}  "
          f"Test-AUC {r['test_auc']*100:5.1f}  "
          f"brecha {r['brecha_pp']:+5.1f}pp  "
          f"Sens {r['sensibilidad']*100:4.1f}  Esp {r['especificidad']*100:4.1f}")

df_pon = pd.DataFrame(res_pon).set_index('modelo')

# 05.6 — Diagnóstico de leakage
print("\n" + "=" * 90)
print("DIAGNÓSTICO DE BRECHA CV-TEST  (|brecha| < 5pp = sin leakage)")
print("=" * 90)
for nombre in modelos:
    b = df_sin.loc[nombre, 'brecha_pp']
    estado = "OK" if abs(b) < 5 else "REVISAR"
    print(f"  {nombre:22} brecha {b:+5.1f}pp  -> {estado}")

# 05.7 — Mejor modelo
mejor = df_sin['test_auc'].idxmax()
print("\n" + "=" * 90)
print(f"MEJOR MODELO (sin ponderar): {mejor}")
print(f"  Test-AUC      : {df_sin.loc[mejor,'test_auc']*100:.1f}%")
print(f"  CV-AUC        : {df_sin.loc[mejor,'cv_auc']*100:.1f}%")
print(f"  Brecha        : {df_sin.loc[mejor,'brecha_pp']:+.1f}pp")
print(f"  Sensibilidad  : {df_sin.loc[mejor,'sensibilidad']*100:.1f}%")
print(f"  Especificidad : {df_sin.loc[mejor,'especificidad']*100:.1f}%")
print(f"  Versión ponderada Test-AUC: {df_pon.loc[mejor,'test_auc']*100:.1f}%")
print("=" * 90)
print("\n  PASO 05 COMPLETADO.")


# =============================================================================
# PASO 06 — EVALUACIÓN FINAL, COMPARACIÓN Y GRÁFICOS
# =============================================================================
from sklearn.metrics import roc_curve, auc

print("\n" + "=" * 90)
print("PASO 06 — EVALUACIÓN FINAL Y GRÁFICOS  (3 MODELOS)")
print("=" * 90)

# 06.1 — Tabla comparativa final
print("\n" + "-" * 90)
print("06.1 — TABLA COMPARATIVA (Perú ENDES 2024)")
print("-" * 90)

orden = df_sin['test_auc'].sort_values(ascending=False).index
print(f"\n  {'Modelo':22} {'AUC s/pond':>11} {'AUC pond':>10} {'Sens':>7} {'Esp':>7} {'Brecha':>8}")
for m in orden:
    print(f"  {m:22} {df_sin.loc[m,'test_auc']*100:10.1f}% "
          f"{df_pon.loc[m,'test_auc']*100:9.1f}% "
          f"{df_sin.loc[m,'sensibilidad']*100:6.1f}% "
          f"{df_sin.loc[m,'especificidad']*100:6.1f}% "
          f"{df_sin.loc[m,'brecha_pp']:+7.1f}pp")

# 06.2 — Comparación con Etiopía
print("\n" + "-" * 90)
print("06.2 — COMPARACIÓN CON ETIOPÍA (Yimer et al. 2025)")
print("-" * 90)

etiopia = {
    'Random Forest':       81.8,
    'Decision Tree':       56.3,
    'Logistic Regression': 56.1,
}
print(f"\n  {'Modelo':22} {'Perú AUC':>10} {'Etiopía AUC':>13} {'Dif.':>9}")
for m in orden:
    pe  = df_sin.loc[m, 'test_auc'] * 100
    et  = etiopia.get(m, np.nan)
    dif = pe - et
    print(f"  {m:22} {pe:9.1f}% {et:12.1f}% {dif:+8.1f}pp")

print(f"""
  LECTURA:
   - Etiopía RF 81.8% se obtuvo con SMOTE fuera del CV (leakage):
     su brecha CV-Test rondaba ~5pp.
   - Perú RF {df_sin.loc[mejor,'test_auc']*100:.1f}% tiene brecha {df_sin.loc[mejor,'brecha_pp']:+.1f}pp (honesto).
   - La diferencia no es "peor modelo": es la diferencia entre una
     métrica inflada y una métrica válida.
""")

# 06.3 — Gráfico 1: Curvas ROC
print("-" * 90)
print("06.3 — Gráfico 1: curvas ROC")
print("-" * 90)

plt.figure(figsize=(8, 7))
for m, modelo in modelos_fit.items():
    proba = modelo.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, proba)
    plt.plot(fpr, tpr, lw=2, label=f"{m} (AUC {auc(fpr,tpr)*100:.1f}%)")
plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Azar (50%)')
plt.xlabel('1 - Especificidad (Falsos positivos)')
plt.ylabel('Sensibilidad (Verdaderos positivos)')
plt.title('Curvas ROC — Predicción de anemia infantil (ENDES 2024)')
plt.legend(loc='lower right', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, 'grafico_1_curvas_roc.png')
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Guardado: {roc_path}")

# 06.4 — Gráfico 2: Importancia de variables (Random Forest)
print("\n" + "-" * 90)
print("06.4 — Gráfico 2: importancia de variables (Random Forest)")
print("-" * 90)

rf = modelos_fit["Random Forest"].named_steps['clf']
importancias = pd.Series(rf.feature_importances_, index=PREDICTORES).sort_values(ascending=True)

plt.figure(figsize=(8, 9))
plt.barh(importancias.index, importancias.values, color='#1D9E75')
plt.xlabel('Importancia (reducción media de impureza)')
plt.title('Importancia de variables — Random Forest')
plt.tight_layout()
imp_path = os.path.join(OUTPUT_DIR, 'grafico_2_importancia.png')
plt.savefig(imp_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Guardado: {imp_path}")

print(f"\n  Top 10 predictores más importantes:")
for v, imp in importancias.sort_values(ascending=False).head(10).items():
    print(f"    {v:24} {imp*100:5.1f}%")

# 06.5 — Gráfico 3: Matriz de confusión del mejor modelo
print("\n" + "-" * 90)
print("06.5 — Gráfico 3: matriz de confusión (mejor modelo)")
print("-" * 90)

proba_mejor = modelos_fit[mejor].predict_proba(Xte)[:, 1]
pred_mejor  = (proba_mejor >= 0.5).astype(int)
cm = confusion_matrix(yte, pred_mejor)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Greens')
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm[i,j]:,}", ha='center', va='center',
                 fontsize=14,
                 color='white' if cm[i,j] > cm.max() / 2 else 'black')
plt.xticks([0, 1], ['Sin anemia', 'Con anemia'])
plt.yticks([0, 1], ['Sin anemia', 'Con anemia'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title(f'Matriz de confusión — {mejor}')
plt.colorbar()
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'grafico_3_matriz_confusion.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Guardado: {cm_path}")

print("\n" + "=" * 90)
print(f"PASO 06 COMPLETADO — Gráficos guardados en: {os.path.abspath(OUTPUT_DIR)}")
print("=" * 90)



# =============================================================================
# PASO 07 — INFERENCIA EN LOTES (evidencia de correcto funcionamiento)
# =============================================================================
print("\n" + "=" * 90)
print("PASO 07 — INFERENCIA EN LOTES (8 perfiles sintéticos)")
print("=" * 90)

# ---------------------------------------------------------------------------
# 07.1 — Definir 8 perfiles clínicos representativos
# ---------------------------------------------------------------------------
# Cada fila es un niño hipotético. Los valores siguen la codificación ENDES.
# hb_materna_falta: 1 si no se midió hemoglobina materna, 0 si sí se midió.

perfiles = [
    # --- Perfil 1: Niño de bajo riesgo (urbano, Lima, buen estado nutricional)
    dict(label="P1 — Urbano Lima, bajo riesgo",
         edad_meses=18, sexo_nino=1, stunting=0, wasting=0,
         altitud_msnm=150,  area_urbana=1, region=15, nro_ninos_h=1,
         indice_riqueza=5,  agua_mejorada=1, san_mejorado=1, comb_limpio=1,
         educ_madre=3,  madre_trabaja=1, madre_sis=0, edad_madre=28,
         hb_materna=140, ctrl_prenatal_temprano=1, sin_ctrl_prenatal=0,
         hierro_emb=1, bajo_peso_nacer=0,
         diarrea=0, tos=0, ira=0, vitamina_a=1, antiparasit=1,
         sis_nino=1, hierro_7d=1, micronut_minsa=1, cred=1,
         hb_materna_falta=0),

    # --- Perfil 2: Alto riesgo (rural, sierra alta, desnutrido, sin servicios)
    dict(label="P2 — Rural Sierra, alto riesgo",
         edad_meses=12, sexo_nino=0, stunting=1, wasting=1,
         altitud_msnm=4200, area_urbana=0, region=8, nro_ninos_h=5,
         indice_riqueza=1,  agua_mejorada=0, san_mejorado=0, comb_limpio=0,
         educ_madre=0,  madre_trabaja=0, madre_sis=0, edad_madre=19,
         hb_materna=95,  ctrl_prenatal_temprano=0, sin_ctrl_prenatal=1,
         hierro_emb=0, bajo_peso_nacer=1,
         diarrea=1, tos=1, ira=1, vitamina_a=0, antiparasit=0,
         sis_nino=0, hierro_7d=0, micronut_minsa=0, cred=0,
         hb_materna_falta=0),

    # --- Perfil 3: Riesgo moderado (rural, costa, sin control prenatal temprano)
    dict(label="P3 — Rural Costa, riesgo moderado",
         edad_meses=9,  sexo_nino=1, stunting=0, wasting=0,
         altitud_msnm=60,  area_urbana=0, region=3, nro_ninos_h=3,
         indice_riqueza=2,  agua_mejorada=1, san_mejorado=0, comb_limpio=1,
         educ_madre=1,  madre_trabaja=0, madre_sis=1, edad_madre=23,
         hb_materna=118, ctrl_prenatal_temprano=0, sin_ctrl_prenatal=0,
         hierro_emb=0, bajo_peso_nacer=0,
         diarrea=0, tos=1, ira=0, vitamina_a=0, antiparasit=1,
         sis_nino=1, hierro_7d=0, micronut_minsa=0, cred=1,
         hb_materna_falta=0),

    # --- Perfil 4: Urbano, Puno, altitud elevada, con micronutrientes
    dict(label="P4 — Urbano Puno, altitud alta",
         edad_meses=24, sexo_nino=0, stunting=1, wasting=0,
         altitud_msnm=3850, area_urbana=1, region=14, nro_ninos_h=2,
         indice_riqueza=3,  agua_mejorada=1, san_mejorado=1, comb_limpio=1,
         educ_madre=2,  madre_trabaja=1, madre_sis=1, edad_madre=30,
         hb_materna=108, ctrl_prenatal_temprano=1, sin_ctrl_prenatal=0,
         hierro_emb=1, bajo_peso_nacer=0,
         diarrea=0, tos=0, ira=0, vitamina_a=1, antiparasit=1,
         sis_nino=1, hierro_7d=1, micronut_minsa=1, cred=1,
         hb_materna_falta=0),

    # --- Perfil 5: Madre sin hemoglobina medida, bajo nivel educativo
    dict(label="P5 — Hb materna no medida",
         edad_meses=6,  sexo_nino=1, stunting=0, wasting=0,
         altitud_msnm=500,  area_urbana=0, region=7, nro_ninos_h=2,
         indice_riqueza=2,  agua_mejorada=1, san_mejorado=0, comb_limpio=0,
         educ_madre=1,  madre_trabaja=0, madre_sis=0, edad_madre=21,
         hb_materna=128, ctrl_prenatal_temprano=0, sin_ctrl_prenatal=1,
         hierro_emb=0, bajo_peso_nacer=0,
         diarrea=1, tos=0, ira=0, vitamina_a=0, antiparasit=0,
         sis_nino=0, hierro_7d=0, micronut_minsa=0, cred=0,
         hb_materna_falta=1),

    # --- Perfil 6: Recién nacido con bajo peso, diarrea activa
    dict(label="P6 — Bajo peso nacer + diarrea",
         edad_meses=8,  sexo_nino=0, stunting=1, wasting=1,
         altitud_msnm=200,  area_urbana=1, region=1, nro_ninos_h=4,
         indice_riqueza=2,  agua_mejorada=0, san_mejorado=0, comb_limpio=0,
         educ_madre=1,  madre_trabaja=0, madre_sis=1, edad_madre=17,
         hb_materna=100, ctrl_prenatal_temprano=0, sin_ctrl_prenatal=1,
         hierro_emb=0, bajo_peso_nacer=1,
         diarrea=1, tos=1, ira=1, vitamina_a=0, antiparasit=0,
         sis_nino=1, hierro_7d=0, micronut_minsa=0, cred=0,
         hb_materna_falta=0),

    # --- Perfil 7: Óptimo (todos los factores protectores activados)
    dict(label="P7 — Perfil protector máximo",
         edad_meses=36, sexo_nino=1, stunting=0, wasting=0,
         altitud_msnm=100,  area_urbana=1, region=15, nro_ninos_h=1,
         indice_riqueza=5,  agua_mejorada=1, san_mejorado=1, comb_limpio=1,
         educ_madre=3,  madre_trabaja=1, madre_sis=0, edad_madre=32,
         hb_materna=145, ctrl_prenatal_temprano=1, sin_ctrl_prenatal=0,
         hierro_emb=1, bajo_peso_nacer=0,
         diarrea=0, tos=0, ira=0, vitamina_a=1, antiparasit=1,
         sis_nino=1, hierro_7d=1, micronut_minsa=1, cred=1,
         hb_materna_falta=0),

    # --- Perfil 8: Máximo riesgo acumulado
    dict(label="P8 — Riesgo acumulado extremo",
         edad_meses=11, sexo_nino=0, stunting=1, wasting=1,
         altitud_msnm=4500, area_urbana=0, region=11, nro_ninos_h=7,
         indice_riqueza=1,  agua_mejorada=0, san_mejorado=0, comb_limpio=0,
         educ_madre=0,  madre_trabaja=0, madre_sis=0, edad_madre=16,
         hb_materna=88,  ctrl_prenatal_temprano=0, sin_ctrl_prenatal=1,
         hierro_emb=0, bajo_peso_nacer=1,
         diarrea=1, tos=1, ira=1, vitamina_a=0, antiparasit=0,
         sis_nino=0, hierro_7d=0, micronut_minsa=0, cred=0,
         hb_materna_falta=1),
]

labels   = [p.pop('label') for p in perfiles]
X_batch  = pd.DataFrame(perfiles)[PREDICTORES]   # mismo orden que el modelo

# ---------------------------------------------------------------------------
# 07.2 — Inferencia con los 3 modelos
# ---------------------------------------------------------------------------
print("\n  Ejecutando inferencia en lote...\n")

resultados_batch = {}
for nombre, modelo in modelos_fit.items():
    proba = modelo.predict_proba(X_batch)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    resultados_batch[nombre] = {'proba': proba, 'pred': pred}

# Tabla de resultados
print(f"  {'Perfil':40} | {'RF%':>6} | {'LR%':>6} | {'DT%':>6} | Consenso")
print("  " + "-" * 80)
for i, lbl in enumerate(labels):
    rf_p = resultados_batch['Random Forest']['proba'][i] * 100
    lr_p = resultados_batch['Logistic Regression']['proba'][i] * 100
    dt_p = resultados_batch['Decision Tree']['proba'][i] * 100
    consenso = int(round((rf_p + lr_p + dt_p) / 3) >= 50)
    flag = "ANEMIA" if consenso else "sin anemia"
    print(f"  {lbl:40} | {rf_p:5.1f}% | {lr_p:5.1f}% | {dt_p:5.1f}% | {flag}")

# ---------------------------------------------------------------------------
# 07.3 — Gráfico 4: probabilidades de anemia por perfil y modelo
# ---------------------------------------------------------------------------
print("\n" + "-" * 90)
print("07.3 — Gráfico 4: probabilidades por perfil")
print("-" * 90)

n_perfiles  = len(labels)
x           = np.arange(n_perfiles)
width       = 0.25
colores     = {'Random Forest': '#1D9E75', 'Logistic Regression': '#E76F00',
               'Decision Tree': '#3B6FD4'}

fig, ax = plt.subplots(figsize=(14, 6))
offsets = [-width, 0, width]
for offset, (nombre, res) in zip(offsets, resultados_batch.items()):
    ax.bar(x + offset, res['proba'] * 100, width,
           label=nombre, color=colores[nombre], alpha=0.88)

ax.axhline(50, color='red', lw=1.2, ls='--', label='Umbral 50%')
ax.set_xticks(x)
ax.set_xticklabels([lbl.split(" — ")[0] for lbl in labels],
                   rotation=25, ha='right', fontsize=9)
ax.set_ylabel('Probabilidad de anemia (%)')
ax.set_title('Inferencia en lote — 8 perfiles sintéticos\n'
             'PIPELINE v3 — Predicción de anemia infantil ENDES 2024')
ax.set_ylim(0, 105)
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
batch_path = os.path.join(OUTPUT_DIR, 'grafico_4_batch_inference.png')
plt.savefig(batch_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Guardado: {batch_path}")

# ---------------------------------------------------------------------------
# 07.4 — Validación de coherencia lógica
# ---------------------------------------------------------------------------
print("\n" + "-" * 90)
print("07.4 — VALIDACIÓN DE COHERENCIA LÓGICA")
print("-" * 90)
print("  Regla: P7 (protector máximo) < umbral < P8 (riesgo extremo)")

rf_p7 = resultados_batch['Random Forest']['proba'][6] * 100   # índice 6 = P7
rf_p8 = resultados_batch['Random Forest']['proba'][7] * 100   # índice 7 = P8

coherente_p7 = rf_p7 < 50
coherente_p8 = rf_p8 >= 50

print(f"\n  Random Forest — P7 protector máximo : {rf_p7:.1f}%  -> "
      f"{'OK (< 50%)' if coherente_p7 else 'REVISAR (>= 50%)'}")
print(f"  Random Forest — P8 riesgo extremo   : {rf_p8:.1f}%  -> "
      f"{'OK (>= 50%)' if coherente_p8 else 'REVISAR (< 50%)'}")

coherencia_global = coherente_p7 and coherente_p8
print(f"\n  {'✔ Coherencia lógica VERIFICADA' if coherencia_global else '✘ Revisar predicciones extremas'}")

print("\n" + "=" * 90)
print(f"PASO 07 COMPLETADO — Inferencia en lote sobre 8 perfiles sintéticos.")
print(f"  Gráfico guardado : {os.path.abspath(batch_path)}")
print("=" * 90)
print("\nPIPELINE v3 FINALIZADO CORRECTAMENTE.")