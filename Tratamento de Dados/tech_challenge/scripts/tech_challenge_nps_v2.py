"""
================================================================================
TECH CHALLENGE FASE 1 — CASE NPS PREDITIVO (VERSÃO 2)
================================================================================
Versão 2: Tratamento de NPS=0 como outliers
--------------------------------------------------------------------------------
Premissa: nesta versão, os 158 clientes com nps_score = 0 são tratados como
outliers e removidos da base antes da análise. O objetivo é avaliar se:

  1. As correlações entre variáveis e NPS mudam significativamente
  2. Os modelos preditivos têm performance diferente
  3. O storytelling para o negócio se altera

ATENÇÃO METODOLÓGICA
--------------------
Na v1 mostramos que os clientes com NPS=0 têm perfil de detratores genuínos
(mais atrasos, mais reclamações, mais contatos com SAC), e portanto removê-los
introduz um viés: estamos "limpando" parte do problema real.

Este script existe para QUANTIFICAR esse viés e gerar uma comparação honesta
com a v1 (que mantém os NPS=0).

Estrutura:
  data/desafio_nps_fase_1.csv          (base original)
  data/desafio_nps_processado.csv      (base v1 com features)
  data/desafio_nps_processado_v2.csv   (base v2, sem NPS=0)
  figures_v2/                          (gráficos da v2)
  models_v2/                           (modelos da v2)
  reports/comparativo_v1_v2.docx       (comparativo)
================================================================================
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 120

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'desafio_nps_fase_1.csv'
FIG_DIR = BASE_DIR / 'figures_v2'
MODEL_DIR = BASE_DIR / 'models_v2'
REPORT_DIR = BASE_DIR / 'reports'
for d in [FIG_DIR, MODEL_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Features usadas em ambas as versões — mantemos iguais para o comparativo ser justo
FEATURES_NUM = ['customer_age', 'customer_tenure_months', 'order_value',
                'items_quantity', 'discount_value', 'payment_installments',
                'delivery_time_days', 'delivery_delay_days', 'freight_value',
                'delivery_attempts', 'customer_service_contacts',
                'resolution_time_days', 'complaints_count',
                'repeat_purchase_30d', 'csat_internal_score',
                'severidade_problemas', 'desconto_pct']
FEATURES_CAT = ['customer_region']

COLS_NUM_CORR = ['nps_score', 'customer_age', 'customer_tenure_months',
                 'order_value', 'items_quantity', 'discount_value',
                 'payment_installments', 'delivery_time_days',
                 'delivery_delay_days', 'freight_value', 'delivery_attempts',
                 'customer_service_contacts', 'resolution_time_days',
                 'complaints_count', 'repeat_purchase_30d',
                 'csat_internal_score']


def secao(titulo: str) -> None:
    print('\n' + '=' * 80)
    print(titulo)
    print('=' * 80)


# ============================================================================
# PREPARAÇÃO DOS DADOS — IDÊNTICA À V1, mas com remoção dos NPS=0
# ============================================================================
def classificar_nps(score: float) -> str:
    score_int = int(np.floor(score))
    if score_int >= 9:
        return 'Promotor'
    if score_int >= 7:
        return 'Neutro'
    return 'Detrator'


def preparar_dados(remover_nps_zero: bool = True):
    """Prepara dados igualzinho à v1, mas com opção de remover NPS=0."""
    df = pd.read_csv(DATA_PATH)

    n_zero = (df['nps_score'] == 0).sum()
    if remover_nps_zero:
        df = df[df['nps_score'] > 0].copy().reset_index(drop=True)
        print(f'Removidos {n_zero} registros com NPS=0 (tratados como outliers)')
        print(f'Base resultante: {len(df)} linhas')
    else:
        print(f'Base mantida com {n_zero} registros de NPS=0')

    # Mesma feature engineering da v1
    df['classificacao_nps'] = df['nps_score'].apply(classificar_nps)
    df['eh_detrator'] = (df['classificacao_nps'] == 'Detrator').astype(int)
    df['teve_atraso'] = (df['delivery_delay_days'] > 0).astype(int)
    df['teve_reclamacao'] = (df['complaints_count'] > 0).astype(int)
    df['contatou_sac'] = (df['customer_service_contacts'] > 0).astype(int)
    df['severidade_problemas'] = (df['teve_atraso'] + df['teve_reclamacao']
                                   + df['contatou_sac'])
    df['desconto_pct'] = df['discount_value'] / df['order_value'].replace(0, np.nan)
    df['desconto_pct'] = df['desconto_pct'].fillna(0)

    return df


# ============================================================================
# EDA + CORRELAÇÕES — V2
# ============================================================================
def analise_descritiva(df: pd.DataFrame) -> dict:
    secao('ANÁLISE DESCRITIVA — V2 (sem NPS=0)')

    dist = df['classificacao_nps'].value_counts(normalize=True).reindex(
        ['Detrator', 'Neutro', 'Promotor']
    ) * 100
    nps_oficial = dist['Promotor'] - dist['Detrator']

    print(f'NPS médio   : {df["nps_score"].mean():.2f}')
    print(f'NPS mediana : {df["nps_score"].median():.2f}')
    print('Distribuição:')
    for cat, pct in dist.items():
        print(f'  {cat:10s}: {pct:5.1f}%')
    print(f'NPS oficial : {nps_oficial:.1f}')

    # Correlações com NPS
    corr = df[COLS_NUM_CORR].corr()['nps_score'].drop('nps_score').sort_values()
    print('\nCorrelações com NPS:')
    for var, c in corr.items():
        print(f'  {var:30s}: {c:+.3f}')

    return {
        'n_linhas': len(df),
        'nps_medio': df['nps_score'].mean(),
        'nps_oficial': nps_oficial,
        'dist': dist,
        'correlacoes': corr,
    }


def gerar_figuras_v2(df: pd.DataFrame) -> None:
    """Mesmas figuras-chave da v1, agora para a base v2."""
    secao('GERANDO FIGURAS V2')

    # 1. Distribuição NPS
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['nps_score'], bins=30, color='steelblue', edgecolor='white')
    ax.axvline(df['nps_score'].mean(), color='red', linestyle='--',
               label=f'Média: {df["nps_score"].mean():.2f}')
    ax.set_title('Distribuição do NPS Score (V2 — sem NPS=0)')
    ax.set_xlabel('NPS Score')
    ax.set_ylabel('Frequência')
    ax.legend()
    plt.savefig(FIG_DIR / '01_distribuicao_nps_v2.png'); plt.close()
    print('  ✓ 01_distribuicao_nps_v2.png')

    # 2. Pizza
    dist = df['classificacao_nps'].value_counts().reindex(
        ['Detrator', 'Neutro', 'Promotor']
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(dist.values, labels=dist.index, autopct='%1.1f%%',
           colors=['#d62728', '#ff7f0e', '#2ca02c'], startangle=90,
           textprops={'fontsize': 12})
    ax.set_title('Composição da base V2 (sem NPS=0)')
    plt.savefig(FIG_DIR / '02_classificacao_nps_v2.png'); plt.close()
    print('  ✓ 02_classificacao_nps_v2.png')

    # 3. Correlação com NPS
    corr = df[COLS_NUM_CORR].corr()['nps_score'].drop('nps_score').sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    cores = ['#d62728' if v < 0 else '#2ca02c' for v in corr.values]
    bars = ax.barh(corr.index, corr.values, color=cores)
    ax.bar_label(bars, fmt='%.2f', padding=3)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title('Correlação de cada variável com o NPS (V2)')
    ax.set_xlabel('Correlação de Pearson')
    plt.savefig(FIG_DIR / '03_correlacao_nps_v2.png'); plt.close()
    print('  ✓ 03_correlacao_nps_v2.png')


# ============================================================================
# MODELAGEM — V2
# ============================================================================
def construir_pipeline(modelo):
    pre = ColumnTransformer([
        ('num', StandardScaler(), FEATURES_NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore'), FEATURES_CAT),
    ])
    return Pipeline([('pre', pre), ('modelo', modelo)])


def treinar_regressao(df: pd.DataFrame) -> dict:
    secao('MODELO DE REGRESSÃO — V2')

    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df['nps_score']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    resultados = {}
    for nome, modelo in [
        ('Regressão Linear', LinearRegression()),
        ('Random Forest', RandomForestRegressor(n_estimators=200, max_depth=10,
                                                random_state=RANDOM_STATE,
                                                n_jobs=-1))
    ]:
        pipe = construir_pipeline(modelo)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        resultados[nome] = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'pipe': pipe,
        }
        print(f'\n[{nome}]  MAE={resultados[nome]["mae"]:.3f}  '
              f'RMSE={resultados[nome]["rmse"]:.3f}  '
              f'R²={resultados[nome]["r2"]:.3f}')

    melhor = max(resultados, key=lambda k: resultados[k]['r2'])
    joblib.dump(resultados[melhor]['pipe'],
                MODEL_DIR / 'modelo_regressao_nps_v2.pkl')
    print(f'\n>>> Melhor: {melhor}')
    return {'resultados': resultados, 'melhor': melhor}


def treinar_classificacao(df: pd.DataFrame) -> dict:
    secao('MODELO DE CLASSIFICAÇÃO — V2 (Detrator vs Não)')

    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df['eh_detrator']

    if y.nunique() < 2:
        print('AVISO: a base v2 só tem uma classe — pulando classificação.')
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f'Taxa de detratores (treino): {y_train.mean()*100:.1f}%')

    resultados = {}
    for nome, modelo in [
        ('Regressão Logística', LogisticRegression(max_iter=1000,
                                                    random_state=RANDOM_STATE)),
        ('Random Forest', RandomForestClassifier(n_estimators=200, max_depth=10,
                                                  random_state=RANDOM_STATE,
                                                  n_jobs=-1))
    ]:
        pipe = construir_pipeline(modelo)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        resultados[nome] = {
            'acc': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba),
            'pipe': pipe,
        }
        r = resultados[nome]
        print(f'\n[{nome}]')
        print(f'  Acc={r["acc"]:.3f}  P={r["precision"]:.3f}  R={r["recall"]:.3f}  '
              f'F1={r["f1"]:.3f}  AUC={r["auc"]:.3f}')

    melhor = max(resultados, key=lambda k: resultados[k]['f1'])
    joblib.dump(resultados[melhor]['pipe'],
                MODEL_DIR / 'modelo_classificacao_detrator_v2.pkl')
    print(f'\n>>> Melhor: {melhor}')
    return {'resultados': resultados, 'melhor': melhor}


# ============================================================================
# COMPARATIVO V1 vs V2
# ============================================================================
def calcular_v1():
    """Roda v1 inteira — base completa, sem remoção — para gerar números
    comparáveis sem depender da execução anterior."""
    secao('CALCULANDO V1 (BASE COMPLETA) PARA COMPARATIVO')

    df_v1 = preparar_dados(remover_nps_zero=False)
    desc_v1 = analise_descritiva(df_v1)
    reg_v1 = treinar_regressao(df_v1)
    cls_v1 = treinar_classificacao(df_v1)
    return {'df': df_v1, 'desc': desc_v1, 'reg': reg_v1, 'cls': cls_v1}


def gerar_comparativo(v1: dict, v2: dict) -> None:
    secao('COMPARATIVO V1 vs V2')

    # Tabela 1: visão geral
    print('\n>>> Visão Geral')
    print(f'{"Métrica":<35} {"V1 (com NPS=0)":>15} {"V2 (sem NPS=0)":>18} {"Δ":>10}')
    print('-' * 80)
    metricas = [
        ('Linhas na base', v1['desc']['n_linhas'], v2['desc']['n_linhas'], 'd'),
        ('NPS médio', v1['desc']['nps_medio'], v2['desc']['nps_medio'], '.2f'),
        ('NPS oficial (% prom - % det)', v1['desc']['nps_oficial'],
         v2['desc']['nps_oficial'], '.1f'),
        ('% Detratores', v1['desc']['dist']['Detrator'],
         v2['desc']['dist']['Detrator'], '.1f'),
        ('% Neutros', v1['desc']['dist']['Neutro'],
         v2['desc']['dist']['Neutro'], '.1f'),
        ('% Promotores', v1['desc']['dist']['Promotor'],
         v2['desc']['dist']['Promotor'], '.1f'),
    ]
    for nome, val1, val2, fmt in metricas:
        delta = val2 - val1
        if fmt == 'd':
            print(f'{nome:<35} {val1:>15,d} {val2:>18,d} {delta:>+10,d}')
        else:
            print(f'{nome:<35} {val1:>15.2f} {val2:>18.2f} {delta:>+10.2f}')

    # Tabela 2: Modelo de regressão
    print('\n>>> Modelo de Regressão (melhor de cada)')
    print(f'{"Métrica":<25} {"V1":>15} {"V2":>15} {"Δ":>12}')
    print('-' * 70)
    r1 = v1['reg']['resultados'][v1['reg']['melhor']]
    r2 = v2['reg']['resultados'][v2['reg']['melhor']]
    for met in ['mae', 'rmse', 'r2']:
        delta = r2[met] - r1[met]
        sinal = ''
        # Para R² maior é melhor; para MAE/RMSE menor é melhor
        if met in ('mae', 'rmse'):
            sinal = ' (melhor)' if delta < 0 else ' (pior)' if delta > 0 else ''
        else:
            sinal = ' (melhor)' if delta > 0 else ' (pior)' if delta < 0 else ''
        print(f'{met.upper():<25} {r1[met]:>15.3f} {r2[met]:>15.3f} {delta:>+12.3f}{sinal}')
    print(f'Melhor modelo V1: {v1["reg"]["melhor"]}')
    print(f'Melhor modelo V2: {v2["reg"]["melhor"]}')

    # Tabela 3: Modelo de classificação
    if v2.get('cls') and v1.get('cls'):
        print('\n>>> Modelo de Classificação (Detrator vs Não)')
        print(f'{"Métrica":<25} {"V1":>15} {"V2":>15} {"Δ":>12}')
        print('-' * 70)
        c1 = v1['cls']['resultados'][v1['cls']['melhor']]
        c2 = v2['cls']['resultados'][v2['cls']['melhor']]
        for met in ['acc', 'precision', 'recall', 'f1', 'auc']:
            delta = c2[met] - c1[met]
            sinal = ' (melhor)' if delta > 0 else ' (pior)' if delta < 0 else ''
            print(f'{met.upper():<25} {c1[met]:>15.3f} {c2[met]:>15.3f} '
                  f'{delta:>+12.3f}{sinal}')
        print(f'Melhor modelo V1: {v1["cls"]["melhor"]}')
        print(f'Melhor modelo V2: {v2["cls"]["melhor"]}')

    # Tabela 4: Mudança nas correlações
    print('\n>>> Mudança nas correlações com NPS')
    print(f'{"Variável":<32} {"V1":>10} {"V2":>10} {"Δ":>10} {"Mudança":>15}')
    print('-' * 85)
    corr_v1 = v1['desc']['correlacoes']
    corr_v2 = v2['desc']['correlacoes']
    diffs = (corr_v2 - corr_v1).abs().sort_values(ascending=False)
    for var in diffs.index:
        c1, c2 = corr_v1[var], corr_v2[var]
        delta = c2 - c1
        # Detecta inversão de sinal (importante!)
        inverteu = (c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)
        comentario = '⚠ inverteu sinal' if inverteu else (
            'enfraqueceu' if abs(c2) < abs(c1) else 'fortaleceu'
        )
        print(f'{var:<32} {c1:>+10.3f} {c2:>+10.3f} {delta:>+10.3f} '
              f'{comentario:>15}')

    # Geração de figuras comparativas
    gerar_figuras_comparativas(v1, v2)


def gerar_figuras_comparativas(v1: dict, v2: dict) -> None:
    """Gráficos lado a lado V1 vs V2."""

    # Figura comparativa 1: Distribuição da classificação
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cores = ['#d62728', '#ff7f0e', '#2ca02c']

    for ax, dados, titulo in [
        (axes[0], v1['desc']['dist'], f'V1 — Base completa\n(n={v1["desc"]["n_linhas"]:,})'),
        (axes[1], v2['desc']['dist'], f'V2 — Sem NPS=0\n(n={v2["desc"]["n_linhas"]:,})'),
    ]:
        ax.pie(dados.values, labels=dados.index, autopct='%1.1f%%',
               colors=cores, startangle=90, textprops={'fontsize': 11})
        ax.set_title(titulo, fontsize=13, fontweight='bold')

    plt.suptitle('Distribuição de Detratores/Neutros/Promotores', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / '04_comparativo_distribuicao.png'); plt.close()
    print('  ✓ 04_comparativo_distribuicao.png')

    # Figura comparativa 2: Correlações
    fig, ax = plt.subplots(figsize=(11, 8))
    corr_v1 = v1['desc']['correlacoes']
    corr_v2 = v2['desc']['correlacoes']

    df_corr = pd.DataFrame({'V1 (com NPS=0)': corr_v1,
                            'V2 (sem NPS=0)': corr_v2})
    df_corr = df_corr.reindex(corr_v1.sort_values().index)

    x = np.arange(len(df_corr))
    largura = 0.4
    ax.barh(x - largura/2, df_corr['V1 (com NPS=0)'], largura,
            label='V1 (com NPS=0)', color='steelblue')
    ax.barh(x + largura/2, df_corr['V2 (sem NPS=0)'], largura,
            label='V2 (sem NPS=0)', color='coral')
    ax.set_yticks(x)
    ax.set_yticklabels(df_corr.index)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Correlação de Pearson com NPS')
    ax.set_title('Comparativo de correlações com NPS — V1 vs V2')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / '05_comparativo_correlacoes.png'); plt.close()
    print('  ✓ 05_comparativo_correlacoes.png')

    # Figura comparativa 3: Métricas dos modelos
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Regressão
    ax = axes[0]
    r1 = v1['reg']['resultados'][v1['reg']['melhor']]
    r2 = v2['reg']['resultados'][v2['reg']['melhor']]
    metricas_reg = ['mae', 'rmse', 'r2']
    x = np.arange(len(metricas_reg))
    largura = 0.35
    ax.bar(x - largura/2, [r1[m] for m in metricas_reg], largura,
           label=f'V1 ({v1["reg"]["melhor"]})', color='steelblue')
    ax.bar(x + largura/2, [r2[m] for m in metricas_reg], largura,
           label=f'V2 ({v2["reg"]["melhor"]})', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(['MAE', 'RMSE', 'R²'])
    ax.set_title('Regressão (melhor modelo de cada versão)')
    ax.legend()
    for i, m in enumerate(metricas_reg):
        ax.text(i - largura/2, r1[m] + 0.01, f'{r1[m]:.2f}',
                ha='center', fontsize=9)
        ax.text(i + largura/2, r2[m] + 0.01, f'{r2[m]:.2f}',
                ha='center', fontsize=9)

    # Classificação
    ax = axes[1]
    if v2.get('cls') and v1.get('cls'):
        c1 = v1['cls']['resultados'][v1['cls']['melhor']]
        c2 = v2['cls']['resultados'][v2['cls']['melhor']]
        metricas_cls = ['acc', 'precision', 'recall', 'f1', 'auc']
        x = np.arange(len(metricas_cls))
        ax.bar(x - largura/2, [c1[m] for m in metricas_cls], largura,
               label=f'V1 ({v1["cls"]["melhor"]})', color='steelblue')
        ax.bar(x + largura/2, [c2[m] for m in metricas_cls], largura,
               label=f'V2 ({v2["cls"]["melhor"]})', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(['Acc', 'Prec', 'Rec', 'F1', 'AUC'])
        ax.set_title('Classificação Detrator (melhor modelo de cada versão)')
        ax.legend()
        ax.set_ylim(0, 1.1)
        for i, m in enumerate(metricas_cls):
            ax.text(i - largura/2, c1[m] + 0.02, f'{c1[m]:.2f}',
                    ha='center', fontsize=8)
            ax.text(i + largura/2, c2[m] + 0.02, f'{c2[m]:.2f}',
                    ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / '06_comparativo_metricas_modelos.png'); plt.close()
    print('  ✓ 06_comparativo_metricas_modelos.png')


# ============================================================================
# SALVAR DADOS DO COMPARATIVO PARA O WORD
# ============================================================================
def salvar_comparativo_csv(v1: dict, v2: dict) -> None:
    """Salva os números do comparativo em CSV — usados pelo gerar_word_comparativo."""

    # Visão geral
    rows = []
    rows.append({'metrica': 'n_linhas',
                 'v1': v1['desc']['n_linhas'],
                 'v2': v2['desc']['n_linhas']})
    rows.append({'metrica': 'nps_medio',
                 'v1': v1['desc']['nps_medio'],
                 'v2': v2['desc']['nps_medio']})
    rows.append({'metrica': 'nps_oficial',
                 'v1': v1['desc']['nps_oficial'],
                 'v2': v2['desc']['nps_oficial']})
    for cat in ['Detrator', 'Neutro', 'Promotor']:
        rows.append({'metrica': f'pct_{cat.lower()}',
                     'v1': v1['desc']['dist'][cat],
                     'v2': v2['desc']['dist'][cat]})

    pd.DataFrame(rows).to_csv(REPORT_DIR / 'comparativo_visao_geral.csv',
                              index=False)

    # Modelos
    r1 = v1['reg']['resultados'][v1['reg']['melhor']]
    r2 = v2['reg']['resultados'][v2['reg']['melhor']]
    pd.DataFrame([
        {'metrica': 'MAE', 'v1': r1['mae'], 'v2': r2['mae']},
        {'metrica': 'RMSE', 'v1': r1['rmse'], 'v2': r2['rmse']},
        {'metrica': 'R²', 'v1': r1['r2'], 'v2': r2['r2']},
        {'metrica': 'modelo_v1', 'v1': v1['reg']['melhor'], 'v2': ''},
        {'metrica': 'modelo_v2', 'v1': '', 'v2': v2['reg']['melhor']},
    ]).to_csv(REPORT_DIR / 'comparativo_regressao.csv', index=False)

    if v1.get('cls') and v2.get('cls'):
        c1 = v1['cls']['resultados'][v1['cls']['melhor']]
        c2 = v2['cls']['resultados'][v2['cls']['melhor']]
        pd.DataFrame([
            {'metrica': 'Accuracy',  'v1': c1['acc'],       'v2': c2['acc']},
            {'metrica': 'Precision', 'v1': c1['precision'], 'v2': c2['precision']},
            {'metrica': 'Recall',    'v1': c1['recall'],    'v2': c2['recall']},
            {'metrica': 'F1',        'v1': c1['f1'],        'v2': c2['f1']},
            {'metrica': 'AUC',       'v1': c1['auc'],       'v2': c2['auc']},
            {'metrica': 'modelo_v1', 'v1': v1['cls']['melhor'], 'v2': ''},
            {'metrica': 'modelo_v2', 'v1': '', 'v2': v2['cls']['melhor']},
        ]).to_csv(REPORT_DIR / 'comparativo_classificacao.csv', index=False)

    # Correlações
    df_corr = pd.DataFrame({
        'v1': v1['desc']['correlacoes'],
        'v2': v2['desc']['correlacoes'],
    })
    df_corr['delta'] = df_corr['v2'] - df_corr['v1']
    df_corr['delta_abs'] = df_corr['delta'].abs()
    df_corr.sort_values('delta_abs', ascending=False, inplace=True)
    df_corr.to_csv(REPORT_DIR / 'comparativo_correlacoes.csv')

    print('\n  ✓ CSVs do comparativo salvos em reports/')


# ============================================================================
# MAIN
# ============================================================================
def main():
    print('TECH CHALLENGE FASE 1 — VERSÃO 2 (NPS=0 como outliers)')
    print('Gerando comparativo com a V1 (base completa)\n')

    # V2 — base sem NPS=0
    secao('PREPARAÇÃO V2')
    df_v2 = preparar_dados(remover_nps_zero=True)
    df_v2.to_csv(BASE_DIR / 'data' / 'desafio_nps_processado_v2.csv', index=False)
    desc_v2 = analise_descritiva(df_v2)
    gerar_figuras_v2(df_v2)
    reg_v2 = treinar_regressao(df_v2)
    cls_v2 = treinar_classificacao(df_v2)
    v2 = {'df': df_v2, 'desc': desc_v2, 'reg': reg_v2, 'cls': cls_v2}

    # V1 — base completa, recalculada
    v1 = calcular_v1()

    # Comparativo
    gerar_comparativo(v1, v2)
    salvar_comparativo_csv(v1, v2)

    secao('PIPELINE V2 + COMPARATIVO CONCLUÍDOS ✓')
    print(f'Figuras V2     : {FIG_DIR}')
    print(f'Modelos V2     : {MODEL_DIR}')
    print(f'CSVs comparativo: {REPORT_DIR}')


if __name__ == '__main__':
    main()
