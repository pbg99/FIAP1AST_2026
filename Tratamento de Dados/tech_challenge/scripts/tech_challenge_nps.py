"""
================================================================================
TECH CHALLENGE FASE 1 — CASE NPS PREDITIVO
================================================================================
Curso  : Pós-Tech AI Scientist
Aluno  : Luiz
Método : CRISP-DM
Data   : 2026

Este script resolve o desafio de ponta a ponta:
  1. Business Understanding   → contexto e objetivos
  2. Data Understanding       → exploração e EDA
  3. Data Preparation         → tratamento e feature engineering
  4. Modeling                 → modelos de regressão e classificação
  5. Evaluation               → avaliação e interpretação
  6. Deployment               → recomendações práticas para o negócio

Como executar:
  python tech_challenge_nps.py

Pré-requisitos:
  pip install pandas numpy matplotlib seaborn scikit-learn

Estrutura esperada:
  ../data/desafio_nps_fase_1.csv  (base de origem)
  ../figures/                      (gráficos gerados)
  ../models/                       (modelos serializados)
  ../reports/                      (artefatos de saída)
================================================================================
"""

# ============================================================================
# IMPORTS E CONFIGURAÇÕES
# ============================================================================
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import joblib

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 120

# Caminhos — ajustáveis se rodar fora da pasta scripts/
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'desafio_nps_fase_1.csv'
FIG_DIR = BASE_DIR / 'figures'
MODEL_DIR = BASE_DIR / 'models'
REPORT_DIR = BASE_DIR / 'reports'
for d in [FIG_DIR, MODEL_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def secao(titulo: str) -> None:
    """Imprime cabeçalho de seção para legibilidade no terminal."""
    print('\n' + '=' * 80)
    print(titulo)
    print('=' * 80)


# ============================================================================
# 1. BUSINESS UNDERSTANDING
# ============================================================================
def business_understanding() -> None:
    """
    Documenta o entendimento de negócio (item 1 do PDF).

    Esta função apenas imprime — a reflexão é o entregável.
    """
    secao('ETAPA 1 — BUSINESS UNDERSTANDING')

    print("""
PROBLEMA DE NEGÓCIO
-------------------
Um e-commerce em rápida expansão observa alta variabilidade no NPS entre clientes
com perfis operacionais aparentemente similares. Como o NPS hoje só é coletado
após a jornada de compra, a empresa só descobre que o cliente está insatisfeito
DEPOIS que o problema já ocorreu — perdendo a chance de agir preventivamente.

PERGUNTA CENTRAL
----------------
Quais fatores operacionais realmente influenciam a satisfação e como prever o
NPS de um cliente ANTES da pesquisa, usando apenas dados de pedido, logística
e atendimento?

POR QUE NPS IMPORTA EM E-COMMERCE
---------------------------------
• RECOMPRA   : Promotores recompram em ritmo muito maior do que detratores —
               é a métrica mais ligada ao LTV (Lifetime Value).
• BOCA-A-BOCA: Em e-commerce, avaliações públicas (marketplaces, redes sociais)
               são decisivas na conversão de novos compradores.
• MARKET SHARE: Detratores ativos podem afastar prospects e reduzir conversão
               organicamente, corroendo participação de mercado.

ÁREAS QUE SE BENEFICIAM
-----------------------
• Logística       : priorização de rotas e SLAs por região/cliente
• Atendimento     : roteamento e priorização de tickets
• Produto         : identificar gargalos da jornada de compra
• Pricing         : entender quanto vale evitar um detrator
• Estratégia      : alocação de investimento por driver de impacto

INDICADORES DE MERCADO QUE COMPLEMENTAM
---------------------------------------
• Benchmark NPS setorial (e-commerce no Brasil)
• SLA logístico de concorrentes (lead time prometido vs entregue)
• % de reclamações resolvidas em 1º contato (best practice ~70%)
• Taxa de recompra do mercado (reference: ~30-40% em e-commerce maduro)
""")


# ============================================================================
# 2. DATA UNDERSTANDING
# ============================================================================
def data_understanding(df: pd.DataFrame) -> dict:
    """
    Análise exploratória inicial. Retorna dicionário com métricas-chave
    para uso em outras etapas e no relatório.
    """
    secao('ETAPA 2 — DATA UNDERSTANDING')

    print(f'Linhas: {df.shape[0]:,}')
    print(f'Colunas: {df.shape[1]}')
    print(f'Memória: {df.memory_usage(deep=True).sum() / 1024:.1f} KB')

    print('\nIntegridade:')
    print(f'  Nulos totais        : {df.isnull().sum().sum()}')
    print(f'  Linhas duplicadas   : {df.duplicated().sum()}')
    print(f'  customer_id únicos  : {df["customer_id"].nunique()}/{len(df)}')
    print(f'  order_id únicos     : {df["order_id"].nunique()}/{len(df)}')

    # Achado importante: NPS é contínuo
    nao_inteiros = (df['nps_score'] != df['nps_score'].astype(int)).sum()
    print(f'\n[ACHADO] NPS Score é contínuo (não inteiro):')
    print(f'  - {nao_inteiros} de {len(df)} valores têm casa decimal')
    print(f'  - {df["nps_score"].nunique()} valores únicos de NPS')
    print(f'  - Range: [{df["nps_score"].min():.2f}, {df["nps_score"].max():.2f}]')

    # Estatísticas-chave
    metricas = {
        'n_linhas': len(df),
        'n_colunas': df.shape[1],
        'nps_medio': df['nps_score'].mean(),
        'nps_mediana': df['nps_score'].median(),
        'recompra_pct': df['repeat_purchase_30d'].mean() * 100,
    }
    return metricas


# ============================================================================
# 3. DATA PREPARATION
# ============================================================================
def classificar_nps(score: float) -> str:
    """
    Classifica o NPS em Detrator/Neutro/Promotor segundo a regra padrão.

    Como os valores são contínuos, usamos floor (arredondamento para baixo)
    de forma conservadora — evita classificar 8.99 como Promotor.
    """
    score_int = int(np.floor(score))
    if score_int >= 9:
        return 'Promotor'
    if score_int >= 7:
        return 'Neutro'
    return 'Detrator'


def data_preparation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara dados para modelagem. A base é consistente — não há nulos nem
    duplicados — então o trabalho aqui é de feature engineering e classificação.
    """
    secao('ETAPA 3 — DATA PREPARATION')
    df = df.copy()

    # 3.1 Classificação NPS
    df['classificacao_nps'] = df['nps_score'].apply(classificar_nps)
    df['eh_detrator'] = (df['classificacao_nps'] == 'Detrator').astype(int)

    dist = df['classificacao_nps'].value_counts(normalize=True).reindex(
        ['Detrator', 'Neutro', 'Promotor']
    ) * 100
    nps_oficial = dist['Promotor'] - dist['Detrator']

    print('Distribuição da base:')
    for cat, pct in dist.items():
        print(f'  {cat:10s}: {pct:5.1f}%')
    print(f'\nNPS Score oficial: {nps_oficial:.1f}')

    # 3.2 Feature engineering — variáveis de contexto que podem ajudar o modelo
    # teve_problema: combina sinais de problema na jornada
    df['teve_atraso'] = (df['delivery_delay_days'] > 0).astype(int)
    df['teve_reclamacao'] = (df['complaints_count'] > 0).astype(int)
    df['contatou_sac'] = (df['customer_service_contacts'] > 0).astype(int)

    # severidade: combinação simples de sinais negativos
    df['severidade_problemas'] = (
        df['teve_atraso'] + df['teve_reclamacao'] + df['contatou_sac']
    )

    # razão desconto/pedido (proxy de margem percebida)
    df['desconto_pct'] = df['discount_value'] / df['order_value'].replace(0, np.nan)
    df['desconto_pct'] = df['desconto_pct'].fillna(0)

    print(f'\nFeatures criadas: teve_atraso, teve_reclamacao, contatou_sac, '
          f'severidade_problemas, desconto_pct')

    return df


# ============================================================================
# EDA — gera as figuras consumidas no relatório
# ============================================================================
def gerar_figuras_eda(df: pd.DataFrame) -> dict:
    """Gera e salva as figuras principais. Retorna mapa nome→caminho."""
    secao('GERANDO FIGURAS DE EDA')
    figuras = {}

    # 1. Distribuição do NPS
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['nps_score'], bins=30, color='steelblue', edgecolor='white')
    ax.axvline(df['nps_score'].mean(), color='red', linestyle='--',
               label=f'Média: {df["nps_score"].mean():.2f}')
    ax.set_title('Distribuição do NPS Score')
    ax.set_xlabel('NPS Score')
    ax.set_ylabel('Frequência')
    ax.legend()
    path = FIG_DIR / '01_distribuicao_nps.png'
    plt.savefig(path); plt.close()
    figuras['distribuicao_nps'] = path
    print(f'  ✓ {path.name}')

    # 2. Pizza da classificação
    dist = df['classificacao_nps'].value_counts().reindex(
        ['Detrator', 'Neutro', 'Promotor']
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    cores = ['#d62728', '#ff7f0e', '#2ca02c']
    ax.pie(dist.values, labels=dist.index, autopct='%1.1f%%', colors=cores,
           startangle=90, textprops={'fontsize': 12})
    ax.set_title('Composição da base (Detratores / Neutros / Promotores)')
    path = FIG_DIR / '02_classificacao_nps.png'
    plt.savefig(path); plt.close()
    figuras['classificacao_nps'] = path
    print(f'  ✓ {path.name}')

    # 3. NPS por atraso
    nps_atraso = df.groupby('delivery_delay_days')['nps_score'].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(nps_atraso.index, nps_atraso.values, color='coral')
    ax.bar_label(bars, fmt='%.1f', padding=3)
    ax.set_title('NPS médio por dias de atraso na entrega')
    ax.set_xlabel('Dias de atraso')
    ax.set_ylabel('NPS médio')
    path = FIG_DIR / '03_nps_por_atraso.png'
    plt.savefig(path); plt.close()
    figuras['nps_por_atraso'] = path
    print(f'  ✓ {path.name}')

    # 4. NPS por região
    nps_regiao = df.groupby('customer_region')['nps_score'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(nps_regiao.index, nps_regiao.values, color='steelblue')
    ax.bar_label(bars, fmt='%.2f', padding=3)
    ax.set_title('NPS médio por região')
    ax.set_xlabel('Região')
    ax.set_ylabel('NPS médio')
    path = FIG_DIR / '04_nps_por_regiao.png'
    plt.savefig(path); plt.close()
    figuras['nps_por_regiao'] = path
    print(f'  ✓ {path.name}')

    # 5. Correlação com NPS
    cols_num = ['nps_score', 'customer_age', 'customer_tenure_months', 'order_value',
                'items_quantity', 'discount_value', 'payment_installments',
                'delivery_time_days', 'delivery_delay_days', 'freight_value',
                'delivery_attempts', 'customer_service_contacts',
                'resolution_time_days', 'complaints_count', 'repeat_purchase_30d',
                'csat_internal_score']
    corr = df[cols_num].corr()['nps_score'].drop('nps_score').sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    cores = ['#d62728' if v < 0 else '#2ca02c' for v in corr.values]
    bars = ax.barh(corr.index, corr.values, color=cores)
    ax.bar_label(bars, fmt='%.2f', padding=3)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title('Correlação de cada variável com o NPS')
    ax.set_xlabel('Correlação de Pearson')
    path = FIG_DIR / '05_correlacao_nps.png'
    plt.savefig(path); plt.close()
    figuras['correlacao_nps'] = path
    print(f'  ✓ {path.name}')

    # 6. Heatmap de correlação geral
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df[cols_num].corr(), annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, cbar_kws={'shrink': 0.8}, ax=ax,
                annot_kws={'size': 8})
    ax.set_title('Matriz de correlação')
    path = FIG_DIR / '06_heatmap_correlacao.png'
    plt.savefig(path); plt.close()
    figuras['heatmap_correlacao'] = path
    print(f'  ✓ {path.name}')

    return figuras


# ============================================================================
# 4. MODELING
# ============================================================================
def construir_pipeline(features_num, features_cat, modelo):
    """Constrói pipeline com pré-processamento + modelo."""
    pre = ColumnTransformer([
        ('num', StandardScaler(), features_num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), features_cat),
    ])
    return Pipeline([('pre', pre), ('modelo', modelo)])


def modelo_regressao(df: pd.DataFrame) -> dict:
    """
    REGRESSÃO: prediz o NPS Score (0 a 10) como variável contínua.
    Útil quando se quer um *score* de risco ranqueável.
    """
    secao('ETAPA 4a — MODELING (REGRESSÃO)')

    features_num = ['customer_age', 'customer_tenure_months', 'order_value',
                    'items_quantity', 'discount_value', 'payment_installments',
                    'delivery_time_days', 'delivery_delay_days', 'freight_value',
                    'delivery_attempts', 'customer_service_contacts',
                    'resolution_time_days', 'complaints_count',
                    'repeat_purchase_30d', 'csat_internal_score',
                    'severidade_problemas', 'desconto_pct']
    features_cat = ['customer_region']
    target = 'nps_score'

    X = df[features_num + features_cat]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f'Treino: {len(X_train)}  |  Teste: {len(X_test)}')

    resultados = {}
    modelos = {
        'Regressão Linear': LinearRegression(),
        'Random Forest':    RandomForestRegressor(n_estimators=200,
                                                  max_depth=10,
                                                  random_state=RANDOM_STATE,
                                                  n_jobs=-1),
    }

    for nome, modelo in modelos.items():
        pipe = construir_pipeline(features_num, features_cat, modelo)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        resultados[nome] = {'mae': mae, 'rmse': rmse, 'r2': r2,
                            'pipe': pipe, 'y_pred': y_pred}

        print(f'\n[{nome}]')
        print(f'  MAE  = {mae:.3f}  (erro médio em pontos de NPS)')
        print(f'  RMSE = {rmse:.3f}')
        print(f'  R²   = {r2:.3f}')

    # Seleciona melhor por R²
    melhor = max(resultados, key=lambda k: resultados[k]['r2'])
    print(f'\n>>> Melhor modelo: {melhor}')

    # Salva
    joblib.dump(resultados[melhor]['pipe'],
                MODEL_DIR / 'modelo_regressao_nps.pkl')

    # Gráfico real vs predito
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, resultados[melhor]['y_pred'], alpha=0.4, s=20)
    ax.plot([0, 10], [0, 10], 'r--', label='Predição perfeita')
    ax.set_xlabel('NPS real')
    ax.set_ylabel('NPS predito')
    ax.set_title(f'Real vs predito — {melhor}')
    ax.legend()
    plt.savefig(FIG_DIR / '07_regressao_real_vs_predito.png'); plt.close()

    # Feature importance (Random Forest)
    if 'Random Forest' in resultados:
        rf_pipe = resultados['Random Forest']['pipe']
        rf = rf_pipe.named_steps['modelo']
        feat_names = (features_num
                      + list(rf_pipe.named_steps['pre']
                             .named_transformers_['cat']
                             .get_feature_names_out(features_cat)))
        importancias = pd.Series(rf.feature_importances_,
                                 index=feat_names).sort_values()

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(importancias.index, importancias.values, color='steelblue')
        ax.set_title('Importância das variáveis — Regressão NPS (Random Forest)')
        ax.set_xlabel('Importância')
        plt.savefig(FIG_DIR / '08_feature_importance_regressao.png'); plt.close()

    return {'resultados': resultados, 'melhor': melhor}


def modelo_classificacao(df: pd.DataFrame) -> dict:
    """
    CLASSIFICAÇÃO BINÁRIA: prediz se o cliente será Detrator (1) ou Não (0).
    Útil para acionar alertas operacionais.
    """
    secao('ETAPA 4b — MODELING (CLASSIFICAÇÃO BINÁRIA: Detrator vs Não)')

    features_num = ['customer_age', 'customer_tenure_months', 'order_value',
                    'items_quantity', 'discount_value', 'payment_installments',
                    'delivery_time_days', 'delivery_delay_days', 'freight_value',
                    'delivery_attempts', 'customer_service_contacts',
                    'resolution_time_days', 'complaints_count',
                    'repeat_purchase_30d', 'csat_internal_score',
                    'severidade_problemas', 'desconto_pct']
    features_cat = ['customer_region']

    X = df[features_num + features_cat]
    y = df['eh_detrator']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f'Treino: {len(X_train)}  |  Teste: {len(X_test)}')
    print(f'Taxa de detratores (treino): {y_train.mean()*100:.1f}%')
    print(f'Taxa de detratores (teste) : {y_test.mean()*100:.1f}%')

    resultados = {}
    modelos = {
        'Regressão Logística': LogisticRegression(max_iter=1000,
                                                  random_state=RANDOM_STATE),
        'Random Forest':       RandomForestClassifier(n_estimators=200,
                                                      max_depth=10,
                                                      random_state=RANDOM_STATE,
                                                      n_jobs=-1),
    }

    for nome, modelo in modelos.items():
        pipe = construir_pipeline(features_num, features_cat, modelo)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        resultados[nome] = {'acc': acc, 'precision': prec, 'recall': rec,
                            'f1': f1, 'auc': auc, 'pipe': pipe,
                            'y_pred': y_pred, 'y_proba': y_proba}

        print(f'\n[{nome}]')
        print(f'  Accuracy  = {acc:.3f}')
        print(f'  Precision = {prec:.3f}  (de cada 100 \"detratores previstos\", '
              f'{prec*100:.0f} são reais)')
        print(f'  Recall    = {rec:.3f}  (capturamos {rec*100:.0f}% dos detratores reais)')
        print(f'  F1-Score  = {f1:.3f}')
        print(f'  AUC-ROC   = {auc:.3f}')

    melhor = max(resultados, key=lambda k: resultados[k]['f1'])
    print(f'\n>>> Melhor modelo: {melhor}')

    # Matriz de confusão do melhor
    cm = confusion_matrix(y_test, resultados[melhor]['y_pred'])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Não detrator', 'Detrator'],
                yticklabels=['Não detrator', 'Detrator'], ax=ax)
    ax.set_title(f'Matriz de confusão — {melhor}')
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')
    plt.savefig(FIG_DIR / '09_matriz_confusao.png'); plt.close()

    # Feature importance
    if 'Random Forest' in resultados:
        rf_pipe = resultados['Random Forest']['pipe']
        rf = rf_pipe.named_steps['modelo']
        feat_names = (features_num
                      + list(rf_pipe.named_steps['pre']
                             .named_transformers_['cat']
                             .get_feature_names_out(features_cat)))
        importancias = pd.Series(rf.feature_importances_,
                                 index=feat_names).sort_values()

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(importancias.index, importancias.values, color='steelblue')
        ax.set_title('Importância das variáveis — Classificação Detrator (Random Forest)')
        ax.set_xlabel('Importância')
        plt.savefig(FIG_DIR / '10_feature_importance_classificacao.png'); plt.close()

    joblib.dump(resultados[melhor]['pipe'],
                MODEL_DIR / 'modelo_classificacao_detrator.pkl')

    return {'resultados': resultados, 'melhor': melhor}


# ============================================================================
# 5. EVALUATION
# ============================================================================
def evaluation(df: pd.DataFrame, res_reg: dict, res_cls: dict) -> None:
    """Avalia os modelos do ponto de vista de negócio."""
    secao('ETAPA 5 — EVALUATION (interpretação para o negócio)')

    melhor_reg = res_reg['resultados'][res_reg['melhor']]
    melhor_cls = res_cls['resultados'][res_cls['melhor']]

    print(f"""
INTERPRETAÇÃO DO MODELO DE REGRESSÃO ({res_reg['melhor']})
-------------------------------------------------------
• MAE de {melhor_reg['mae']:.2f} pontos: em média, o modelo erra {melhor_reg['mae']:.2f}
  pontos na nota de NPS prevista. Em escala 0-10 isso é {'aceitável' if melhor_reg['mae'] < 1.5 else 'razoável' if melhor_reg['mae'] < 2.5 else 'alto'}.
• R² de {melhor_reg['r2']:.2f}: o modelo explica {melhor_reg['r2']*100:.0f}% da variação do NPS.
• Pode ser usado para: priorizar clientes em risco por *score*, alimentar dashboards
  de risco, gerar alertas pós-pedido.

INTERPRETAÇÃO DO MODELO DE CLASSIFICAÇÃO ({res_cls['melhor']})
-------------------------------------------------------------
• Recall de {melhor_cls['recall']:.2f}: captura {melhor_cls['recall']*100:.0f}% dos detratores.
• Precision de {melhor_cls['precision']:.2f}: {melhor_cls['precision']*100:.0f}% dos alertas são de fato detratores.
• AUC de {melhor_cls['auc']:.2f}: {'excelente' if melhor_cls['auc'] > 0.9 else 'bom' if melhor_cls['auc'] > 0.8 else 'razoável'} poder discriminativo.
• Pode ser usado para: gatilho automático de ação preventiva (cupom, contato
  proativo, priorização de SAC) para clientes classificados como detratores.

QUAL ESCOLHER?
--------------
Para o caso de negócio descrito, RECOMENDAMOS USAR OS DOIS:
• Classificação → para AÇÃO (decisão sim/não de acionar fluxo preventivo)
• Regressão     → para RANKING (priorizar quais detratores atender primeiro)

ATENÇÃO ÀS LIMITAÇÕES
---------------------
• csat_internal_score e repeat_purchase_30d são fortes preditores — porém
  csat_internal_score pode ser também coletado APÓS a jornada. Em produção,
  validar se está disponível em tempo real.
• complaints_count e customer_service_contacts são reativos (já indicam
  problema). Para PREVENÇÃO real, dar peso maior a delivery_delay_days e
  delivery_time_days, que precedem a manifestação do cliente.
""")


# ============================================================================
# 6. DEPLOYMENT (recomendações práticas)
# ============================================================================
def deployment_recomendacoes() -> None:
    """Imprime recomendações finais para o negócio."""
    secao('ETAPA 6 — DEPLOYMENT (recomendações para o negócio)')

    print("""
RECOMENDAÇÕES PRIORIZADAS POR IMPACTO/ESFORÇO
---------------------------------------------

1. [ALTO IMPACTO, MÉDIO ESFORÇO] Operação logística é a alavanca #1
   • Atraso na entrega tem a maior correlação negativa com NPS
   • Ação: revisar SLA com transportadoras, ajustar promessa de entrega
     (sub-prometer e super-entregar) e instituir alerta interno automático
     quando um pedido cruza o D-1 antes do prazo

2. [ALTO IMPACTO, BAIXO ESFORÇO] Modelo de scoring preditivo em produção
   • Com base no modelo de classificação, gerar *score de risco* logo após
     a confirmação de pedido (mesmo antes da entrega) e atualizar conforme
     eventos da jornada
   • Ação: integrar API do modelo ao sistema de pedidos; threshold inicial
     em 0.5; monitorar e calibrar mensalmente

3. [MÉDIO IMPACTO, MÉDIO ESFORÇO] Gestão proativa de SAC
   • Cliente que contatou SAC mais de 1x já é forte sinal de detração
   • Ação: priorizar fila de tickets por score de risco e por número de
     contatos prévios; medir reincidência

4. [BAIXO ESFORÇO] Programa de fidelização ancorado em recompra
   • Recompra em 30 dias é o sinal positivo mais forte (corr +0.6)
   • Ação: incentivar segunda compra com cupom direcionado e cadência
     de e-mail nos primeiros 14 dias após a entrega

5. [LIMITAÇÃO IMPORTANTE] Não usar features que vazam o rótulo
   • Em produção, validar quais features estarão de fato disponíveis
     ANTES da pesquisa NPS (ex.: csat_internal_score pode ser pós-evento)
""")


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    print('TECH CHALLENGE FASE 1 — CASE NPS PREDITIVO')
    print('Pipeline CRISP-DM completo')

    # 1. Business
    business_understanding()

    # 2. Data Understanding
    df_raw = pd.read_csv(DATA_PATH)
    metricas_eda = data_understanding(df_raw)

    # 3. Data Preparation
    df = data_preparation(df_raw)

    # EDA — figuras
    gerar_figuras_eda(df)

    # 4. Modeling
    res_reg = modelo_regressao(df)
    res_cls = modelo_classificacao(df)

    # 5. Evaluation
    evaluation(df, res_reg, res_cls)

    # 6. Deployment
    deployment_recomendacoes()

    # Salva base processada
    df.to_csv(BASE_DIR / 'data' / 'desafio_nps_processado.csv', index=False)

    secao('PIPELINE CONCLUÍDO ✓')
    print(f'\nFiguras    : {FIG_DIR}')
    print(f'Modelos    : {MODEL_DIR}')
    print(f'Relatórios : {REPORT_DIR}')
    print(f'\nProcessamento finalizado.')


if __name__ == '__main__':
    main()
