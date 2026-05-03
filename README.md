# FIAP1AST_2026

# EDA e Modelo Preditivo — NPS E-commerce

Este notebook apresenta a análise exploratória de dados e a construção de modelos preditivos para o Net Promoter Score (NPS) de um e-commerce.

**Pergunta central:** Quais fatores operacionais mais influenciam a satisfação do cliente e como a empresa pode agir de forma proativa para melhorar a experiência antes da pesquisa de NPS?

**Estrutura do notebook:**

1. Diagnosticar o NPS geral e identificar a distribuição de detratores, neutros e promotores
2. Identificar os principais fatores associados à insatisfação (atendimento, reclamações, logística)
3. Separar sinais fortes de sinais complementares
4. Aprofundar a principal hipótese logística: atraso de entrega como ponto de ruptura
5. Transformar os achados em recomendações operacionais
6. Construir modelos preditivos (regressão e classificação) para antecipar detratores

**Base de dados:** 2.500 pedidos com informações de pedido, logística, atendimento e indicadores internos.
## Visão geral da análise

O notebook está organizado em duas grandes frentes:

- **EDA geral (seções 0-5):** atendimento, reclamações, CSAT, recompra, região, perfil demográfico e score simples de risco de detrator.
- **Aprofundamento logístico (seções 6-10):** atraso, faixas de atraso, frete, tentativas de entrega, faturamento e análise regional logística.
- **Modelo preditivo (seção 11):** regressão para estimar a nota NPS e classificação para identificar detratores antes da pesquisa.

Cada seção traz os gráficos, as métricas e as interpretações de negócio correspondentes.
# 0. Preparação e carregamento da base geral

Esta etapa carrega a base inicial e cria as variáveis auxiliares usadas no EDA geral.

## 0.1 Importando bibliotecas

import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

plt.rcParams["figure.figsize"] = (9, 4)
plt.rcParams["axes.grid"] = True

print("Bibliotecas carregadas com sucesso")
## 0.2 Carregando a base geral no VSCode

Esta célula procura automaticamente o arquivo `desafio_nps_fase_1.csv` na pasta atual e nas subpastas.

Para funcionar sem alterar nada, deixe o CSV na mesma pasta do notebook ou em alguma subpasta do projeto aberto no VSCode.

from pathlib import Path

# -------------------------------------------------------------------------
# CONFIGURAÇÃO ÚNICA DA BASE
# -------------------------------------------------------------------------
# Para usar no VSCode sem alterar o notebook inteiro:
# 1. Deixe o arquivo desafio_nps_fase_1.csv na mesma pasta deste notebook; ou
# 2. Deixe o arquivo em alguma subpasta do projeto aberto no VSCode.
#
# Se quiser forçar uma pasta específica, altere apenas a linha abaixo:
PASTA_PROJETO = Path.cwd()

NOME_BASE_V1 = "desafio_nps_fase_1.csv"
NOME_BASE_V2 = "desafio_nps_processado.csv"


def localizar_arquivo(nome_arquivo, pasta_base=PASTA_PROJETO):
    """Localiza um arquivo na pasta atual, nos pais próximos e nas subpastas."""
    pasta_base = Path(pasta_base).resolve()

    candidatos = []

    # 1) pasta atual
    candidatos.append(pasta_base / nome_arquivo)

    # 2) pais próximos, útil quando o kernel inicia em uma subpasta
    candidatos.extend([p / nome_arquivo for p in list(pasta_base.parents)[:4]])

    # 3) subpastas do projeto aberto no VSCode
    try:
        candidatos.extend(pasta_base.rglob(nome_arquivo))
    except Exception:
        pass

    # Remove duplicados preservando ordem
    vistos = set()
    candidatos_unicos = []
    for caminho in candidatos:
        caminho = Path(caminho)
        if caminho not in vistos:
            vistos.add(caminho)
            candidatos_unicos.append(caminho)

    for caminho in candidatos_unicos:
        if caminho.exists():
            return caminho

    caminhos_testados = "\n".join(str(c) for c in candidatos_unicos[:20])
    raise FileNotFoundError(
        f"Arquivo '{nome_arquivo}' não encontrado.\n\n"
        f"Pasta atual do kernel: {pasta_base}\n\n"
        f"Coloque o CSV na mesma pasta do notebook ou ajuste PASTA_PROJETO.\n\n"
        f"Alguns caminhos testados:\n{caminhos_testados}"
    )


path_v1 = localizar_arquivo(NOME_BASE_V1)
caminho_csv = path_v1

df = pd.read_csv(path_v1)

print("Pasta atual do kernel:", Path.cwd())
print("Arquivo usado:", path_v1)
print("Base carregada com", df.shape[0], "linhas e", df.shape[1], "colunas")

df.head()

## 0.3 Validação inicial da base

print("Tamanho da base:")
print(df.shape)

print("\nColunas da base:")
print(df.columns.tolist())

print("\nTipos de dados:")
display(df.dtypes)

print("\nValores nulos por coluna:")
display(df.isna().sum().sort_values(ascending=False))
## 0.4 Checagem de duplicidade

print("Quantidade de pedidos duplicados:")

if "order_id" in df.columns:
    print(df["order_id"].duplicated().sum())
else:
    print("A coluna order_id não existe na base.")
## 0.5 Criação das variáveis auxiliares

condicoes_nps = [
    df["nps_score"] <= 6,
    (df["nps_score"] > 6) & (df["nps_score"] < 9),
    df["nps_score"] >= 9
]

categorias_nps = ["Detrator", "Neutro", "Promotor"]

df["nps_category"] = np.select(
    condicoes_nps,
    categorias_nps,
    default="Sem classificação"
)

df["is_detractor"] = (df["nps_score"] <= 6).astype(int)

df["contact_range"] = pd.cut(
    df["customer_service_contacts"],
    bins=[-1, 0, 2, 999],
    labels=["Nenhum", "1-2 contatos", "3+ contatos"]
)

df["complaint_range"] = pd.cut(
    df["complaints_count"],
    bins=[-1, 0, 2, 5, 999],
    labels=["Nenhuma", "1-2", "3-5", "6+"]
)

df["resolution_range"] = pd.cut(
    df["resolution_time_days"],
    bins=[-1, 2, 5, 10, 999],
    labels=["Até 2 dias", "3-5 dias", "6-10 dias", "11+ dias"]
)

df["delay_range"] = pd.cut(
    df["delivery_delay_days"],
    bins=[-1, 0, 2, 5, 999],
    labels=["Sem atraso", "1-2 dias", "3-5 dias", "6+ dias"]
)

df["age_range"] = pd.cut(
    df["customer_age"],
    bins=[0, 25, 35, 50, 100],
    labels=["18-25", "26-35", "36-50", "51+"]
)

df["tenure_range"] = pd.cut(
    df["customer_tenure_months"],
    bins=[-1, 12, 36, 72, 999],
    labels=["< 1 ano", "1-3 anos", "3-6 anos", "6+ anos"]
)

print("Variáveis auxiliares criadas")

df[[
    "customer_id",
    "nps_score",
    "nps_category",
    "is_detractor",
    "contact_range",
    "complaint_range",
    "resolution_range",
    "delay_range"
]].head(10)
# 1. Diagnóstico inicial do NPS

Distribuição geral do NPS, NPS médio e concentração de detratores, neutros e promotores na base.
## 1.1 Critério de classificação do NPS

## 1.2 Resumo executivo da base geral

# Resumo executivo da base

total_clientes = len(df)
nps_medio = df["nps_score"].mean()

dist_nps = (
    df["nps_category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

nps_geral = dist_nps.get("Promotor", 0) - dist_nps.get("Detrator", 0)

print("Resumo executivo da base")
print("-" * 40)
print("Total de clientes/pedidos:", total_clientes)
print("NPS médio:", round(nps_medio, 2))
print("NPS geral estimado:", round(nps_geral, 2))
print("\nDistribuição por categoria:")
display(dist_nps)
# 2. Drivers de atendimento e relacionamento

Este bloco concentra as variáveis mais ligadas à experiência de atendimento: contatos, reclamações, tempo de resolução, CSAT e recompra.

A ideia é evitar analisar cada indicador isoladamente sem conectá-lo à jornada do cliente.

## 2.1 Contatos com atendimento e NPS

resumo_p1 = (
    df.groupby("contact_range", observed=False)
      .agg(
          qtd_clientes=("customer_id", "count"),
          nps_medio=("nps_score", "mean"),
          nps_mediano=("nps_score", "median"),
          pct_detrator=("is_detractor", "mean")
      )
      .reset_index()
)

resumo_p1["pct_detrator"] = resumo_p1["pct_detrator"] * 100

display(resumo_p1)
### Gráfico: NPS médio por faixa de contatos

plt.figure(figsize=(8, 4))

plt.bar(
    resumo_p1["contact_range"].astype(str),
    resumo_p1["nps_medio"]
)

plt.title("NPS médio por faixa de contatos com atendimento")
plt.xlabel("Faixa de contatos")
plt.ylabel("NPS médio")
plt.ylim(0, 10)

for i, valor in enumerate(resumo_p1["nps_medio"]):
    plt.text(i, valor + 0.15, f"{valor:.2f}", ha="center")

plt.show()
### Correlação: contatos com atendimento x NPS

r, p = stats.spearmanr(
    df["customer_service_contacts"],
    df["nps_score"]
)

print("Correlação Spearman entre contatos com atendimento e NPS:", round(r, 3))
print("p-valor:", round(p, 4))

if r < 0:
    print("Leitura: quanto mais contatos com atendimento, menor tende a ser o NPS.")
else:
    print("Leitura: não apareceu relação negativa clara entre contatos e NPS.")
## 2.2 Reclamações e queda do NPS

resumo_p2 = (
    df.groupby("complaint_range", observed=False)
      .agg(
          qtd_clientes=("customer_id", "count"),
          nps_medio=("nps_score", "mean"),
          pct_detrator=("is_detractor", "mean")
      )
      .reset_index()
)

resumo_p2["pct_detrator"] = resumo_p2["pct_detrator"] * 100

display(resumo_p2)
### Gráfico: % de detratores por faixa de reclamações

plt.figure(figsize=(8, 4))

plt.bar(
    resumo_p2["complaint_range"].astype(str),
    resumo_p2["pct_detrator"]
)

plt.title("% de detratores por faixa de reclamações")
plt.xlabel("Faixa de reclamações")
plt.ylabel("% de detratores")
plt.ylim(0, 100)

for i, valor in enumerate(resumo_p2["pct_detrator"]):
    plt.text(i, valor + 2, f"{valor:.1f}%", ha="center")

plt.show()
### Correlação: reclamações x NPS

r, p = stats.spearmanr(
    df["complaints_count"],
    df["nps_score"]
)

print("Correlação Spearman entre quantidade de reclamações e NPS:", round(r, 3))
print("p-valor:", round(p, 4))

print("\nNPS médio por quantidade exata de reclamações:")
display(
    df.groupby("complaints_count")["nps_score"]
      .mean()
      .reset_index()
)
## 2.3 Tempo de resolução e satisfação

resumo_p3 = (
    df.groupby("resolution_range", observed=False)
      .agg(
          qtd_clientes=("customer_id", "count"),
          nps_medio=("nps_score", "mean"),
          pct_detrator=("is_detractor", "mean")
      )
      .reset_index()
)

resumo_p3["pct_detrator"] = resumo_p3["pct_detrator"] * 100

display(resumo_p3)
### Gráfico: NPS médio por tempo de resolução

plt.figure(figsize=(8, 4))

plt.plot(
    resumo_p3["resolution_range"].astype(str),
    resumo_p3["nps_medio"],
    marker="o"
)

plt.title("NPS médio por tempo de resolução")
plt.xlabel("Tempo de resolução")
plt.ylabel("NPS médio")
plt.ylim(0, 10)

for i, valor in enumerate(resumo_p3["nps_medio"]):
    plt.text(i, valor + 0.15, f"{valor:.2f}", ha="center")

plt.show()
### Correlação: tempo de resolução x NPS

r, p = stats.spearmanr(
    df["resolution_time_days"],
    df["nps_score"]
)

print("Correlação Spearman entre tempo de resolução e NPS:", round(r, 3))
print("p-valor:", round(p, 4))

if r < 0:
    print("Leitura: quanto maior o tempo de resolução, menor tende a ser o NPS.")
else:
    print("Leitura: não apareceu relação negativa clara.")
## 2.4 CSAT interno e NPS

print("CSAT médio geral:")
print(round(df["csat_internal_score"].mean(), 2))

print("\nCSAT médio dos detratores:")
print(round(df.loc[df["is_detractor"] == 1, "csat_internal_score"].mean(), 2))

print("\nCSAT médio dos não detratores:")
print(round(df.loc[df["is_detractor"] == 0, "csat_internal_score"].mean(), 2))

r, p = stats.spearmanr(
    df["csat_internal_score"],
    df["nps_score"]
)

print("\nCorrelação Spearman entre CSAT interno e NPS:", round(r, 3))
print("p-valor:", round(p, 4))
### Gráfico: CSAT interno x NPS

plt.figure(figsize=(8, 4))

plt.scatter(
    df["csat_internal_score"],
    df["nps_score"],
    alpha=0.15
)

# Linha de tendência simples
coef = np.polyfit(df["csat_internal_score"], df["nps_score"], 1)
linha = np.poly1d(coef)

x = np.linspace(
    df["csat_internal_score"].min(),
    df["csat_internal_score"].max(),
    100
)

plt.plot(x, linha(x), linewidth=2)

plt.title("Relação entre CSAT interno e NPS")
plt.xlabel("CSAT interno")
plt.ylabel("NPS")
plt.ylim(0, 10)

plt.show()
## 2.5 Recompra em 30 dias e NPS

resumo_p6 = (
    df.groupby("repeat_purchase_30d")
      .agg(
          qtd_clientes=("customer_id", "count"),
          nps_medio=("nps_score", "mean"),
          pct_detrator=("is_detractor", "mean")
      )
      .reset_index()
)

resumo_p6["label"] = resumo_p6["repeat_purchase_30d"].map({
    0: "Não recomprou",
    1: "Recomprou"
})

resumo_p6["pct_detrator"] = resumo_p6["pct_detrator"] * 100

display(
    resumo_p6[["label", "qtd_clientes", "nps_medio", "pct_detrator"]]
)
### Gráfico: NPS médio por recompra em 30 dias

plt.figure(figsize=(7, 4))

plt.bar(
    resumo_p6["label"],
    resumo_p6["nps_medio"]
)

plt.title("NPS médio por recompra em 30 dias")
plt.xlabel("Recompra em 30 dias")
plt.ylabel("NPS médio")
plt.ylim(0, 10)

for i, valor in enumerate(resumo_p6["nps_medio"]):
    plt.text(i, valor + 0.15, f"{valor:.2f}", ha="center")

plt.show()
**Leitura sugerida:** recompra deve ser tratada com cuidado. Ela acompanha fortemente a satisfação, mas pode ser consequência da experiência positiva, e não necessariamente causa direta do NPS.

# 3. Ranking dos fatores associados ao NPS

Após olhar os principais indicadores individualmente, este bloco consolida o peso relativo das variáveis operacionais.

Aqui começa a aparecer a hipótese que justifica o P3: **o atraso de entrega é um dos fatores mais fortes para explicar queda de NPS**.

## 3.1 Ranking operacional

variaveis_operacionais = [
    "delivery_delay_days",
    "complaints_count",
    "customer_service_contacts",
    "resolution_time_days",
    "delivery_attempts",
    "freight_value"
]

ranking = []

for coluna in variaveis_operacionais:
    r, p = stats.spearmanr(df[coluna], df["nps_score"])

    ranking.append({
        "variavel": coluna,
        "correlacao_spearman": r,
        "p_valor": p,
        "impacto_abs": abs(r)
    })

ranking_p4 = pd.DataFrame(ranking)

ranking_p4 = ranking_p4.sort_values(
    "impacto_abs",
    ascending=False
)

display(ranking_p4)
### Gráfico: correlação das variáveis operacionais com NPS

plt.figure(figsize=(9, 4))

plt.barh(
    ranking_p4["variavel"],
    ranking_p4["correlacao_spearman"]
)

plt.title("Correlação das variáveis operacionais com NPS")
plt.xlabel("Correlação Spearman")
plt.axvline(0)
plt.gca().invert_yaxis()

plt.show()
# 4. Análises complementares: o que parece menos explicativo

Este bloco reúne variáveis que ajudam a testar hipóteses, mas que não devem ser o centro da recomendação.

A proposta é separar o que é achado forte do que é apenas contexto.

## 4.1 Variação por região

resumo_p7 = (
    df.groupby("customer_region")
      .agg(
          qtd_clientes=("customer_id", "count"),
          nps_medio=("nps_score", "mean"),
          pct_detrator=("is_detractor", "mean")
      )
      .reset_index()
      .sort_values("nps_medio")
)

resumo_p7["pct_detrator"] = resumo_p7["pct_detrator"] * 100

display(resumo_p7)
### Teste estatístico por região

grupos_regiao = []

for regiao, grupo in df.groupby("customer_region"):
    grupos_regiao.append(grupo["nps_score"])

estatistica, p_valor = stats.kruskal(*grupos_regiao)

print("Teste Kruskal-Wallis para diferença de NPS entre regiões")
print("Estatística:", round(estatistica, 3))
print("p-valor:", round(p_valor, 4))

if p_valor < 0.05:
    print("Leitura: existe diferença estatística entre pelo menos duas regiões.")
else:
    print("Leitura: não apareceu diferença estatística relevante entre regiões.")
### Diferença entre regiões

menor_nps_regiao = resumo_p7["nps_medio"].min()
maior_nps_regiao = resumo_p7["nps_medio"].max()
diferenca_regiao = maior_nps_regiao - menor_nps_regiao

print("Menor NPS médio regional:", round(menor_nps_regiao, 2))
print("Maior NPS médio regional:", round(maior_nps_regiao, 2))
print("Diferença entre regiões:", round(diferenca_regiao, 2))

if diferenca_regiao < 0.5:
    print("Leitura: a variação regional é baixa. A insatisfação parece mais operacional do que geográfica.")
else:
    print("Leitura: há diferença regional relevante que merece investigação.")
### Gráfico: NPS médio por região

plt.figure(figsize=(8, 4))

plt.barh(
    resumo_p7["customer_region"],
    resumo_p7["nps_medio"]
)

plt.title("NPS médio por região")
plt.xlabel("NPS médio")
plt.ylabel("Região")
plt.xlim(0, 10)

plt.show()
## 4.2 Idade e tempo de relacionamento

resumo_idade = (
    df.groupby("age_range", observed=False)
      .agg(
          qtd_clientes=("customer_id", "count"),
          nps_medio=("nps_score", "mean"),
          pct_detrator=("is_detractor", "mean")
      )
      .reset_index()
)

resumo_tenure = (
    df.groupby("tenure_range", observed=False)
      .agg(
          qtd_clientes=("customer_id", "count"),
          nps_medio=("nps_score", "mean"),
          pct_detrator=("is_detractor", "mean")
      )
      .reset_index()
)

resumo_idade["pct_detrator"] = resumo_idade["pct_detrator"] * 100
resumo_tenure["pct_detrator"] = resumo_tenure["pct_detrator"] * 100

print("Idade x NPS")
display(resumo_idade)

print("Tempo de relacionamento x NPS")
display(resumo_tenure)
### Correlação: idade, tempo de relacionamento e NPS

r_idade, p_idade = stats.spearmanr(
    df["customer_age"],
    df["nps_score"]
)

r_tenure, p_tenure = stats.spearmanr(
    df["customer_tenure_months"],
    df["nps_score"]
)

print("Correlação idade x NPS:", round(r_idade, 3))
print("p-valor idade:", round(p_idade, 4))

print("\nCorrelação tempo de relacionamento x NPS:", round(r_tenure, 3))
print("p-valor tempo de relacionamento:", round(p_tenure, 4))
### Gráficos: idade e tempo de relacionamento

plt.figure(figsize=(8, 4))

plt.bar(
    resumo_idade["age_range"].astype(str),
    resumo_idade["nps_medio"]
)

plt.title("NPS médio por faixa etária")
plt.xlabel("Faixa etária")
plt.ylabel("NPS médio")
plt.ylim(0, 10)

plt.show()


plt.figure(figsize=(8, 4))

plt.bar(
    resumo_tenure["tenure_range"].astype(str),
    resumo_tenure["nps_medio"]
)

plt.title("NPS médio por tempo de relacionamento")
plt.xlabel("Tempo de relacionamento")
plt.ylabel("NPS médio")
plt.ylim(0, 10)

plt.show()
**Leitura sugerida:** se a variação regional, idade e tempo de relacionamento não apresentarem diferença relevante, a recomendação deve priorizar a jornada operacional, não uma estratégia segmentada por perfil demográfico ou região.

# 5. Consolidação operacional e score simples de risco

Este bloco transforma os achados em uma regra prática para priorização.

A ideia não é substituir um modelo preditivo, mas criar um critério simples e explicável para ação operacional.

## 5.1 Clientes com atraso e concentração de detratores

df_atraso = df[df["delivery_delay_days"] > 0].copy()

print("Clientes com atraso:", len(df_atraso))

pct_base_atraso = len(df_atraso) / len(df) * 100
print("% da base com atraso:", round(pct_base_atraso, 1))

pct_detratores_atraso = df_atraso["is_detractor"].mean() * 100
print("% de detratores entre atrasados:", round(pct_detratores_atraso, 1))
## 5.2 Comparação entre detratores e não detratores

comparacao_simples = (
    df.groupby("is_detractor")
      .agg(
          media_contatos=("customer_service_contacts", "mean"),
          media_reclamacoes=("complaints_count", "mean"),
          tempo_resolucao_medio=("resolution_time_days", "mean"),
          csat_medio=("csat_internal_score", "mean"),
          recompra_30d=("repeat_purchase_30d", "mean")
      )
      .reset_index()
)

comparacao_simples["grupo"] = comparacao_simples["is_detractor"].map({
    0: "Não detrator",
    1: "Detrator"
})

comparacao_simples["recompra_30d"] = comparacao_simples["recompra_30d"] * 100

display(comparacao_simples)
### Gráfico: detratores x não detratores

comparacao_plot = comparacao_simples.set_index("grupo")[[
    "media_contatos",
    "media_reclamacoes",
    "tempo_resolucao_medio",
    "csat_medio",
    "recompra_30d"
]].T

comparacao_plot.plot(kind="bar", figsize=(10, 4))

plt.title("Comparação entre detratores e não detratores")
plt.xlabel("Indicadores")
plt.ylabel("Média / percentual")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Grupo")
plt.tight_layout()
plt.show()
## 5.3 Score simples de risco de detrator

df["risco_atraso"] = (df["delivery_delay_days"] >= 3).astype(int)
df["risco_reclamacao"] = (df["complaints_count"] >= 3).astype(int)
df["risco_contato"] = (df["customer_service_contacts"] >= 2).astype(int)

df["score_risco"] = (
    df["risco_atraso"] +
    df["risco_reclamacao"] +
    df["risco_contato"]
)

resumo_p10 = (
    df.groupby("score_risco")
      .agg(
          qtd_clientes=("customer_id", "count"),
          nps_medio=("nps_score", "mean"),
          pct_detrator=("is_detractor", "mean")
      )
      .reset_index()
)

resumo_p10["pct_detrator"] = resumo_p10["pct_detrator"] * 100

display(resumo_p10)
### Gráfico: % de detratores por score de risco

plt.figure(figsize=(8, 4))

plt.bar(
    resumo_p10["score_risco"].astype(str),
    resumo_p10["pct_detrator"]
)

plt.title("% de detratores por score de risco")
plt.xlabel("Score de risco")
plt.ylabel("% de detratores")
plt.ylim(0, 100)

for i, valor in enumerate(resumo_p10["pct_detrator"]):
    plt.text(i, valor + 2, f"{valor:.1f}%", ha="center")

plt.show()
### Regra do score

print("Regras usadas no score:")
print("+1 se atraso na entrega for maior ou igual a 3 dias")
print("+1 se o cliente tiver 3 ou mais reclamações")
print("+1 se o cliente tiver 2 ou mais contatos com atendimento")

print("\nClassificação sugerida:")
print("Score 0: baixo risco")
print("Score 1: acompanhar")
print("Score 2: priorizar atendimento")
print("Score 3: ação urgente para evitar detrator")
## 5.4 Resumo numérico do EDA geral

print("Resumo final da EDA")
print("-" * 40)

print("Total de clientes analisados:", len(df))
print("NPS médio geral:", round(df["nps_score"].mean(), 2))
print("% de detratores:", round(df["is_detractor"].mean() * 100, 2), "%")
print("Média de contatos com atendimento:", round(df["customer_service_contacts"].mean(), 2))
print("Média de reclamações:", round(df["complaints_count"].mean(), 2))
print("Tempo médio de resolução:", round(df["resolution_time_days"].mean(), 2))
print("Atraso médio:", round(df["delivery_delay_days"].mean(), 2))
print("CSAT médio:", round(df["csat_internal_score"].mean(), 2))
print("% recompra em 30 dias:", round(df["repeat_purchase_30d"].mean() * 100, 2), "%")
## 5.5 Recomendações operacionais do EDA geral

1. Criar fila de recuperação para clientes com múltiplas reclamações.
2. Gerar alerta na segunda ou terceira interação com atendimento.
3. Tratar resolução longa como agravante da experiência.
4. Usar atraso de entrega como variável prioritária de investigação.
5. Não priorizar a tese regional como causa principal, salvo se novas bases mostrarem concentração específica.
6. Usar o score simples como primeira camada de priorização operacional.

# 6. Aprofundamento logístico e pedido

Esta seção aprofunda a análise nos fatores logísticos e de pedido que impactam o NPS, complementando os achados do EDA geral com foco em atraso de entrega, frete, tentativas e faturamento.
## 6.1 Preparação do ambiente para análise logística
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

from scipy.stats import chi2_contingency, kruskal

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)
## 6.2 Carregamento da base para aprofundamento logístico

Carregamento da base tratada para as análises de logística e pedido.
# -------------------------------------------------------------------------
# BASE DO APROFUNDAMENTO LOGÍSTICO / P3
# -------------------------------------------------------------------------
# Preferência: usar a base processada, caso exista.
# Se ela não existir, usa a base V1 e remove nps_score = 0.

try:
    path = localizar_arquivo(NOME_BASE_V2)
    arquivo_usado = path
    df = pd.read_csv(path)
    origem_base_p3 = "V2 processada"
except FileNotFoundError:
    path_v1 = localizar_arquivo(NOME_BASE_V1)
    arquivo_usado = path_v1
    df = pd.read_csv(path_v1)
    df = df[df["nps_score"] > 0].copy().reset_index(drop=True)
    origem_base_p3 = "V1 com nps_score = 0 removido"

print("Arquivo usado no P3:", arquivo_usado)
print("Origem da base P3:", origem_base_p3)
print("Linhas e colunas:", df.shape)

df.head()

df.shape
df.info()
df.isnull().sum()
df.duplicated().sum()
df.describe()
## 6.3 Distribuição do NPS na base
dist_nps = df["classificacao_nps"].value_counts(normalize=True) * 100
dist_nps = dist_nps.reindex(["Detrator", "Neutro", "Promotor"])

nps_oficial = dist_nps["Promotor"] - dist_nps["Detrator"]

print("Distribuição das categorias de NPS:")
print(dist_nps.round(2))
print(f"NPS oficial: {nps_oficial:.2f}")
sns.countplot(data=df, x="classificacao_nps", order=["Detrator", "Neutro", "Promotor"])
plt.title("Distribuição das Categorias de NPS")
plt.xlabel("Classificação NPS")
plt.ylabel("Quantidade de Clientes")
plt.show()
A distribuição da variável `classificacao_nps` permite observar a composição da base entre Detratores, Neutros e Promotores. Como esta versão da base remove os registros com `nps_score = 0`, a leitura dos percentuais deve considerar que a análise está sendo feita sobre a base.

A base ainda apresenta forte concentração de Detratores e NPS oficial negativo, mantendo um cenário crítico de satisfação mesmo após a remoção dos casos extremos.
## 6.4 Variáveis derivadas para logística e pedido

df["freight_ratio"] = np.where(df["order_value"] > 0, df["freight_value"] / df["order_value"], np.nan)
df["has_multiple_attempts"] = df["delivery_attempts"] > 1
df["teve_atraso"] = (df["delivery_delay_days"] > 0).astype(int)

max_delay_value = df["delivery_delay_days"].max()
bins = [-1, 0, 1, 3, 5, 10]
labels = ["Sem atraso", "Até 1 dia", "2 a 3 dias", "4 a 5 dias", "6 a 10 dias"]

if max_delay_value > 10:
    bins.append(max_delay_value)
    labels.append("Acima de 10 dias")

df["delay_range"] = pd.cut(df["delivery_delay_days"], bins=bins, labels=labels, include_lowest=True)

df[["delivery_delay_days", "delay_range", "freight_value", "order_value", "freight_ratio", "delivery_attempts", "has_multiple_attempts"]].head()
df["delay_range"].value_counts(dropna=False).sort_index()
df["has_multiple_attempts"].value_counts(normalize=True) * 100
Após a criação das variáveis derivadas, foi realizada uma validação para verificar se as faixas de atraso, o peso proporcional do frete e a marcação de múltiplas tentativas foram gerados corretamente.
## 6.5 Estatística descritiva de logística e pedido

logistics_cols = ["delivery_delay_days", "delivery_time_days", "freight_value", "freight_ratio", "order_value", "items_quantity", "delivery_attempts"]
estatistica_logistica = df.groupby("classificacao_nps")[logistics_cols].agg(["count", "mean", "median", "std", "min", "max"])
estatistica_logistica
A estatística descritiva por categoria de NPS permite comparar o comportamento das variáveis logísticas e de pedido entre Detratores, Neutros e Promotores.

A média mostra o comportamento geral de cada grupo, a mediana reduz o impacto de valores extremos, o desvio padrão indica a dispersão dos dados e os valores mínimo e máximo ajudam a observar possíveis outliers.
# 7. Atraso de entrega: principal aprofundamento

O objetivo é sair da leitura genérica de correlação e entender como o atraso de entrega afeta o NPS de forma concreta: por faixas, por ponto de ruptura e por impacto financeiro.
## 7.1 Atraso por categoria de NPS

atraso_por_nps = df.groupby("classificacao_nps")["delivery_delay_days"].agg(["count", "mean", "median", "std", "min", "max"])
atraso_por_nps
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="classificacao_nps", y="delivery_delay_days", order=["Detrator", "Neutro", "Promotor"])
plt.title("Atraso de Entrega por Categoria de NPS")
plt.xlabel("Classificação NPS")
plt.ylabel("Dias de Atraso")
plt.show()
A comparação do atraso de entrega por categoria de NPS permite observar se os clientes Detratores apresentam maior atraso em relação aos Neutros e Promotores.

Se Detratores apresentarem média, mediana ou dispersão superior, isso reforça a hipótese de que a experiência logística está associada à concentração de insatisfação.
## 7.2 Classificação NPS em pedidos com e sem atraso

delay_nps = pd.crosstab(df["teve_atraso"], df["classificacao_nps"], normalize="index") * 100
delay_nps = delay_nps.reindex(columns=["Detrator", "Neutro", "Promotor"])
delay_nps
delay_nps_plot = delay_nps.copy()
delay_nps_plot.index = delay_nps_plot.index.map({0: "Sem atraso", 1: "Com atraso"})
delay_nps_plot.plot(kind="bar", figsize=(8, 5))
plt.title("Distribuição Percentual de NPS em Pedidos com e sem Atraso")
plt.xlabel("Condição da Entrega")
plt.ylabel("% de Clientes")
plt.xticks(rotation=0)
plt.legend(title="Classificação NPS")
plt.show()
A tabela apresenta a distribuição percentual das categorias de NPS dentro dos grupos com e sem atraso. Essa é uma análise de probabilidade condicional, pois observa a probabilidade de cada classificação de NPS dado que o pedido teve ou não atraso.

O resultado não indica causalidade, mas mostra um padrão relevante para decisões de melhoria logística.
## 7.3 Ponto de ruptura no atraso

delay_range_nps = pd.crosstab(df["delay_range"], df["classificacao_nps"], normalize="index") * 100
delay_range_nps = delay_range_nps.reindex(columns=["Detrator", "Neutro", "Promotor"])
delay_range_nps
delay_range_nps.plot(kind="bar", figsize=(10, 6))
plt.title("Distribuição de NPS por Faixa de Atraso")
plt.xlabel("Faixa de Atraso")
plt.ylabel("% de Clientes")
plt.xticks(rotation=45)
plt.legend(title="Classificação NPS")
plt.tight_layout()
plt.show()
detratores_por_faixa = delay_range_nps["Detrator"].sort_index()
detratores_por_faixa
detratores_por_faixa.plot(kind="line", marker="o", figsize=(10, 5))
plt.title("Proporção de Detratores por Faixa de Atraso")
plt.xlabel("Faixa de Atraso")
plt.ylabel("% de Detratores")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
A análise por faixa de atraso permite observar se existe um ponto de ruptura na experiência logística. Esse ponto representa o intervalo em que a proporção de Detratores começa a aumentar de forma mais evidente.

Essa leitura é importante porque transforma o atraso em um limite operacional observável.
# 8. Sinais logísticos complementares e impacto financeiro

Este bloco analisa frete, tentativas de entrega e faturamento observado.

A leitura recomendada é separar o que reforça a hipótese principal do que serve apenas como sinal complementar.

## 8.1 Frete e peso proporcional do frete

frete_por_nps = df.groupby("classificacao_nps")["freight_value"].agg(["count", "mean", "median", "std", "min", "max"])
frete_por_nps
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="classificacao_nps", y="freight_value", order=["Detrator", "Neutro", "Promotor"])
plt.title("Valor do Frete por Categoria de NPS")
plt.xlabel("Classificação NPS")
plt.ylabel("Valor do Frete")
plt.show()
freight_ratio_por_nps = df.groupby("classificacao_nps")["freight_ratio"].agg(["count", "mean", "median", "std", "min", "max"])
freight_ratio_por_nps
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="classificacao_nps", y="freight_ratio", order=["Detrator", "Neutro", "Promotor"])
plt.title("Peso Proporcional do Frete no Pedido por Categoria de NPS")
plt.xlabel("Classificação NPS")
plt.ylabel("Frete / Valor do Pedido")
plt.show()
A análise do frete absoluto permite observar se o custo médio de entrega apresenta diferenças entre Detratores, Neutros e Promotores.

Já o `freight_ratio` complementa essa leitura ao mostrar quanto o frete representa proporcionalmente no valor total do pedido.
## 8.2 Tentativas de entrega

tentativas_por_nps = df.groupby("classificacao_nps")["delivery_attempts"].agg(["count", "mean", "median", "std", "min", "max"])
tentativas_por_nps
df["delivery_attempts"].value_counts().sort_index()
df["has_multiple_attempts"].value_counts(normalize=True) * 100
attempts_nps = pd.crosstab(df["has_multiple_attempts"], df["classificacao_nps"], normalize="index") * 100
attempts_nps = attempts_nps.reindex(columns=["Detrator", "Neutro", "Promotor"])
attempts_nps
attempts_nps_plot = attempts_nps.copy()
attempts_nps_plot.index = attempts_nps_plot.index.map({False: "Uma tentativa", True: "Múltiplas tentativas"})
attempts_nps_plot.plot(kind="bar", figsize=(8, 5))
plt.title("Distribuição de NPS por Múltiplas Tentativas de Entrega")
plt.xlabel("Condição de Tentativas")
plt.ylabel("% de Clientes")
plt.xticks(rotation=0)
plt.legend(title="Classificação NPS")
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="classificacao_nps", y="delivery_attempts", order=["Detrator", "Neutro", "Promotor"])
plt.title("Tentativas de Entrega por Categoria de NPS")
plt.xlabel("Classificação NPS")
plt.ylabel("Quantidade de Tentativas")
plt.show()
A análise de tentativas de entrega deve ser lida com cautela.

Conceitualmente, múltiplas tentativas podem indicar fricção logística, falha de comunicação, ausência do cliente, dificuldade de roteirização ou necessidade de reprocessamento operacional.

Porém, nesta base, a variável não aparece como um dos fatores mais discriminantes de NPS. Portanto, ela deve ser tratada como **sinal complementar**, e não como principal explicação para a insatisfação.

## 8.3 Faturamento por categoria de NPS

faturamento_nps = df.groupby("classificacao_nps").agg(
    total_pedidos=("order_value", "count"),
    faturamento_total=("order_value", "sum"),
    ticket_medio=("order_value", "mean"),
    mediana_pedido=("order_value", "median"),
    atraso_medio=("delivery_delay_days", "mean"),
    tentativas_medias=("delivery_attempts", "mean"),
    recompra_media_30d=("repeat_purchase_30d", "mean")
).reindex(["Detrator", "Neutro", "Promotor"])

faturamento_nps["participacao_faturamento_%"] = (faturamento_nps["faturamento_total"] / faturamento_nps["faturamento_total"].sum()) * 100

# Aplicar formatação de moeda para as colunas relevantes
formatted_faturamento_nps = faturamento_nps.copy()
for col in ["faturamento_total", "ticket_medio", "mediana_pedido"]:
    formatted_faturamento_nps[col] = formatted_faturamento_nps[col].apply(lambda x: f"R$ {x:,.2f}")

display(formatted_faturamento_nps)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="classificacao_nps", y="order_value", order=["Detrator", "Neutro", "Promotor"])
plt.title("Valor do Pedido por Categoria de NPS")
plt.xlabel("Classificação NPS")
plt.ylabel("Valor do Pedido")
plt.show()
ticket_medio_nps = df.groupby("classificacao_nps")["order_value"].mean().reindex(["Detrator", "Neutro", "Promotor"])
ax = ticket_medio_nps.plot(kind="bar", figsize=(8, 5))
plt.title("Ticket Médio por Categoria de NPS")
plt.xlabel("Classificação NPS")
plt.ylabel("Ticket Médio")
plt.xticks(rotation=0)
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("R$ {x:,.2f}"))
plt.tight_layout()
plt.show()
faturamento_total_nps = df.groupby("classificacao_nps")["order_value"].sum().reindex(["Detrator", "Neutro", "Promotor"])
ax = faturamento_total_nps.plot(kind="bar", figsize=(8, 5))
plt.title("Faturamento Total por Categoria de NPS")
plt.xlabel("Classificação NPS")
plt.ylabel("Faturamento Total")
plt.xticks(rotation=0)
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("R$ {x:,.2f}"))
plt.tight_layout()
plt.show()
A análise de faturamento por categoria de NPS permite observar como o valor financeiro dos pedidos está distribuído entre Detratores, Neutros e Promotores.

O faturamento total tende a ser influenciado pelo volume de clientes em cada categoria. Como a base possui maior concentração de Detratores, é esperado que esse grupo concentre parte relevante do faturamento total.
## 8.4 Faturamento associado ao atraso

df["teve_atraso"] = (df["delivery_delay_days"] > 0).astype(int)

faturamento_atraso = df.groupby("teve_atraso").agg(
    total_pedidos=("order_value", "count"),
    faturamento_total=("order_value", "sum"),
    ticket_medio=("order_value", "mean"),
    mediana_pedido=("order_value", "median"),
    atraso_medio=("delivery_delay_days", "mean"),
    tentativas_medias=("delivery_attempts", "mean"),
    recompra_media_30d=("repeat_purchase_30d", "mean")
)

faturamento_atraso["percentual_pedidos_%"] = (faturamento_atraso["total_pedidos"] / faturamento_atraso["total_pedidos"].sum()) * 100
faturamento_atraso["participacao_faturamento_%"] = (faturamento_atraso["faturamento_total"] / faturamento_atraso["faturamento_total"].sum()) * 100

faturamento_atraso_plot = faturamento_atraso.copy()
faturamento_atraso_plot.index = faturamento_atraso_plot.index.map({0: "Sem atraso", 1: "Com atraso"})
faturamento_atraso_plot = faturamento_atraso_plot.reindex(["Sem atraso", "Com atraso"])

# Aplicar formatação de moeda para as colunas relevantes
for col in ["faturamento_total", "ticket_medio", "mediana_pedido"]:
    faturamento_atraso_plot[col] = faturamento_atraso_plot[col].apply(lambda x: f"R$ {x:,.2f}")

display(faturamento_atraso_plot)
ax = faturamento_atraso["faturamento_total"].plot(kind="bar", figsize=(8, 5)) # Use o DataFrame original com dados numéricos
plt.title("Faturamento Observado por Condição de Atraso")
plt.xlabel("Condição da Entrega")
plt.ylabel("Faturamento Total")
plt.xticks(rotation=0)
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("R$ {x:,.2f}"))
for container in ax.containers:
    ax.bar_label(container, labels=[f"R$ {v:,.2f}" for v in container.datavalues], padding=3, fontsize=9)
plt.tight_layout()
plt.show()
ax = faturamento_atraso["ticket_medio"].plot(kind="bar", figsize=(8, 5))
plt.title("Ticket Médio por Condição de Atraso")
plt.xlabel("Condição da Entrega")
plt.ylabel("Ticket Médio")
plt.xticks(rotation=0)
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("R$ {x:,.2f}"))
for container in ax.containers:
    ax.bar_label(container, labels=[f"R$ {v:,.2f}" for v in container.datavalues], padding=3, fontsize=9)
plt.tight_layout()
plt.show()
q75_order_value = df["order_value"].quantile(0.75)
high_value_delayed = df[(df["order_value"] >= q75_order_value) & (df["teve_atraso"] == 1)]
high_value_delayed_summary = high_value_delayed.agg({
    "order_value": ["count", "sum", "mean", "median"],
    "delivery_delay_days": ["mean", "median"],
    "delivery_attempts": ["mean", "median"],
    "repeat_purchase_30d": ["mean"]
})
high_value_delayed_summary
from IPython.display import display, Markdown

total_high_value = df[df["order_value"] >= q75_order_value].shape[0]
total_high_value_delayed = high_value_delayed.shape[0]
percentual_high_value_delayed = (total_high_value_delayed / total_high_value) * 100

markdown_output = f"""
### Análise de Pedidos de Alto Valor com Atraso

*   **Total de Pedidos de Alto Valor:** {total_high_value}
*   **Pedidos de Alto Valor com Atraso:** {total_high_value_delayed}
*   **Percentual de Pedidos de Alto Valor com Atraso:** **{percentual_high_value_delayed:.2f}%**
"""
display(Markdown(markdown_output))
nps_high_value_delayed = (high_value_delayed["classificacao_nps"].value_counts(normalize=True) * 100)
nps_high_value_delayed = nps_high_value_delayed.reindex(["Detrator", "Neutro", "Promotor"])
nps_high_value_delayed
nps_high_value_delayed.plot(kind="bar", figsize=(8, 5))
plt.title("Distribuição de NPS para Pedidos de Alto Valor com Atraso")
plt.xlabel("Classificação NPS")
plt.ylabel("Percentual de Clientes")
plt.xticks(rotation=0)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.show()
A análise mostra que os pedidos com atraso concentram maior faturamento observado em comparação aos pedidos sem atraso.

Esse resultado não deve ser interpretado como se o atraso gerasse faturamento. A leitura correta é que parte relevante da receita está associada a pedidos que tiveram uma experiência logística pior.

O ponto central é que o atraso aparece em pedidos financeiramente relevantes. Isso torna a melhoria logística uma pauta não apenas de satisfação, mas também de preservação de faturamento, retenção e recompra.
# 9. Validações finais: região, correlação e teste estatístico

Este bloco fecha a análise logística com validações estatísticas.

A região entra como controle: ela não é um driver forte de NPS, mas precisa ser verificada para confirmar que os padrões observados não são efeitos regionais. A correlação e o teste estatístico consolidam a evidência quantitativa dos achados anteriores.
## 9.1 Análise regional logística

regional_logistics = df.groupby("customer_region").agg(
    total_pedidos=("order_value", "count"),
    atraso_medio=("delivery_delay_days", "mean"),
    atraso_mediana=("delivery_delay_days", "median"),
    tempo_entrega_medio=("delivery_time_days", "mean"),
    frete_medio=("freight_value", "mean"),
    peso_medio_frete=("freight_ratio", "mean"),
    tentativas_medias=("delivery_attempts", "mean"),
    ticket_medio=("order_value", "mean"),
    mediana_pedido=("order_value", "median"),
    faturamento_total=("order_value", "sum"),
    recompra_media_30d=("repeat_purchase_30d", "mean")
).sort_values("atraso_medio", ascending=False)
display(regional_logistics.round(2))
regional_logistics["participacao_faturamento_%"] = (regional_logistics["faturamento_total"] / regional_logistics["faturamento_total"].sum()) * 100
regional_logistics["participacao_pedidos_%"] = (regional_logistics["total_pedidos"] / regional_logistics["total_pedidos"].sum()) * 100
display(regional_logistics.round(2))
regional_nps = pd.crosstab(df["customer_region"], df["classificacao_nps"], normalize="index") * 100
regional_nps = regional_nps.reindex(columns=["Detrator", "Neutro", "Promotor"])

regional_summary = regional_logistics.join(regional_nps[["Detrator", "Neutro", "Promotor"]])
regional_summary = regional_summary.sort_values("Detrator", ascending=False)

# Apply currency formatting to relevant columns
formatted_regional_summary = regional_summary.copy()
for col in ["faturamento_total", "ticket_medio", "mediana_pedido"]:
    formatted_regional_summary[col] = formatted_regional_summary[col].apply(lambda x: f"R$ {x:,.2f}")

display(formatted_regional_summary[["total_pedidos", "participacao_pedidos_%", "faturamento_total", "participacao_faturamento_%", "ticket_medio", "atraso_medio", "tempo_entrega_medio", "frete_medio", "peso_medio_frete", "tentativas_medias", "Detrator", "Neutro", "Promotor"]].round(2))
ax = regional_nps.plot(kind="bar", figsize=(10, 6), stacked=True) # Use stacked bar for clarity
plt.title("Distribuição Percentual de NPS por Região", fontsize=14) # Increase title font size
plt.xlabel("Região", fontsize=12) # Increase x-label font size
plt.ylabel("% de Clientes", fontsize=12) # Increase y-label font size
plt.xticks(rotation=45, ha='right', fontsize=10) # Rotate and align x-axis labels
plt.yticks(fontsize=10)

# Add percentage labels to the bars
for container in ax.containers:
    labels = [f'{v.get_height():.1f}%' if v.get_height() > 0 else '' for v in container]
    ax.bar_label(container, labels=labels, label_type='center', fontsize=9, color='white') # Center labels, white color for contrast

plt.legend(title="Classificação NPS", bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside to prevent overlap
plt.tight_layout() # Adjust layout to prevent labels from being cut off
plt.show()
ax = regional_logistics["atraso_medio"].sort_values(ascending=False).plot(kind="bar", figsize=(10, 5))
plt.title("Atraso Médio por Região", fontsize=14) # Increase title font size
plt.xlabel("Região", fontsize=12) # Increase x-label font size
plt.ylabel("Atraso Médio em Dias", fontsize=12) # Increase y-label font size
plt.xticks(rotation=45, ha='right', fontsize=10) # Rotate and align x-axis labels
plt.yticks(fontsize=10)
for container in ax.containers:
    ax.bar_label(container, labels=[f"{v:.2f}" for v in container.datavalues], padding=3, fontsize=9)
plt.tight_layout() # Adjust layout to prevent labels from being cut off
plt.show()
ax = regional_logistics["faturamento_total"].sort_values(ascending=False).plot(kind="bar", figsize=(10, 5))
plt.title("Faturamento Total por Região")
plt.xlabel("Região")
plt.ylabel("Faturamento Total")
plt.xticks(rotation=45)
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("R$ {x:,.2f}"))
for container in ax.containers:
    ax.bar_label(container, labels=[f"R$ {v:,.0f}" for v in container.datavalues], padding=3, fontsize=8)
plt.tight_layout()
plt.show()
A análise regional permite observar se o comportamento logístico e a satisfação variam de forma relevante entre as regiões.

Ao comparar atraso médio, tempo de entrega, frete, tentativas, ticket médio, faturamento total e distribuição de NPS, conseguimos avaliar se o problema está concentrado em regiões específicas ou se aparece de forma mais sistêmica na operação.
## 9.2 Correlação entre NPS, logística e pedido

corr_cols = ["nps_score", "delivery_delay_days", "delivery_time_days", "freight_value", "freight_ratio", "order_value", "items_quantity", "delivery_attempts", "repeat_purchase_30d"]
corr_matrix = df[corr_cols].corr()
corr_matrix
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlação entre NPS, Logística e Pedido")
plt.show()
corr_nps = corr_matrix["nps_score"].drop("nps_score").sort_values()
corr_nps
ax = corr_nps.plot(kind="barh", figsize=(10, 6))
plt.title("Correlação das Variáveis com o NPS")
plt.xlabel("Correlação de Pearson")
plt.ylabel("Variáveis")
for container in ax.containers:
    ax.bar_label(container, labels=[f"{v:.2f}" for v in container.datavalues], padding=3, fontsize=9)
plt.tight_layout()
plt.show()
A matriz de correlação permite observar quais variáveis apresentam maior associação linear com o NPS.

Se `delivery_delay_days` apresentar correlação negativa relevante, isso reforça o achado de que o atraso de entrega está associado à menor satisfação. Caso `repeat_purchase_30d` apresente correlação positiva, isso indica que clientes com recompra em 30 dias tendem a apresentar maior NPS.

O comparativo V1 vs mostrou que a remoção dos registros com `nps_score = 0` não alterou a estrutura principal do problema.
## 9.3 Teste estatístico: atraso entre categorias de NPS

from IPython.display import display, Markdown
from scipy.stats import kruskal

grupo_detrator = df[df["classificacao_nps"] == "Detrator"]["delivery_delay_days"]
grupo_neutro = df[df["classificacao_nps"] == "Neutro"]["delivery_delay_days"]
grupo_promotor = df[df["classificacao_nps"] == "Promotor"]["delivery_delay_days"]

stat, p_value = kruskal(grupo_detrator, grupo_neutro, grupo_promotor)

alpha = 0.05

conclusion = ""
interpretation = ""
if p_value < alpha:
    conclusion = "rejeitamos a hipótese nula"
    interpretation = "há uma **diferença estatisticamente significativa** na distribuição dos dias de atraso de entrega entre pelo menos dois dos grupos de NPS (Detratores, Neutros, Promotores)."
else:
    conclusion = "não rejeitamos a hipótese nula"
    interpretation = "não há diferença estatisticamente significativa na distribuição dos dias de atraso de entrega entre os grupos de NPS."

markdown_output = f"""
### Resultados do Teste de Kruskal-Wallis

**Variável Analisada:** Atraso de Entrega (`delivery_delay_days`)

**Grupos Comparados:** Detratores, Neutros e Promotores (`classificacao_nps`)

-   **Estatística H (Kruskal-Wallis):** `{stat:.4f}`
-   **Valor-p:** `{p_value:.4f}`
-   **Nível de Significância (alpha):** `{alpha}`

---

**Conclusão:**

Com base no valor-p (`{p_value:.4f}`) ser {'menor' if p_value < alpha else 'maior'} que alpha (`{alpha}`), **{conclusion}**.

Isso sugere que {interpretation}
"""

display(Markdown(markdown_output))
from scipy.stats import mannwhitneyu
from itertools import combinations
from IPython.display import display, Markdown
import pandas as pd
import numpy as np

# Variável analisada
variavel = "delivery_delay_days"

# Grupos de NPS
grupos = {
    "Detrator": df[df["classificacao_nps"] == "Detrator"][variavel].dropna(),
    "Neutro": df[df["classificacao_nps"] == "Neutro"][variavel].dropna(),
    "Promotor": df[df["classificacao_nps"] == "Promotor"][variavel].dropna()
}

resultados = []

for grupo1, grupo2 in combinations(grupos.keys(), 2):
    dados_g1 = grupos[grupo1]
    dados_g2 = grupos[grupo2]

    stat, p_value = mannwhitneyu(
        dados_g1,
        dados_g2,
        alternative="two-sided"
    )

    media_g1 = dados_g1.mean()
    media_g2 = dados_g2.mean()
    mediana_g1 = dados_g1.median()
    mediana_g2 = dados_g2.median()

    resultados.append({
        "Comparação": f"{grupo1} vs {grupo2}",
        "N grupo 1": len(dados_g1),
        "N grupo 2": len(dados_g2),
        "Média grupo 1": media_g1,
        "Média grupo 2": media_g2,
        "Mediana grupo 1": mediana_g1,
        "Mediana grupo 2": mediana_g2,
        "Diferença mediana": mediana_g1 - mediana_g2,
        "Estatística U": stat,
        "p-valor": p_value
    })

resultado_posthoc = pd.DataFrame(resultados)

# Correção de Bonferroni para múltiplas comparações
n_comparacoes = len(resultado_posthoc)

resultado_posthoc["p-valor ajustado"] = (
    resultado_posthoc["p-valor"] * n_comparacoes
).clip(upper=1)

resultado_posthoc["Diferença significativa?"] = np.where(
    resultado_posthoc["p-valor ajustado"] < 0.05,
    "Sim",
    "Não"
)

# Interpretação automática simples
def interpretar_linha(row):
    if row["Diferença significativa?"] == "Não":
        return "Não há diferença estatisticamente significativa entre os grupos."

    grupo1, grupo2 = row["Comparação"].split(" vs ")

    if row["Diferença mediana"] > 0:
        return f"{grupo1} apresenta mediana de atraso maior que {grupo2}."
    elif row["Diferença mediana"] < 0:
        return f"{grupo2} apresenta mediana de atraso maior que {grupo1}."
    else:
        return "Há diferença estatística, mas as medianas são iguais; verificar distribuição e dispersão."

resultado_posthoc["Interpretação"] = resultado_posthoc.apply(interpretar_linha, axis=1)

display(Markdown("### Resultados dos Testes Post-Hoc de Mann-Whitney com Correção de Bonferroni"))

with pd.option_context("display.float_format", "{:.8f}".format):
    display(resultado_posthoc)
df.groupby("classificacao_nps")["delivery_delay_days"].agg(
    ["count", "mean", "median", "std", "min", "max"]
).reindex(["Detrator", "Neutro", "Promotor"])
O teste de Kruskal-Wallis indicou que existe diferença estatisticamente significativa na distribuição dos dias de atraso entre as categorias de NPS.

Como o teste não informa entre quais grupos ocorre a diferença, foi realizado um teste par a par de Mann-Whitney com correção de Bonferroni.

A partir das comparações par a par e das medidas descritivas, é possível identificar quais grupos apresentam diferenças significativas e qual categoria concentra maior atraso médio ou mediano.
# 10. Conclusão executiva consolidada

## Síntese

O EDA indica que a insatisfação do cliente está mais associada à experiência operacional do que ao perfil demográfico. Os fatores com maior impacto no NPS são:

- **Atraso de entrega** — principal driver negativo, com ponto de ruptura a partir de 3 dias
- **Reclamações recorrentes** — clientes com 5+ reclamações apresentam NPS médio abaixo de 3
- **Múltiplos contatos com atendimento** — sinal de fricção e resolução ineficaz

Região, idade e tempo de relacionamento apresentaram baixa associação com o NPS, indicando que a insatisfação é transversal ao perfil do cliente.

Os achados da análise exploratória fundamentam as features utilizadas na seção seguinte (Modelo Preditivo) e as recomendações operacionais propostas.
## 10.1 Relação com a etapa preditiva

## 10.2 Encaminhamentos sugeridos

1. Transformar o score simples em uma fila de priorização para atendimento/logística.
2. Criar monitoramento de clientes com atraso a partir de 1 dia.
3. Escalar clientes com atraso de 3 dias ou mais, especialmente se houver reclamações ou múltiplos contatos.
4. Separar, em próximas análises, clientes que não responderam NPS de clientes com nota realmente igual a zero.
5. Usar atraso, reclamações, contatos, resolução e recompra como candidatas principais para modelagem.

---
# 11. Modelo Preditivo de NPS
---

Nesta seção, construímos dois modelos a partir dos dados operacionais explorados nas seções anteriores:

1. **Modelo de Regressão** — estima a nota NPS em escala contínua (0 a 10)
2. **Modelo de Classificação** — categoriza clientes em **detrator** (NPS ≤ 6) vs **não-detrator** (NPS > 6)

A construção segue as seguintes premissas:
- `repeat_purchase_30d` foi **excluída** por leakage (ocorre após a experiência de compra)
- `csat_internal_score` foi testada em dois cenários (com e sem) para documentar possível leakage
- Features derivadas foram criadas com base nos pontos de ruptura identificados na EDA
- Desbalanceamento tratado com `class_weight='balanced'`
- Explicabilidade garantida via SHAP values
## 11.1 Importações e configuração
!pip install shap xgboost scikit-learn pandas numpy matplotlib seaborn -q
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, recall_score, precision_score, roc_curve,
    ConfusionMatrixDisplay
)

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')
## 11.2 Preparação dos dados para modelagem
### 11.2.1 Carregamento e criação da variável target
df = pd.read_csv('desafio_nps_fase_1.csv')

df['is_detractor'] = (df['nps_score'] <= 6).astype(int)

print(f'Shape: {df.shape}')
print(f'\nDistribuição da target de classificação:')
print(df['is_detractor'].value_counts())
print(f'\nDetratores: {df["is_detractor"].mean()*100:.1f}%')
print(f'NPS médio: {df["nps_score"].mean():.2f}')
### 11.2.2 Feature engineering

Variáveis criadas a partir dos pontos de ruptura identificados na análise exploratória:
df['atraso_critico'] = (df['delivery_delay_days'] >= 3).astype(int)
df['reclamacao_alta'] = (df['complaints_count'] >= 5).astype(int)
df['multiplos_contatos'] = (df['customer_service_contacts'] >= 3).astype(int)
df['atraso_x_reclamacao'] = df['delivery_delay_days'] * df['complaints_count']

print('Features derivadas criadas:')
print(f'  atraso_critico (delay >= 3 dias): {df["atraso_critico"].sum()} pedidos ({df["atraso_critico"].mean()*100:.1f}%)')
print(f'  reclamacao_alta (complaints >= 5): {df["reclamacao_alta"].sum()} pedidos ({df["reclamacao_alta"].mean()*100:.1f}%)')
print(f'  multiplos_contatos (contacts >= 3): {df["multiplos_contatos"].sum()} pedidos ({df["multiplos_contatos"].mean()*100:.1f}%)')
### 11.2.3 Seleção de features e separação treino/teste

**Excluídas:**
- `repeat_purchase_30d` — leakage (recompra ocorre após a experiência)
- `customer_id`, `order_id` — identificadores sem poder preditivo
- `nps_score`, `is_detractor` — variáveis target
features_base = [
    'delivery_time_days', 'delivery_delay_days', 'freight_value', 'delivery_attempts',
    'order_value', 'items_quantity', 'discount_value', 'payment_installments',
    'customer_service_contacts', 'resolution_time_days', 'complaints_count',
    'customer_age', 'customer_tenure_months',
    'atraso_critico', 'reclamacao_alta', 'multiplos_contatos', 'atraso_x_reclamacao'
]

df_model = pd.get_dummies(df, columns=['customer_region'], drop_first=True)
region_cols = [col for col in df_model.columns if col.startswith('customer_region_')]
features_all = features_base + region_cols

features_sem_csat = features_all.copy()
features_com_csat = features_all + ['csat_internal_score']

print(f'Features sem csat: {len(features_sem_csat)}')
print(f'Features com csat: {len(features_com_csat)}')
X = df_model[features_sem_csat]
y_reg = df_model['nps_score']
y_clf = df_model['is_detractor']

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print(f'Treino: {X_train.shape[0]} registros')
print(f'Teste:  {X_test.shape[0]} registros')
print(f'Detratores no treino: {y_clf_train.mean()*100:.1f}%')
print(f'Detratores no teste:  {y_clf_test.mean()*100:.1f}%')
## 11.3 Modelo de regressão

Objetivo: prever a nota NPS (0-10) com base nos dados operacionais.

Três algoritmos são comparados:
- **Ridge Regression** — regressão linear com regularização L2 (controla colinearidade entre features como `delivery_delay_days` e `delivery_time_days`)
- **Random Forest Regressor** — ensemble de árvores de decisão por bagging
- **XGBoost Regressor** — ensemble de árvores sequenciais por boosting

**Escolha dos hiperparâmetros:**
- `n_estimators=200`: quantidade suficiente para estabilizar as predições sem custo computacional excessivo em uma base de 2.500 registros
- `max_depth=8` (RF) e `max_depth=6` (XGBoost): limita a profundidade para evitar overfitting em dataset pequeno. XGBoost usa profundidade menor porque o boosting acumula complexidade a cada árvore
- `learning_rate=0.05` (XGBoost): taxa conservadora para aprendizado gradual e melhor generalização
- `subsample=0.8, colsample_bytree=0.8` (XGBoost): regularização adicional por amostragem de linhas e colunas
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_test_scaled)

rf_reg = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_reg_train)
y_pred_rf_reg = rf_reg.predict(X_test)

xgb_reg = XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
)
xgb_reg.fit(X_train, y_reg_train)
y_pred_xgb_reg = xgb_reg.predict(X_test)
def avaliar_regressao(nome, y_true, y_pred):
    return {
        'Modelo': nome,
        'MAE': round(mean_absolute_error(y_true, y_pred), 3),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        'R²': round(r2_score(y_true, y_pred), 3)
    }

resultados_reg = pd.DataFrame([
    avaliar_regressao('Ridge Regression', y_reg_test, y_pred_ridge),
    avaliar_regressao('Random Forest', y_reg_test, y_pred_rf_reg),
    avaliar_regressao('XGBoost', y_reg_test, y_pred_xgb_reg)
])

print('Comparação dos modelos de regressão')
print('=' * 55)
print(resultados_reg.to_string(index=False))
### Interpretação dos resultados da regressão

A tabela acima mostra o desempenho de cada modelo ao estimar a nota NPS:

- **MAE** indica o erro médio em pontos de NPS. Um MAE de 1.0, por exemplo, significa que o modelo erra em média 1 ponto na nota (ex: prevê 4.0 quando o real é 5.0).
- **RMSE** penaliza erros grandes. Se o RMSE for muito maior que o MAE, significa que o modelo tem alguns erros discrepantes.
- **R²** indica quanto da variação do NPS o modelo consegue explicar. Quanto mais próximo de 1, melhor.

Se os modelos baseados em árvore (Random Forest e XGBoost) superarem o Ridge, isso confirma que a relação entre as variáveis e o NPS é **não-linear** — coerente com os pontos de ruptura encontrados na EDA (ex: NPS despenca a partir de 3 dias de atraso, não de forma gradual).
cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []
for nome, modelo, X_cv in [
    ('Ridge', Ridge(alpha=1.0, random_state=42), X_train_scaled),
    ('Random Forest', RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1), X_train),
    ('XGBoost', XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), X_train)
]:
    scores = cross_val_score(modelo, X_cv, y_reg_train, cv=cv, scoring='r2')
    cv_results.append({'Modelo': nome, 'R² médio': f'{scores.mean():.3f}', 'Desvio': f'± {scores.std():.3f}'})

print('\nCross-Validation 5-Fold (Regressão)')
print(pd.DataFrame(cv_results).to_string(index=False))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, nome, y_pred in zip(axes, ['Ridge', 'Random Forest', 'XGBoost'],
                             [y_pred_ridge, y_pred_rf_reg, y_pred_xgb_reg]):
    ax.scatter(y_reg_test, y_pred, alpha=0.4, s=20, color='#4A90D9')
    ax.plot([0, 10], [0, 10], 'r--', linewidth=1.5, label='Predição perfeita')
    ax.set_xlabel('NPS Real')
    ax.set_ylabel('NPS Previsto')
    ax.set_title(f'{nome} — R² = {r2_score(y_reg_test, y_pred):.3f}')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.legend()

plt.suptitle('Regressão: NPS real vs NPS previsto', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
### Análise de resíduos

Os resíduos (erro = valor real - valor previsto) revelam **onde o modelo erra mais**. Um bom modelo deve ter resíduos distribuídos de forma aleatória em torno de zero, sem padrões visíveis.
# Análise de resíduos do melhor modelo (XGBoost)
residuos = y_reg_test - y_pred_xgb_reg

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Distribuição dos resíduos
axes[0].hist(residuos, bins=30, color='#4A90D9', edgecolor='white', alpha=0.8)
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
axes[0].set_xlabel('Resíduo (Real - Previsto)')
axes[0].set_ylabel('Frequência')
axes[0].set_title('Distribuição dos resíduos')

# 2. Resíduos vs valor previsto
axes[1].scatter(y_pred_xgb_reg, residuos, alpha=0.4, s=20, color='#E67E22')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('NPS Previsto')
axes[1].set_ylabel('Resíduo')
axes[1].set_title('Resíduos vs valor previsto')

# 3. Resíduos por faixa de NPS real
faixas = pd.cut(y_reg_test, bins=[0, 3, 6, 8, 10], labels=['0-3', '4-6', '7-8', '9-10'])
residuos_df = pd.DataFrame({'faixa': faixas, 'residuo_abs': np.abs(residuos)})
residuos_df.groupby('faixa')['residuo_abs'].mean().plot(kind='bar', ax=axes[2], color='#27AE60', edgecolor='white')
axes[2].set_xlabel('Faixa de NPS real')
axes[2].set_ylabel('Erro absoluto médio')
axes[2].set_title('Erro médio por faixa de NPS')
axes[2].tick_params(axis='x', rotation=0)

plt.suptitle('Análise de resíduos — XGBoost Regressão', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print(f'Erro médio absoluto geral: {np.abs(residuos).mean():.2f} pontos')
print(f'Desvio padrão dos resíduos: {residuos.std():.2f}')
print(f'\nO modelo erra mais nas notas extremas (muito baixas ou muito altas)?')
print(residuos_df.groupby('faixa')['residuo_abs'].agg(['mean', 'count']).rename(columns={'mean': 'erro_medio', 'count': 'n_registros'}))
**Leitura dos gráficos:**

- Se a distribuição dos resíduos está centrada em zero e simétrica, o modelo não tem viés sistemático.
- Se os resíduos no gráfico central formam um "funil" (mais dispersos em algumas faixas), o modelo tem **heterocedasticidade** — erra mais em certas faixas de NPS.
- O gráfico de barras mostra em quais faixas o modelo é mais ou menos preciso. Se o erro é maior nos extremos (notas muito baixas ou muito altas), é esperado: são os casos mais difíceis de prever.
## 11.4 Modelo de classificação

Objetivo: classificar clientes como **detrator** (NPS ≤ 6) ou **não-detrator** (NPS > 6).

Três algoritmos são comparados:
- **Regressão Logística** — classificador linear com `class_weight='balanced'` para compensar o desbalanceamento
- **Random Forest Classifier** — ensemble por bagging com `class_weight='balanced'`
- **XGBoost Classifier** — ensemble por boosting com `scale_pos_weight` ajustado

**Tratamento do desbalanceamento (74% detratores):**

Sem tratamento, o modelo poderia simplesmente classificar todos como detrator e acertar 74% — uma acurácia alta mas inútil. O `class_weight='balanced'` faz o modelo prestar mais atenção nos não-detratores durante o treinamento, equilibrando o aprendizado.

**Por que foco em Recall:**

Na perspectiva do negócio, é mais caro deixar passar um detrator (que pode gerar churn e boca a boca negativo) do que disparar uma ação preventiva para um cliente que não seria detrator. Por isso, priorizamos o Recall (capturar o máximo de detratores reais).
scale_pos = (y_clf_train == 0).sum() / (y_clf_train == 1).sum()

log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_clf_train)
y_pred_log = log_reg.predict(X_test_scaled)
y_prob_log = log_reg.predict_proba(X_test_scaled)[:, 1]

rf_clf = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1
)
rf_clf.fit(X_train, y_clf_train)
y_pred_rf_clf = rf_clf.predict(X_test)
y_prob_rf_clf = rf_clf.predict_proba(X_test)[:, 1]

xgb_clf = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1/scale_pos,
    random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss'
)
xgb_clf.fit(X_train, y_clf_train)
y_pred_xgb_clf = xgb_clf.predict(X_test)
y_prob_xgb_clf = xgb_clf.predict_proba(X_test)[:, 1]
def avaliar_classificacao(nome, y_true, y_pred, y_prob):
    return {
        'Modelo': nome,
        'Recall': round(recall_score(y_true, y_pred), 3),
        'Precisão': round(precision_score(y_true, y_pred), 3),
        'F1': round(f1_score(y_true, y_pred), 3),
        'AUC-ROC': round(roc_auc_score(y_true, y_prob), 3)
    }

resultados_clf = pd.DataFrame([
    avaliar_classificacao('Regressão Logística', y_clf_test, y_pred_log, y_prob_log),
    avaliar_classificacao('Random Forest', y_clf_test, y_pred_rf_clf, y_prob_rf_clf),
    avaliar_classificacao('XGBoost', y_clf_test, y_pred_xgb_clf, y_prob_xgb_clf)
])

print('Comparação dos modelos de classificação')
print('=' * 60)
print(resultados_clf.to_string(index=False))
### Interpretação dos resultados da classificação

A tabela mostra como cada modelo se comporta ao separar detratores de não-detratores:

- **Recall alto** = o modelo captura a maioria dos detratores reais (poucos escapam)
- **Precisão alta** = quando o modelo diz "detrator", geralmente acerta (poucos alarmes falsos)
- **F1 alto** = bom equilíbrio entre recall e precisão
- **AUC-ROC alto** = boa capacidade geral de discriminação

Para este caso de uso, um **Recall acima de 0.85** é desejável — significa que o modelo captura pelo menos 85% dos clientes que realmente se tornariam detratores, viabilizando ação preventiva.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, nome, y_pred in zip(axes,
    ['Regressão Logística', 'Random Forest', 'XGBoost'],
    [y_pred_log, y_pred_rf_clf, y_pred_xgb_clf]):
    cm = confusion_matrix(y_clf_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Não-Detrator', 'Detrator']).plot(ax=ax, cmap='Blues')
    ax.set_title(nome, fontweight='bold')

plt.suptitle('Matrizes de confusão', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(8, 6))

for nome, y_prob, color in [
    ('Regressão Logística', y_prob_log, '#4A90D9'),
    ('Random Forest', y_prob_rf_clf, '#E67E22'),
    ('XGBoost', y_prob_xgb_clf, '#27AE60')
]:
    fpr, tpr, _ = roc_curve(y_clf_test, y_prob)
    auc = roc_auc_score(y_clf_test, y_prob)
    ax.plot(fpr, tpr, label=f'{nome} (AUC = {auc:.3f})', color=color, linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatório (AUC = 0.500)')
ax.set_xlabel('Taxa de falso positivo')
ax.set_ylabel('Taxa de verdadeiro positivo (Recall)')
ax.set_title('Curvas ROC', fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()
### Ajuste de threshold (ponto de corte)

Por padrão, o modelo classifica como detrator se a probabilidade prevista for > 0.5. Mas esse corte não é necessariamente o melhor.

Reduzir o threshold (ex: 0.4 ou 0.3) aumenta o Recall (captura mais detratores), mas também gera mais falsos positivos. Aumentar o threshold faz o oposto.

A curva abaixo mostra esse trade-off para o XGBoost:
# Análise de threshold — XGBoost Classificação
thresholds = np.arange(0.1, 0.9, 0.05)

results = []
for t in thresholds:
    y_pred_t = (y_prob_xgb_clf >= t).astype(int)
    results.append({
        'Threshold': round(t, 2),
        'Recall': recall_score(y_clf_test, y_pred_t),
        'Precisão': precision_score(y_clf_test, y_pred_t, zero_division=0),
        'F1': f1_score(y_clf_test, y_pred_t, zero_division=0)
    })

df_thresh = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_thresh['Threshold'], df_thresh['Recall'], label='Recall', color='#E74C3C', linewidth=2)
ax.plot(df_thresh['Threshold'], df_thresh['Precisão'], label='Precisão', color='#4A90D9', linewidth=2)
ax.plot(df_thresh['Threshold'], df_thresh['F1'], label='F1-Score', color='#27AE60', linewidth=2)
ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, label='Threshold padrão (0.5)')

# Marcar o threshold que maximiza F1
best_f1_idx = df_thresh['F1'].idxmax()
best_t = df_thresh.loc[best_f1_idx, 'Threshold']
best_f1 = df_thresh.loc[best_f1_idx, 'F1']
ax.axvline(x=best_t, color='#E67E22', linestyle='--', linewidth=1.5, label=f'Melhor F1 (threshold={best_t})')

ax.set_xlabel('Threshold')
ax.set_ylabel('Métrica')
ax.set_title('Trade-off Recall vs Precisão por Threshold — XGBoost', fontweight='bold')
ax.legend()
ax.set_xlim(0.1, 0.85)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()

print(f'Threshold que maximiza F1: {best_t}')
print(f'F1 nesse threshold: {best_f1:.3f}')
print(f'Recall nesse threshold: {df_thresh.loc[best_f1_idx, "Recall"]:.3f}')
print(f'Precisão nesse threshold: {df_thresh.loc[best_f1_idx, "Precisão"]:.3f}')
**Leitura do gráfico:**

- A curva vermelha (Recall) cai conforme o threshold aumenta — o modelo fica mais exigente para classificar como detrator
- A curva azul (Precisão) sobe — menos alarmes falsos, mas mais detratores escapam
- O ponto onde o F1 (verde) é máximo representa o melhor equilíbrio

**Para aplicação no negócio:** se o custo de perder um detrator for alto (churn, boca a boca negativo), pode valer usar um threshold mais baixo que o ótimo de F1, aceitando mais falsos positivos em troca de capturar praticamente todos os detratores.
# Comparação: threshold padrão (0.5) vs threshold otimizado
y_pred_default = (y_prob_xgb_clf >= 0.5).astype(int)
y_pred_optimized = (y_prob_xgb_clf >= best_t).astype(int)

print('XGBoost — Comparação de thresholds')
print('=' * 55)
print(f'\nThreshold 0.50 (padrão):')
print(f'  Recall: {recall_score(y_clf_test, y_pred_default):.3f} | Precisão: {precision_score(y_clf_test, y_pred_default):.3f} | F1: {f1_score(y_clf_test, y_pred_default):.3f}')
print(f'\nThreshold {best_t} (otimizado por F1):')
print(f'  Recall: {recall_score(y_clf_test, y_pred_optimized):.3f} | Precisão: {precision_score(y_clf_test, y_pred_optimized):.3f} | F1: {f1_score(y_clf_test, y_pred_optimized):.3f}')

# Matriz de confusão com threshold otimizado
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(confusion_matrix(y_clf_test, y_pred_default), display_labels=['Não-Detrator', 'Detrator']).plot(ax=axes[0], cmap='Blues')
axes[0].set_title(f'Threshold = 0.50 (padrão)', fontweight='bold')
ConfusionMatrixDisplay(confusion_matrix(y_clf_test, y_pred_optimized), display_labels=['Não-Detrator', 'Detrator']).plot(ax=axes[1], cmap='Oranges')
axes[1].set_title(f'Threshold = {best_t} (otimizado)', fontweight='bold')
plt.suptitle('Impacto do ajuste de threshold na matriz de confusão', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_clf_results = []
for nome, modelo, X_cv in [
    ('Logística', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42), X_train_scaled),
    ('Random Forest', RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1), X_train),
    ('XGBoost', XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1/scale_pos, random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss'), X_train)
]:
    f1_scores = cross_val_score(modelo, X_cv, y_clf_train, cv=cv_strat, scoring='f1')
    auc_scores = cross_val_score(modelo, X_cv, y_clf_train, cv=cv_strat, scoring='roc_auc')
    cv_clf_results.append({
        'Modelo': nome,
        'F1 médio': f'{f1_scores.mean():.3f} ± {f1_scores.std():.3f}',
        'AUC médio': f'{auc_scores.mean():.3f} ± {auc_scores.std():.3f}'
    })

print('Cross-Validation 5-Fold (Classificação)')
print(pd.DataFrame(cv_clf_results).to_string(index=False))
## 11.5 Comparação: cenário com e sem `csat_internal_score`

A variável `csat_internal_score` apresenta alta correlação com NPS (+0.56), mas sua temporalidade de coleta é ambígua. Para documentar o impacto, treinamos o XGBoost nos dois cenários e comparamos as métricas.
X_csat = df_model[features_com_csat]

X_train_c, X_test_c, y_reg_train_c, y_reg_test_c, y_clf_train_c, y_clf_test_c = train_test_split(
    X_csat, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

xgb_reg_csat = XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
)
xgb_reg_csat.fit(X_train_c, y_reg_train_c)
y_pred_reg_c = xgb_reg_csat.predict(X_test_c)

xgb_clf_csat = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1/scale_pos,
    random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss'
)
xgb_clf_csat.fit(X_train_c, y_clf_train_c)
y_pred_clf_c = xgb_clf_csat.predict(X_test_c)
y_prob_clf_c = xgb_clf_csat.predict_proba(X_test_c)[:, 1]

print('XGBoost: cenário SEM vs COM csat_internal_score')
print('=' * 60)
print(f'\nREGRESSÃO')
print(f'  SEM csat — MAE: {mean_absolute_error(y_reg_test, y_pred_xgb_reg):.3f} | RMSE: {np.sqrt(mean_squared_error(y_reg_test, y_pred_xgb_reg)):.3f} | R²: {r2_score(y_reg_test, y_pred_xgb_reg):.3f}')
print(f'  COM csat — MAE: {mean_absolute_error(y_reg_test_c, y_pred_reg_c):.3f} | RMSE: {np.sqrt(mean_squared_error(y_reg_test_c, y_pred_reg_c)):.3f} | R²: {r2_score(y_reg_test_c, y_pred_reg_c):.3f}')
print(f'\nCLASSIFICAÇÃO')
print(f'  SEM csat — Recall: {recall_score(y_clf_test, y_pred_xgb_clf):.3f} | F1: {f1_score(y_clf_test, y_pred_xgb_clf):.3f} | AUC: {roc_auc_score(y_clf_test, y_prob_xgb_clf):.3f}')
print(f'  COM csat — Recall: {recall_score(y_clf_test_c, y_pred_clf_c):.3f} | F1: {f1_score(y_clf_test_c, y_pred_clf_c):.3f} | AUC: {roc_auc_score(y_clf_test_c, y_prob_clf_c):.3f}')
## 11.6 Explicabilidade com SHAP

SHAP (SHapley Additive exPlanations) quantifica a contribuição de cada variável para cada predição individual, permitindo entender não apenas *quais* fatores importam, mas *como* eles influenciam o resultado.
explainer_reg = shap.TreeExplainer(xgb_reg)
shap_values_reg = explainer_reg.shap_values(X_test)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_reg, X_test, show=False, plot_size=(10, 8))
plt.title('SHAP — Regressão (XGBoost)', fontweight='bold', pad=20)
plt.tight_layout()
plt.show()
explainer_clf = shap.TreeExplainer(xgb_clf)
shap_values_clf = explainer_clf.shap_values(X_test)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_clf, X_test, show=False, plot_size=(10, 8))
plt.title('SHAP — Classificação (XGBoost)', fontweight='bold', pad=20)
plt.tight_layout()
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

plt.sca(axes[0])
shap.summary_plot(shap_values_reg, X_test, plot_type='bar', show=False, max_display=10)
axes[0].set_title('Top 10 features — Regressão', fontweight='bold')

plt.sca(axes[1])
shap.summary_plot(shap_values_clf, X_test, plot_type='bar', show=False, max_display=10)
axes[1].set_title('Top 10 features — Classificação', fontweight='bold')

plt.tight_layout()
plt.show()
## 11.7 Conclusões e recomendações
### 11.7.1 Resumo dos resultados

O modelo de **regressão** estima a nota NPS de cada cliente antes da pesquisa. Isso permite criar um ranking contínuo de risco: quanto menor a nota prevista, maior a urgência de intervenção. A análise de resíduos mostra em quais faixas de NPS o modelo é mais ou menos confiável.

O modelo de **classificação** responde a uma pergunta direta: esse cliente vai ser detrator ou não? O output é uma probabilidade (0 a 1) que alimenta diretamente uma fila de priorização operacional. O ajuste de threshold permite calibrar o modelo conforme o apetite de risco da empresa — se prefere capturar mais detratores (recall alto) ou reduzir alarmes falsos (precisão alta).

A comparação entre cenários com e sem `csat_internal_score` documenta o quanto o modelo depende dessa variável. Se a performance cai pouco sem ela, confirma que os dados operacionais (atraso, reclamações, contatos) são suficientes para a predição — o que é mais robusto para produção.
### 11.7.2 Aplicação prática do modelo

O modelo de classificação gera uma **probabilidade de detrator** para cada pedido. Essa probabilidade pode ser usada diretamente pela operação:

**Fila de priorização automática:**
- Pedidos com probabilidade > threshold otimizado entram na fila de ação preventiva
- A operação recebe diariamente uma lista ordenada por risco, do maior para o menor
- Isso substitui a lógica atual de só agir depois da pesquisa de NPS

**Ações direcionadas pelo SHAP:**

O SHAP mostra quais variáveis mais pesam na predição de cada cliente individual. Isso permite direcionar a ação correta:
- Se o SHAP aponta `delivery_delay_days` como principal fator para aquele cliente → logística aciona comunicação proativa e rastreamento prioritário
- Se o SHAP aponta `complaints_count` → atendimento escalona o caso para resolução imediata
- Se o SHAP aponta `customer_service_contacts` → célula de recuperação entra em contato para resolver a fricção acumulada

**Calibração contínua:**
- Comparar periodicamente o NPS predito vs NPS real coletado na pesquisa
- Se a diferença aumentar, recalibrar o modelo com dados mais recentes
- O threshold pode ser ajustado conforme a capacidade operacional (quantos casos a equipe consegue tratar por dia)
### 11.7.3 Limitações e próximos passos

**Limitações do modelo atual:**
- Base com 2.500 registros — o modelo pode não generalizar para volumes significativamente maiores ou para perfis de cliente que não estão representados na amostra
- 74% de detratores na base — se essa proporção mudar no futuro (ex: empresa melhorar a operação), o modelo precisa ser retreinado
- Temporalidade de `csat_internal_score` não foi validada com a área de negócio — se for coletada junto ou após o NPS, o cenário sem essa variável é o correto para produção
- O modelo assume que os padrões históricos se mantêm — mudanças operacionais (novo parceiro logístico, nova política de atendimento) podem invalidar as predições

**Próximos passos para colocar em produção:**
- Validar com as áreas de negócio o momento exato de coleta de cada variável (garantir que todas estão disponíveis antes da pesquisa)
- Aplicar validação temporal: treinar com dados de um período e testar com dados do período seguinte
- Implementar monitoramento de drift para detectar quando o modelo começa a perder precisão
- Estimar o ROI: custo de uma intervenção preventiva vs custo estimado de perder um cliente detrator (churn, boca a boca negativo)
- Conduzir teste A/B: aplicar o modelo em um grupo de clientes e comparar o NPS resultante com um grupo controle sem intervenção
