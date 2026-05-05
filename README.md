
# 📊 EDA e Modelo Preditivo — NPS E-commerce

Análise exploratória de dados e modelos preditivos para o Net Promoter Score (NPS) de um e-commerce, com foco em identificar os drivers operacionais de insatisfação e antecipar detratores antes da pesquisa de NPS.

---

## 📁 Estrutura do projeto

```
├── desafio_nps_fase_1.csv                       # Base de dados original (2.500 pedidos)
├── EDA_Modelo_Preditivo_NPS_Final_CORRIGIDO.ipynb  # Notebook principal (corrigido)
├── RESUMO_EXECUTIVO_NPS.md                      # Resumo executivo com achados e conclusões
└── README.md                                    # Este arquivo
```

---

## 🎯 Pergunta central

> **Quais fatores operacionais mais influenciam a satisfação do cliente e como a empresa pode agir de forma proativa para melhorar a experiência antes da pesquisa de NPS?**

---

## 📋 Estrutura do notebook

| Seção | Conteúdo |
|-------|----------|
| **0** | Preparação, carregamento e variáveis auxiliares |
| **1** | Diagnóstico inicial do NPS (distribuição, score oficial) |
| **2** | Drivers de atendimento: contatos, reclamações, resolução, CSAT |
| **3** | Ranking dos fatores por força de associação |
| **4** | Análises complementares: região, idade, tempo de relacionamento |
| **5** | Score simples de risco de detrator e recomendações operacionais |
| **6** | Aprofundamento logístico — carregamento e variáveis derivadas |
| **7** | Atraso de entrega: ponto de ruptura e impacto por faixa |
| **8** | Sinais logísticos complementares: frete, tentativas, faturamento |
| **9** | Validações finais: região, correlação, testes Kruskal-Wallis e Mann-Whitney |
| **10** | Conclusão executiva consolidada |
| **11** | Modelos preditivos: regressão (nota NPS) e classificação (detrator) |

---

## 🤖 Modelos preditivos (seção 11)

### Regressão — estima a nota NPS (0–10)

| Modelo | Descrição |
|--------|-----------|
| Ridge | Regressão linear com regularização L2 (controla colinearidade) |
| Random Forest | Ensemble de árvores por bagging |
| XGBoost | Ensemble sequencial com boosting |

**Métricas:** MAE (erro médio em pontos), RMSE (penaliza erros grandes), R² (variância explicada).

### Classificação — identifica detratores (NPS ≤ 6)

| Modelo | Descrição |
|--------|-----------|
| Regressão Logística | Baseline linear |
| Random Forest | Ensemble de árvores por bagging |
| XGBoost | Ensemble sequencial com boosting |

**Métricas:** Recall, Precisão, F1-score, AUC-ROC.
**Balanceamento:** `class_weight='balanced'` — necessário pois 74% da base é detrator.

### Extras
- Matriz de confusão e curva ROC
- Análise de resíduos (onde o modelo de regressão erra mais)
- Threshold tuning (trade-off recall vs precisão)
- SHAP values (importância real das features)

---

## ⚙️ Decisões de modelagem

| Decisão | Justificativa |
|---------|---------------|
| `repeat_purchase_30d` **excluída** | Leakage — recompra ocorre após a experiência de compra |
| `csat_internal_score` testada **com e sem** | Correlação r = 0,564 com NPS; possível leakage dependendo do timing de coleta. Modelo principal usa versão **sem** CSAT. Não usar com CSAT em produção sem confirmar timing. |
| `delivery_attempts` mantida como feature | Correlação com NPS r = 0,028 (fraca). SHAP confirmará baixo peso. Mantida para não enviesar seleção manual. |
| Features derivadas criadas | `atraso_critico` (≥ 3 dias), `reclamacao_alta` (≥ 5), `multiplos_contatos` (≥ 3) — ancoradas nos pontos de ruptura da EDA |
| Interação `atraso_x_reclamacao` | Captura efeito combinado dos dois principais drivers |

---

## 🗒️ Correções aplicadas nesta versão

1. **Bug de variável** — `classificacao_nps` nunca era definida nas seções 6–10 (causaria `NameError`). Adicionado alias `df["classificacao_nps"] = df["nps_category"]` na seção 0.5 e recriação completa após o reload da base na seção 6.2.
2. **Boxplot de tentativas de entrega** — substituído por gráfico de distribuição proporcional com anotação da correlação r = 0,028. Boxplot era inadequado para variável com apenas 3 valores únicos.
3. **Nota metodológica** — adicionada célula explicando por que `delivery_attempts` é sinal complementar e não driver de NPS.
4. **Aviso de leakage CSAT** — adicionado comentário explícito na célula de seleção de features da seção 11.

---

## 🚀 Como rodar

1. Coloque `desafio_nps_fase_1.csv` na mesma pasta do notebook
2. Abra no VS Code com a extensão Jupyter
3. Execute célula por célula com `Shift + Enter`
4. Dependências: `pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost shap`

---

## 📌 Contexto da análise

- **Base:** 2.500 pedidos com dados de logística, atendimento, perfil do cliente e NPS
- **Escopo:** EDA exploratória (seções 0–10) + modelagem preditiva (seção 11)
- **Foco:** Análise operacional — não demográfica
- **Público-alvo:** Times de operações, logística e atendimento; gestão sênior

---

*Nota: a seção de modelagem foi construída com suporte de IA e está documentada para revisão por especialistas em ML antes de uso em ambiente de produção.*
=======
# Modelo Final em Produção

Notebook de produção para o projeto de análise e modelagem preditiva de NPS. Desenvolvido a partir dos resultados do `EDA_Modelo_Preditivo_NPS_Final.ipynb`, com foco em transformar a predição em uma ferramenta operacional de priorização preventiva.

---

## Problema de negócio

A base analisada apresenta **74% dos clientes como detratores** (NPS de 6 ou menos). Esse nível de concentração não é uma exceção, é um padrão recorrente na operação.

A insatisfação não é explicada por perfil demográfico (região, idade, tempo de relacionamento). Ela é explicada pela **jornada operacional**, o que significa que pode ser antecipada e tratada antes de se consolidar.

**Três drivers com maior impacto negativo no NPS** (validados com Kruskal-Wallis):

| Driver | Ponto de ruptura |
|---|---|
| Atraso de entrega | 3 dias ou mais |
| Reclamações recorrentes | 5 reclamações ou mais |
| Múltiplos contatos com atendimento | 3 contatos ou mais |

---

## Solução proposta

Usar o modelo preditivo como uma camada de inteligência operacional para antecipar clientes com maior probabilidade de se tornarem detratores, transformando o NPS de um indicador retrospectivo em uma ferramenta de priorização preventiva.

**Fluxo de produção:**

1. Capturar dados operacionais da jornada do pedido
2. Calcular a probabilidade de o cliente se tornar detrator
3. Classificar o cliente em faixas de risco (Crítico / Alto / Médio / Baixo)
4. Acionar intervenções preventivas de acordo com a criticidade
5. Monitorar impacto no NPS e na retenção

---

## Estrutura do notebook

| Seção | Conteúdo |
|---|---|
| 1. Setup e Configuração | Imports, XGBoost, SHAP, utilitários |
| 2. Carregamento dos Dados | CSV: 2.500 registros (`desafio_nps_fase_1.csv`) |
| 3. Preparação dos Dados | Target `is_detractor`, feature engineering com 5 variáveis derivadas |
| 4. Split Holdout (75/25) | Split estratificado único + StratifiedKFold 5 folds sobre o treino |
| 5. Pipeline e Treinamento | RF vs XGBoost, validação cruzada estratificada |
| 6. Avaliação do Modelo | AUC-ROC, curvas ROC e Precision-Recall no holdout |
| 6.5 Threshold Ótimo + SHAP | Threshold via F2-Score, SHAP top 15 features |
| 6.6 Comparação com/sem CSAT | Documenta impacto do `csat_internal_score` |
| 7. Score de Risco e Faixas | Probabilidade convertida em score % e faixas Crítico / Alto / Médio / Baixo |
| 8. Uso Operacional | Ranking por risco, ações recomendadas, simulação de capacidade diária |
| 9. Análise Operacional de Frete | Faixas por quartil, regras de decisão frete x atraso x score |
| 10. ROI e Decisão Executiva | R$ 15/intervenção, R$ 120/perda, 20% sucesso |
| 11. Monitoramento e Evolução | Drift de variáveis, proporção de detratores, gatilhos de retreinamento |
| 12. Teste A/B e Inferência | Proporção + Mann-Whitney, 18% de recuperação dos detratores tratados |
| 13. Impacto Financeiro e Reputacional | Receita em risco, benefícios com 35% de melhoria nos grupos críticos |
| 14. Conclusão Executiva | Frentes de atuação e síntese do impacto |

---

## Decisões de modelagem

**Variáveis excluídas (risco de leakage):**

| Variável | Motivo |
|---|---|
| `repeat_purchase_30d` | Recompra ocorre após a resposta ao NPS |
| `csat_internal_score` | Momento de coleta não validado com o negócio |
| `nps_score` / `classificacao_nps` / `is_detractor` | Variáveis target |

**Feature engineering (pontos de ruptura da EDA):**

| Variável | Critério |
|---|---|
| `atraso_critico` | `delivery_delay_days >= 3` |
| `reclamacao_alta` | `complaints_count >= 5` |
| `multiplos_contatos` | `customer_service_contacts >= 3` |
| `atraso_x_reclamacao` | `delivery_delay_days x complaints_count` |
| `freight_ratio` | `freight_value / order_value` |

**Modelo selecionado:** XGBoost (melhor AUC/F1 na validação cruzada e no holdout)

**Threshold otimizado via F2-Score:** prioriza recall, pois identificar detratores é mais importante do que evitar falsos positivos.

---

## Estratégia de validação

A validação adota dois níveis complementares para garantir que a avaliação de performance seja robusta e sem contaminação de dados.

**Nível 1 — Holdout único 75/25**

O conjunto de dados é dividido uma única vez em treino (75%) e teste (25%) com estratificação, garantindo que a proporção de detratores se mantém igual nos dois conjuntos. O conjunto de teste fica isolado e não é tocado durante nenhuma etapa de treinamento ou ajuste.

**Nível 2 — StratifiedKFold (5 folds) sobre o treino**

O conjunto de treino é dividido em 5 partes iguais e estratificadas. A cada iteração, 4 partes são usadas para treinar e 1 para validar, rotacionando até que todos os subconjuntos sirvam como validação. Isso produz 5 estimativas de performance que, combinadas, dão uma visão estável do comportamento do modelo sem usar o holdout.

| Etapa | Conjunto utilizado | Objetivo |
|---|---|---|
| Cross-validation (5 folds) | 75% do treino | Comparar modelos e detectar overfitting |
| Treinamento final | 100% do treino (75%) | Treinar com o máximo de dados disponível |
| Avaliação final | Holdout (25%) | Medir performance real em dados nunca vistos |

---

## Faixas de risco e ações recomendadas

| Faixa | Critério | Ação |
|---|---|---|
| Crítico | Score acima de 80% | Contato humano prioritário e análise imediata do pedido |
| Alto | Score entre 60% e 79% | Contato preventivo e acompanhamento da jornada |
| Médio | Score entre 40% e 59% | Comunicação automatizada e monitoramento |
| Baixo | Score abaixo de 40% | Fluxo padrão sem intervenção imediata |

---

## Monitoramento em produção

**Indicadores monitorados:**

- Proporção de detratores reais por período
- Score médio de risco por período
- Distribuição das variáveis operacionais (`delivery_delay_days`, `complaints_count`, `customer_service_contacts`, `resolution_time_days`)

**Gatilhos de retreinamento:**

- Queda relevante no recall de detratores
- Variação acima de 10 p.p. na proporção de detratores
- Mudança abrupta no score médio de risco
- Alteração nos padrões das variáveis operacionais
- Mudança operacional relevante (novo parceiro logístico, nova política de atendimento)
- Acúmulo de nova base mensal ou trimestral

---

## Dependências

```
pandas
numpy
matplotlib
scikit-learn
xgboost
shap
scipy
```

---

## Arquivo de dados

```
desafio_nps_fase_1.csv
```

2.500 registros com variáveis operacionais da jornada do pedido. Target: `nps_score` (detrator = NPS de 6 ou menos).
>>>>>>> hotfix-merge
