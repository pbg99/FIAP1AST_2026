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
