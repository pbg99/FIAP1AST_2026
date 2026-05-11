# NPS Preditivo para E-commerce

Modelo preditivo de NPS (Net Promoter Score) para e-commerce que antecipa clientes detratores **antes** da pesquisa de satisfação, permitindo ações preventivas pela operação.

Desenvolvido como Tech Challenge da Pós-Tech AI Scientist — FIAP (Fase 1, 2026).

---

## Pipeline

![Pipeline NPS Preditivo](docs/pipeline.svg)

---

## Problema

A base analisada apresenta **74% dos clientes como detratores** (NPS ≤ 6) e um NPS oficial em torno de **-66** — situação crítica. A insatisfação não é explicada por perfil demográfico, mas pela **jornada operacional**: atraso de entrega, reclamações recorrentes e múltiplos contatos com o SAC.

**Pergunta central:** como antecipar a insatisfação do cliente a partir de dados operacionais e agir antes que ele se torne um detrator?

---

## Resultados

| Modelo | Tipo | Métrica | Resultado |
|--------|------|---------|-----------|
| XGBoost | Classificação (detrator?) | AUC-ROC | ~0.92 |
| XGBoost | Classificação (detrator?) | F1-Score | ~0.95 |
| Random Forest | Regressão (nota NPS 0–10) | R² | ~0.65 |
| Random Forest | Regressão (nota NPS 0–10) | MAE | ~1.2 pontos |

### Drivers de insatisfação identificados

| Driver | Ponto de ruptura | % Detratores |
|--------|-----------------|--------------|
| Reclamações recorrentes | ≥ 5 reclamações | 94,5% |
| Múltiplos contatos com SAC | ≥ 3 contatos | 90,4% |
| Tempo de resolução alto | > 11 dias | 84,1% |
| Atraso na entrega | ≥ 3 dias | Escala crítica |

---

## Metodologia

O projeto segue a metodologia **CRISP-DM** de ponta a ponta:

1. **Business Understanding** — Definição do problema e métricas de sucesso
2. **Data Understanding** — EDA com análise estatística (Spearman, Kruskal-Wallis, Mann-Whitney)
3. **Data Preparation** — Feature engineering com variáveis derivadas dos pontos de ruptura da EDA
4. **Modeling** — Regressão (Ridge, Random Forest, XGBoost) e Classificação binária de detratores
5. **Evaluation** — Validação cruzada estratificada (5-fold) + holdout 75/25, SHAP values
6. **Deployment** — Score de risco com faixas operacionais (Crítico/Alto/Médio/Baixo), simulação de ROI e teste A/B

---

## Estrutura do Repositório

```
├── data/                          # Bases de dados centralizadas (original + processadas)
│
├── Tratamento de Dados/
│   └── tech_challenge/
│       ├── notebooks/             # Notebook de limpeza e EDA
│       ├── scripts/               # Pipeline completo em Python
│       ├── figures/               # Gráficos gerados (v1 e v2)
│       └── reports/               # Resumos executivos e comparativos
│
├── EAD Atendimento e Cliente/     # EDA: reclamações, SAC, tempo de resolução
├── EAD Logistica e Pedido/        # EDA: atraso, frete, tentativas de entrega
├── Modelo Preditivo/              # Notebook completo: EDA + modelagem (RF, XGBoost, SHAP)
├── Modelo em Produção/            # Pipeline final para deploy com faixas de risco
├── Storytelling/                  # Narrativa executiva e comunicação de insights
│
├── Apresentacao_NPS_FIAP.pptx     # Slide deck da apresentação
├── Apresentação Executiva.pdf     # PDF executivo
└── Video de Apresentação.mp4      # Vídeo de apresentação
```

---

## Stack Técnica

**Linguagem:** Python 3

**Bibliotecas:**
- `pandas`, `numpy` — manipulação de dados
- `matplotlib`, `seaborn` — visualizações
- `scikit-learn` — modelagem (Random Forest, Ridge, Logistic Regression)
- `xgboost` — classificação e regressão
- `shap` — interpretabilidade do modelo
- `scipy` — testes estatísticos (Kruskal-Wallis, Mann-Whitney, Spearman)

---

## Como Executar

```bash
# Instalar dependências
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap scipy joblib

# Pipeline completo de tratamento
cd "Tratamento de Dados/tech_challenge/scripts"
python tech_challenge_nps.py

# Ou abrir os notebooks individualmente
jupyter notebook "Modelo Preditivo/EDA_Modelo_Preditivo_NPS_Final.ipynb"
jupyter notebook "Modelo em Produção/Modelo_Final_em_Producao.ipynb"
```

---

## Recomendações para o Negócio

1. **Logística primeiro** — Revisar SLA com transportadoras e instituir alertas a partir do 1º dia de atraso
2. **Score de risco em produção** — Integrar o modelo de classificação ao fluxo de pedidos para priorização automática
3. **SAC priorizado por risco** — Fila de atendimento ordenada por probabilidade de detração
4. **Programa de recompra** — Incentivar segunda compra nos primeiros 14 dias como sinal positivo de retenção
5. **Monitoramento contínuo** — Acompanhar drift das variáveis e retreinar o modelo periodicamente

---

## Demo Rápido

Abra o notebook [`demo.ipynb`](demo.ipynb) para ver o projeto completo em ação: carregamento dos dados, treinamento do modelo, resultados visuais e score de risco — tudo em um único notebook.

---

## Autor

**Gustavo Pietro Seferin**
Pós-graduação em AI Scientist — FIAP (2026)
