# Tech Challenge Fase 1 — Case NPS Preditivo

> Pós-Tech AI Scientist · Trabalho em grupo · Metodologia CRISP-DM

## 🎯 Objetivo

Antecipar o NPS (Net Promoter Score) de clientes de e-commerce a partir de dados operacionais (pedidos, logística, atendimento) **antes** da aplicação da pesquisa de satisfação, permitindo ações preventivas.

## 📊 Base de Dados

Arquivo: `data/desafio_nps_fase_1.csv`

- **2.500 clientes** (cada cliente com 1 pedido)
- **19 colunas** divididas em:
  - Identificação: `customer_id`, `order_id`
  - Perfil do cliente: `customer_age`, `customer_region`, `customer_tenure_months`
  - Pedido: `order_value`, `items_quantity`, `discount_value`, `payment_installments`
  - Logística: `delivery_time_days`, `delivery_delay_days`, `freight_value`, `delivery_attempts`
  - Atendimento: `customer_service_contacts`, `resolution_time_days`, `complaints_count`
  - Indicadores: `csat_internal_score`, `repeat_purchase_30d`
  - Target: `nps_score` (0 a 10, **contínuo**)

> **Nota:** o dicionário do PDF descreve `nps_score` como variando de 0 a 10. Na prática a variável é contínua (com casas decimais), o que sugere se tratar de um *score* internamente calculado ou ponderado, não uma resposta direta de pesquisa.

## 🔬 Metodologia (CRISP-DM)

| Etapa | Onde está |
|---|---|
| 1. Business Understanding | `scripts/tech_challenge_nps.py` (função `business_understanding`) |
| 2. Data Understanding | `notebooks/01_limpeza_eda.ipynb` |
| 3. Data Preparation | `notebooks/01_limpeza_eda.ipynb` + função `data_preparation` |
| 4. Modeling | `scripts/tech_challenge_nps.py` (regressão + classificação) |
| 5. Evaluation | função `evaluation` |
| 6. Deployment | função `deployment_recomendacoes` |

## 📁 Estrutura

```
tech_challenge/
├── data/
│   ├── desafio_nps_fase_1.csv          # Base original
│   └── desafio_nps_processado.csv      # Base com features derivadas
├── notebooks/
│   └── 01_limpeza_eda.ipynb            # Limpeza + EDA com narrativa
├── scripts/
│   └── tech_challenge_nps.py           # Pipeline completo
├── figures/                            # Gráficos gerados
├── models/                             # Modelos serializados (.pkl)
└── reports/
    └── resumo_executivo.docx           # Resumo gerencial
```

## ▶️ Como reproduzir

### Pré-requisitos
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter joblib
```

### Pipeline completo
```bash
cd scripts/
python tech_challenge_nps.py
```

### Notebook EDA
```bash
cd notebooks/
jupyter notebook 01_limpeza_eda.ipynb
```

## 🔑 Principais Achados

- **NPS oficial em torno de -66**: situação crítica (qualquer valor < 0 indica mais detratores que promotores)
- **74% da base é detratora** (NPS < 7), 18% neutros, 8% promotores
- **Atraso na entrega é o pior driver** (correlação ≈ -0.60 com NPS)
- **Recompra em 30 dias** e **CSAT interno** são os melhores preditores positivos
- **Idade, tenure e valor do pedido têm impacto baixo** — o problema é operacional, não demográfico
- **Regiões têm NPS parecidos** — problema sistêmico, não localizado

## 🤖 Modelos

| Modelo | Tipo | Métrica principal | Resultado |
|---|---|---|---|
| Regressão Linear / RF | Regressão (predizer NPS 0-10) | R² ≈ 0.65, MAE ≈ 1.2 | Útil para *ranking* de risco |
| Logística / RF | Classificação binária (detrator?) | F1 ≈ 0.95, AUC ≈ 0.92 | Útil para *acionamento* automático |

## 💡 Recomendações para o Negócio

1. **Logística primeiro** — revisar SLA com transportadoras e instituir alertas pré-prazo
2. **Score de risco em produção** — integrar modelo de classificação ao fluxo de pedidos
3. **SAC priorizado por risco** — fila de atendimento ordenada por probabilidade de detração
4. **Programa de recompra** — incentivar segunda compra nos primeiros 14 dias

## ⚠️ Limitações

- Base com 2.500 clientes — pequena para modelos de alta complexidade
- `csat_internal_score` pode ser coletado *após* o evento e vazar o rótulo em produção
- Algumas features (reclamações, contatos com SAC) são *reativas* — para prevenção real, peso maior em variáveis logísticas
- Sem dados temporais explícitos — não modelamos tendências ou sazonalidade

## 👥 Autores

Grupo Pós-Tech AI Scientist — Fase 1 (2026)
