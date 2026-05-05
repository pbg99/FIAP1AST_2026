# P3 — EDA Logística e Pedido

> **Atraso, Frete, Tentativas e Valor do Pedido na Satisfação do Cliente**

---

## Visão Geral

Este projeto aprofunda a análise exploratória de dados (EDA) das variáveis logísticas e de pedido com foco na satisfação do cliente medida por NPS. A base utilizada é a **versão V2** do dataset processado (`desafio_nps_processado_v2.csv`), na qual registros com `nps_score = 0` foram removidos para validação de robustez dos padrões identificados na V1.

O notebook parte de uma pergunta central:

> **Como atraso, frete, tentativas de entrega e valor do pedido se relacionam com a satisfação do cliente e com o faturamento observado?**

---

## Resumo Executivo

Mesmo após a remoção dos casos extremos (NPS = 0), a base V2 manteve um perfil crítico de satisfação: **forte concentração de Detratores** e **NPS oficial negativo**. Isso confirma que o problema estrutural identificado na V1 não é artefato dos outliers — ele é real e sistêmico.

A análise identificou que a deterioração da experiência do cliente está diretamente associada à mecânica logística: atraso de entrega, tentativas fracassadas, custo proporcional do frete e faturamento em risco.

### Achados principais

| Dimensão | Achado |
|---|---|
| **Atraso de entrega** | Principal variável associada à insatisfação; Detratores concentram maior atraso médio e mediano |
| **Ponto de ruptura** | A análise por faixas revela o intervalo exato em que a proporção de Detratores escala de forma relevante |
| **Frete proporcional** | O `freight_ratio` (frete / valor do pedido) revela impacto que o valor absoluto do frete não captura |
| **Múltiplas tentativas** | Pedidos com mais de uma tentativa de entrega apresentam maior fricção logística e sinal de insatisfação |
| **Faturamento com atraso** | Pedidos atrasados concentram faturamento relevante — melhoria logística é também preservação de receita |
| **Análise regional** | Diferenças e semelhanças entre regiões indicam se o problema é localizado ou sistêmico |
| **Correlação e testes** | Kruskal-Wallis + Mann-Whitney (Bonferroni) confirmam que a diferença de atraso entre grupos de NPS é **estatisticamente significativa** |

---

## Dataset

**Arquivo:** `desafio_nps_processado_v2.csv`  
**Registros:** ~2.342 (após remoção dos NPS = 0)  
**Colunas:** 26

### Principais variáveis

| Coluna | Descrição |
|---|---|
| `customer_id` | Identificador único do cliente |
| `customer_region` | Região do cliente |
| `customer_age` | Idade do cliente |
| `customer_tenure_months` | Tempo de relacionamento (meses) |
| `order_id` | Identificador do pedido |
| `order_value` | Valor total do pedido |
| `items_quantity` | Quantidade de itens |
| `discount_value` / `desconto_pct` | Valor e percentual de desconto |
| `payment_installments` | Parcelas de pagamento |
| `delivery_time_days` | Tempo total de entrega (dias) |
| `delivery_delay_days` | Dias de atraso em relação ao prazo |
| `freight_value` | Valor do frete |
| `delivery_attempts` | Número de tentativas de entrega |
| `customer_service_contacts` | Contatos com o SAC |
| `resolution_time_days` | Tempo de resolução (dias) |
| `nps_score` | Score NPS do cliente (0–10) |
| `classificacao_nps` | Detrator / Neutro / Promotor |
| `eh_detrator` | Flag binária de Detrator |
| `repeat_purchase_30d` | Recompra em 30 dias (binário) |
| `complaints_count` | Número de reclamações |
| `csat_internal_score` | Score CSAT interno |
| `teve_atraso` | Flag: pedido teve atraso |
| `teve_reclamacao` | Flag: cliente registrou reclamação |
| `contatou_sac` | Flag: cliente contatou SAC |
| `severidade_problemas` | Severidade dos problemas relatados |

### Variáveis derivadas (criadas no notebook)

| Coluna | Descrição |
|---|---|
| `freight_ratio` | `freight_value / order_value` — peso proporcional do frete no pedido |
| `has_multiple_attempts` | Booleano: mais de uma tentativa de entrega |
| `delay_range` | Faixa de atraso: Sem atraso / Até 1 dia / 2–3 dias / 4–5 dias / 6–10 dias / Acima de 10 dias |

---

## Estrutura do Notebook

```
P3_EDA_Logistica_e_Pedido.ipynb
│
├── 0.  Preparação do ambiente
├── 1.  Carregamento e validação inicial da base
├── 2.  Distribuição da classificação NPS
├── 3.  Criação e validação das variáveis derivadas
├── 4.  Estatística descritiva de logística e pedido por categoria NPS
├── 5.  Atraso de entrega por categoria de NPS
├── 6.  Probabilidade condicional: NPS dado atraso
├── 7.  Ponto de ruptura no atraso (por faixas)
├── 8.  Frete absoluto e peso proporcional do frete
├── 9.  Tentativas de entrega
├── 10. Valor do pedido e faturamento observado  [implícito nas análises regionais]
├── 11. Análise regional: logística, NPS e faturamento
├── 12. [continuação regional — tabela consolidada]
├── 13. Matriz de correlação (NPS × logística × pedido)
├── 14. Testes estatísticos: Kruskal-Wallis + Mann-Whitney (Bonferroni)
├── 15. Relação do P3 com o modelo preditivo
└── 16. Resumo executivo dos achados
```

---

## Hipóteses Testadas

| ID | Hipótese | Status |
|---|---|---|
| **H1** | Pedidos com atraso apresentam maior proporção de Detratores | ✅ Confirmada |
| **H2** | Existe um ponto de ruptura no atraso onde Detratores aumentam relevantemente | ✅ Identificada por faixas |
| **H3** | O peso proporcional do frete pode estar associado à menor satisfação | 🔍 Analisada via `freight_ratio` |
| **H4** | Múltiplas tentativas de entrega podem estar associadas à menor satisfação | 🔍 Avaliada via `has_multiple_attempts` |
| **H5** | Pedidos de maior valor com atraso representam oportunidades de melhoria logística e preservação de faturamento | ✅ Relevante por faturamento observado |

---

## Metodologia

- **Estatística descritiva:** média, mediana, desvio padrão, mínimo e máximo por categoria de NPS
- **Probabilidade condicional:** tabelas cruzadas normalizadas para observar distribuição de NPS dado atraso / tentativas
- **Análise de faixas:** `delay_range` como variável operacional para identificar ponto de ruptura
- **Correlação de Pearson:** matriz de correlação entre NPS e variáveis numéricas
- **Testes não paramétricos:**
  - Kruskal-Wallis — diferença entre três grupos
  - Mann-Whitney U com correção de Bonferroni — comparações par a par
- **Análise regional:** cruzamento de logística, NPS e faturamento por `customer_region`

---

## Recomendações Operacionais

1. **Monitorar atrasos desde o 1º dia** — o sinal de insatisfação começa cedo.
2. **Definir um SLA de ruptura** — usar as faixas do `delay_range` para estabelecer limite de tolerância operacional.
3. **Tratar múltiplas tentativas como alerta** — sinaliza falha logística ou de comunicação com o cliente.
4. **Usar `freight_ratio`** — analisar frete proporcional, não apenas valor absoluto.
5. **Priorizar pedidos de alto valor com atraso** — preservação de faturamento e risco de churn.
6. **Avaliar estratégia regional** — verificar se ações corretivas devem ser localizadas ou aplicadas a toda a operação.
7. **Usar as variáveis do P3 como insumo para modelagem preditiva de NPS** — atraso, frete, tentativas e recompra são candidatas relevantes.

---

## Relação com o Projeto Maior

Este notebook integra uma série de análises (P1, P2, P3...) voltadas à compreensão do NPS e à construção de um modelo preditivo.

- A **V1** permanece como base de referência principal (inclui Detratores com NPS = 0).
- A **V2** (utilizada neste notebook) valida a robustez dos padrões após remoção dos extremos.
- O **P3** conecta os achados exploratórios às variáveis candidatas para a próxima etapa de modelagem preditiva de NPS.

---

## Dependências

```
pandas
numpy
matplotlib
seaborn
scipy
```

---

## Como Executar

```bash
# Clone o repositório ou abra no Jupyter
jupyter notebook P3_EDA_Logistica_e_Pedido.ipynb
```

Certifique-se de que o arquivo `desafio_nps_processado_v2.csv` está no mesmo diretório do notebook, ou ajuste o `path_v2` na célula de carregamento.

---

> *"O EDA geral encontrou o problema. O P3 explicou a mecânica logística desse problema: atraso, tentativas, frete e valor do pedido ajudam a entender onde a experiência se deteriora e onde a operação pode atuar para melhorar satisfação e preservar faturamento."*
