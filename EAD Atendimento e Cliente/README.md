# FASE 1 — EDA: Atendimento e Cliente

## Visão Geral

Análise exploratória de dados (EDA) da base NPS do Tech Challenge FIAP — Fase 1.  
O objetivo é identificar os principais drivers de insatisfação de clientes a partir de variáveis de atendimento, logística e comportamento de compra.

---

## Base de Dados

| Atributo | Valor |
|---|---|
| Arquivo | `desafio_nps_fase_1.csv` |
| Linhas | 2.500 |
| Colunas | 19 |

### Colunas disponíveis

| Coluna | Descrição |
|---|---|
| `customer_id` | Identificador único do cliente |
| `customer_age` | Idade do cliente |
| `customer_region` | Região geográfica |
| `customer_tenure_months` | Tempo de relacionamento em meses |
| `order_id` | Identificador do pedido |
| `order_value` | Valor do pedido (R$) |
| `items_quantity` | Quantidade de itens |
| `discount_value` | Valor de desconto aplicado |
| `payment_installments` | Número de parcelas |
| `delivery_time_days` | Tempo de entrega em dias |
| `delivery_delay_days` | Dias de atraso na entrega |
| `freight_value` | Valor do frete |
| `delivery_attempts` | Tentativas de entrega |
| `customer_service_contacts` | Contatos com atendimento |
| `resolution_time_days` | Dias para resolução de chamado |
| `nps_score` | Nota NPS (0–10) |
| `repeat_purchase_30d` | Recompra em 30 dias (0/1) |
| `complaints_count` | Quantidade de reclamações |
| `csat_internal_score` | Nota CSAT interna |

---

## Metodologia

### Classificação NPS

| Categoria | Critério |
|---|---|
| Detrator | Nota de 0 a 6 |
| Neutro | Nota maior que 6 e menor que 9 |
| Promotor | Nota igual ou maior que 9 |

### Técnicas utilizadas

- Análise descritiva por faixa de variáveis
- Correlação de Spearman entre variáveis numéricas e `nps_score`
- Teste estatístico para diferença regional
- Construção de score de risco composto (0–3)

---

## Principais Achados

### Reclamações × NPS

| Faixa de reclamações | % Detratores |
|---|---|
| 3 a 5 reclamações | 78,79% |
| 6 ou mais reclamações | 94,50% |

### Contatos com atendimento × NPS

| Contatos | NPS médio | % Detratores |
|---|---|---|
| 3 ou mais | 2,94 | 90,41% |

### Tempo de resolução × NPS

| Faixa de resolução | % Detratores |
|---|---|
| Acima de 11 dias | 84,11% |

### Score de risco composto

| Score | % Detratores |
|---|---|
| 2 | 84,54% |
| 3 | 97,34% |

### Análise regional

Variação baixa entre regiões. Teste estatístico não indicou diferença relevante. **Priorizar problemas de jornada, não segmentação regional.**

---

## Decisões Operacionais Sugeridas

1. **Fila de recuperação** — acionar antes de nova reclamação para clientes com 3+ reclamações
2. **Alerta na 3ª interação** — priorizar atendimento humano ou revisão de caso
3. **Monitoramento de resolução** — escalar casos acima de 6 dias; crítico acima de 11 dias
4. **Desconsiderar segmentação regional** — nenhuma evidência estatística de diferença
5. **Score de risco como régua operacional** — score 2 = priorização; score 3 = atuação urgente

---

## Como Executar

### Pré-requisitos

```bash
pip install pandas numpy matplotlib scipy
```

### Ambiente

O notebook foi desenvolvido no **Google Colab** com acesso ao Google Drive.  
Para rodar localmente, substituir o bloco de montagem do Drive pelo carregamento local do CSV:

```python
df = pd.read_csv("caminho/para/desafio_nps_fase_1.csv")
```

### Ordem de execução

Executar as células em ordem sequencial. Todas as dependências são carregadas na primeira célula de importação.

---

## Dependências

| Biblioteca | Uso |
|---|---|
| `pandas` | Manipulação e análise de dados |
| `numpy` | Operações numéricas |
| `matplotlib` | Visualizações |
| `scipy` | Correlação de Spearman e testes estatísticos |

---

## Estrutura do Projeto

```
FASE_1_-_EDA_atendimento_e_cliente.ipynb   ← notebook principal
desafio_nps_fase_1.csv                     ← base de dados (não inclusa no repositório)
README.md                                  ← este arquivo
```

---

## Autor

Gustavo Pietro — Pós-graduação FIAP · Tech Challenge NPS
