# BGSK-FS: Binary Gaining–Sharing Knowledge-based Optimization

Esta é uma implementação do algoritmo **BGSK** (Binary Gaining–Sharing Knowledge), uma meta-heurística moderna aplicada ao problema de **Seleção de Características (Feature Selection)**. O projeto busca otimizar a acurácia de modelos de Machine Learning através da redução inteligente da dimensionalidade dos dados.

---

## 🧠 Inspiração e Conceito

O algoritmo é inspirado no comportamento humano de compartilhamento de conhecimento ao longo da vida. Ele divide o processo de otimização em duas fases principais:

1. **Fase Júnior (Early Life):** Simula o aprendizado inicial com pessoas próximas (pais, professores e parentes). No algoritmo, isso se traduz em uma busca onde apenas os indivíduos com desempenho mais próximo influenciam a adesão ou abandono de características.
2. **Fase Sênior (Late Life):** Simula a participação em grandes comunidades e redes sociais. Aqui, a influência de indivíduos mais distantes na população é levada em conta, permitindo uma exploração global mais eficiente do espaço de busca.

---

## 🛠️ Funcionamento do Algoritmo

### 1. Representação da População
Cada indivíduo da população é representado por um **vetor binário** de features, gerado aleatoriamente dentro de limites mínimos e máximos de elementos.
* **1**: A feature está presente no modelo.
* **0**: A feature foi descartada.

### 2. Avaliação e Ranqueamento (Fitness)
O ranqueamento dos indivíduos é baseado em uma função multiobjetivo que busca o equilíbrio entre performance e simplicidade:
* **Complemento da Acurácia:** $1 - \text{acurácia}$
* **Proporção de Features:** $1 - (\frac{\text{features selecionadas}}{\text{total de features}})$

### 3. Redução Dinâmica da População
O algoritmo utiliza um critério de gerações máximas em conjunto com um número mínimo de indivíduos. 
* A cada geração, após as fases de compartilhamento de conhecimento, os indivíduos com pior desempenho são descartados.
* Esse descarte estratégico garante **maior flexibilidade** e foca o processamento nas soluções que apresentam os melhores resultados de convergência.

---

## 🚀 Como Utilizar

```python
# Exemplo básico de uso (ajuste conforme sua implementação)
from bgsk_fs import BGSKOptimizer

# Inicialize o otimizador
optimizer = BGSKOptimizer(min_features=5, max_population=50)

# Execute a seleção de características
best_features = optimizer.fit(X_train, y_train)

print(f"Features selecionadas: {best_features}")
