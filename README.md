# pBGSK-FS: Population-based Binary Gaining–Sharing Knowledge-based Optimization for Feature Selection

This project implements the **pBGSK-FS** algorithm, a population-based optimization technique tailored for the feature selection task in machine learning.


Feature selection is a well-known NP-hard problem; given $n$ features, there are $O(n!)$ possible ways to combine them. To solve this, the algorithm acts as a **wrapper method**. It evaluates the fitness of different feature subsets by training a K-Nearest Neighbors (KNN) classifier and calculating a score based on both classification accuracy and the ratio of selected features. 
## Where we are.
Although still more benchmarks are necessary the base code is working ,and some unit testing is already implemented. Have fun experimenting.

## How to Use

The project provides an easy-to-use functional interface. You need to load your data, split it into training and testing sets, and pass it to the `feature_selection` function.

### Quick Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.data_importer import data_loader
from src.utils.pBGSK import feature_selection

# 1. Load a benchmark dataset (e.g., Breast Cancer from UCI)
dataset = data_loader("breast_cancer")
X = dataset.data.features
y = dataset.data.targets

# Basic preprocessing for KNN (handling categorical/NaNs)
X = pd.get_dummies(X).fillna(X.mean())
y = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
data_tuple = (X_train, X_test, y_train, y_test)

# 3. Run the pBGSK Feature Selection algorithm
population, best_features_mask, best_score = feature_selection(
    data_tuple=data_tuple,
    num_population=20,               # Initial population size
    nfe_total=100,                   # Max number of function evaluations
    lower_k=1,                       # Min features to select initially
    upper_k=X_train.shape[1] // 2,   # Max features to select initially
    columns_names=X_train.columns.tolist(),
    data_set_name="breast_cancer",
    knn_val=3                        # K value for the KNN evaluator
)

print(f"Final Best Score: {best_score:.4f}")
print(f"Selected Features: {X_train.columns[best_features_mask].tolist()}")
```
### Main Function Parameters

*   `data_tuple`: A tuple containing `(X_train, X_test, y_train, y_test)`.
*   `num_population`: The initial number of individuals (solutions) in the population. Must be > 12 to allow for the reduction phases.
*   `nfe_total`: Total budget for function evaluations. The algorithm stops when this is reached.
*   `lower_k` / `upper_k`: Bounds for the random number of features assigned to individuals during initialization.
*   `columns_names`: A list of strings representing the feature names.
*   `k`: Knowledge factor for dimension distribution (default `0.95`).
*   `p`: Population partitioning ratio for the Senior phase (default `0.1` or 10%).
*   `knn_val`: The `n_neighbors` parameter used by the internal KNeighborsClassifier to evaluate fitness.
---
## What is different.

A major differentiator in the computing sector is the ability to emulate natural or social solutions, adding abstraction and ingenuity. While many feature selection algorithms draw inspiration from animal behaviors or physical processes, this algorithm attempts to abstract the **accumulation of human knowledge** over a lifetime into a computational model.

## Inspiration : Human Knowledge and Relationships

Throughout our lives, we are exposed to various influences. But what are the main influences at each stage of life?

*   **Childhood (Junior Phase):** Our influences are limited to a small, contained social network: family, neighbors, and early teachers.
*   **Adulthood (Senior Phase):** As we enter college and the job market, our networks expand. Positive and negative influences grow, and our environment dictates different acceptable behaviors.

The core idea is that immediate, local environments are more influential during early stages (Junior), while broader, previously distant influences become important later in life (Senior).

## How it is Applied

The algorithm relies on the following key elements:
1.  **Population:** The collection of all individuals.
2.  **Individuals:** Each individual possesses a binary vector indicating which dimensions (features) are currently selected (active).
3.  **Dimensions:** The features of the dataset.
4.  **Seniority:** Dimensions are dynamically classified as either "Junior" or "Senior" as the optimization progresses.

### Algorithm Phases

1.  **Population Creation:** A population of individuals is created with an initial number of selected dimensions between `[lower_k, upper_k]`. A seniority vector is established for the dimensions.
2.  **Population Classification (Fitness Evaluation):** Individuals are evaluated and sorted based on their fitness score. Lower scores are better. The score is calculated as: `(1 - accuracy) + (1 - features_used / total_features)`.
3.  **Gain-Share Junior (Beginner):** Information sharing occurs between neighboring individuals.
4.  **Gain-Share Senior (Intermediate):** Information sharing occurs using the best, middle, and worst individuals of the population.
5.  **Population Reduction:** The population size is linearly reduced as the evaluations progress to focus on the most promising individuals.

#### Dimension Classification
As the search progresses (measured by the Number of Function Evaluations, or NFE), features transition from the Junior stage (exploration-heavy) to the Senior stage (exploitation-heavy).

$$d_{junior} = d \times \left( 1 - \frac{NFE}{MaxNFE} \right)^K$$
$$d_{senior} = d - d_{junior}$$

#### Junior Phase (Beginners)
During the Junior phase, individuals learn from their immediate neighbors (the one immediately better and the one immediately worse in the sorted population) and a random individual.

Pseudo-code:
```text
for t = 1 to Np do
    for k = 1 to d do
        if rand ≤ kR (knowledge ratio) then
            if f(xt) > f(xR) then
                xtk_new = xt + kf * [(xt-1 - xt+1) + (xR - xt)]
            else
                xtk_new = xt + kf * [(xt-1 - xt+1) + (xt - xR)]
            end if
        else
            xtk_new = xtk_old
        end if
    end for
end for
```

#### Senior Phase (Intermediate)
In the Senior phase, influences are broader. The population is partitioned into three groups based on a percentage `P` (e.g., 10%): the best `P`, the worst `P`, and the middle individuals.

Pseudo-code:
```text
for t = 1 to Np do
    for k = 1 to d do
        if rand ≤ kR (knowledge ratio) then
            if f(xt) > f(xmiddle) then
                xtk_new = xt + kf * [(xpbest - xpworst) + (xmiddle - xt)]
            else
                xtk_new = xt + kf * [(xpbest - xpworst) + (xt - xmiddle)]
            end if
        else
            xtk_new = xtk_old
        end if
    end for
end for
```

---
