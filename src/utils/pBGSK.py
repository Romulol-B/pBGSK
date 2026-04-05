"""
Gaining Sharing Knowledge (GSK) algorithm for feature selection.

This module implements the pBGSK algorithm, a population-based optimization
technique inspired by the human process of gaining and sharing knowledge
throughout different life stages (Junior and Senior), tailored for the
feature selection task in machine learning.
"""

import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def k_factor(kf: float = 0.95) -> int:
    """
    Stochastic knowledge factor multiplier.

    Returns 1 if a random value is greater than or equal to the knowledge factor,
    otherwise returns 0. This is used to introduce randomness in the GSK update
    rules.

    Parameters
    ----------
    kf : float, default=0.95
        The knowledge factor threshold.

    Returns
    -------
    int
        1 or 0 based on the stochastic comparison.
    """
    return 1 if random.random() >= kf else 0


class Individual:
    """
    Represents a single solution (individual) in the population.

    Each individual maintains a binary vector representing the selection of features
    and its associated fitness metrics.

    Parameters
    ----------
    data_set_name : str
        Name of the dataset being processed.
    individual_id : int
        Unique identifier for the individual.
    features : np.ndarray
        Binary or boolean array indicating which features are selected.
    columns_names : list of str
        Names of all available features in the dataset.

    Attributes
    ----------
    data_set_name : str
        Name of the dataset.
    individual_id : int
        ID of the individual.
    features : np.ndarray (bool)
        Boolean mask of selected features.
    column_names : list of str
        Names of the features.
    acc : float
        Accuracy achieved by this individual's feature subset.
    score : float
        Fitness score of the individual (lower is better).
        Calculated as (1 - accuracy) + (1 - feature_ratio).
    df : pd.DataFrame or None
        Placeholder for data related to the individual.
    number_of_features : int
        Count of selected features.
    """
    def __init__(
        self,
        individual_id: int,
        features: np.ndarray,
    ):
        self.individual_id = individual_id
        
        self.features = np.array(features, dtype=bool)
        
        self.acc = 0.0
        self.score = 2.0  # Pior score possível (1 - acc + 1 - ratio)
        self.df = None
        self.number_of_features = 0


class FeatureSelectorEvaluator:
    """
    Evaluates the fitness of a feature subset.

    This class handles the training and testing of a classifier to determine
     the quality of a specific feature combination.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training feature set.
    X_test : pd.DataFrame or np.ndarray
        Testing feature set.
    y_train : pd.Series or np.ndarray
        Training labels.
    y_test : pd.Series or np.ndarray
        Testing labels.
    knn_val : int, default=5
        Number of neighbors for the KNeighborsClassifier.
    other_classifier : estimator object, default=None
        An optional scikit-learn compatible classifier. If None,
        KNeighborsClassifier is used.

    Attributes
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features.
    X_test : pd.DataFrame or np.ndarray
        Testing features.
    y_train : pd.Series or np.ndarray
        Training labels.
    y_test : pd.Series or np.ndarray
        Testing labels.
    classifier : estimator object
        The classifier used for evaluation.
    """
    def __init__(
        self, X_train, X_test, y_train, y_test, knn_val: int = 5, other_classifier=None
    ):
        self.X_train = np.asarray(X_train)
        self.X_test = np.asarray(X_test)
        self.y_train = np.asarray(y_train)
        self.y_test = np.asarray(y_test)
        if other_classifier is None:
            self.classifier = KNeighborsClassifier(n_neighbors=knn_val)
        else:
            self.classifier = other_classifier

    def calculate_fitness(
        self,
        features: np.ndarray,
    ) -> tuple[np.float64, np.float64]:
        """
        Calculate fitness score and accuracy for a given feature mask.

        The fitness score is defined as:
        score = (1 - accuracy) + (1 - (selected_features / total_features))

        Parameters
        ----------
        features : np.ndarray
            Boolean mask of features to evaluate.

        Returns
        -------
        score : np.float64
            The calculated fitness score (lower is better).
        acc : np.float64
            The classification accuracy.
        """
        number_of_features = np.sum(features)
        if number_of_features == 0:
            return 2.0, 0.0

        # Seleciona apenas as features ativas
        X_train_selected = self.X_train[:, features]
        X_test_selected = self.X_test[:, features]

        # Treina e pontua o modelo
        self.classifier.fit(X_train_selected, self.y_train)
        y_pred = self.classifier.predict(X_test_selected)
        
        acc = accuracy_score(self.y_test, y_pred)
        
        feature_ratio = number_of_features / len(features)
        
        score = np.float64((1 - acc) + (1 - feature_ratio))
        return score, np.float64(acc)


class Population:
    """
    Container for the population of individuals and optimization parameters.

    Parameters
    ----------
    individuals : list of Individual
        Initial list of individuals in the population.
    data_tuple : tuple
        A tuple containing (X_train, X_test, y_train, y_test).
    knowledge : float, default=0.95
        Knowledge factor threshold for dimension distribution.
    partition : float, default=0.1
        The proportion of the population considered as "best", "middle", or "worst"
        during intermediate GSK phases.
    knn_val : int, default=5
        Number of neighbors for the default KNN evaluator.

    Attributes
    ----------
    X_train, X_test, y_train, y_test : data structures
        Training and testing data.
    individuals : list of Individual
        The current population members.
    len : int
        Current population size.
    evaluator : FeatureSelectorEvaluator
        Object used to calculate fitness for individuals.
    df : pd.DataFrame or None
        Dataframe containing the current population state and metrics.
    geng_df : pd.DataFrame
        Historical record of mean population metrics across generations.
    d_junior : int
        Number of dimensions assigned to the Junior stage.
    d_senior : int
        Number of dimensions assigned to the Senior stage.
    junior_features : np.ndarray (int)
        Binary mask of features currently in the Junior stage.
    senior_features : np.ndarray (int)
        Binary mask of features currently in the Senior stage.
    knowledge : float
        Current knowledge factor threshold.
    partition : float
        Population partitioning ratio.
    nfe : int
        Number of Function Evaluations performed so far.
    """
    def __init__(
        self,
        individuals: list,
        data_tuple: tuple,
        data_set_name: str,
        columns_names: list[str],
        knowledge: float = 0.95,
        partition: float = 0.1,
        knn_val: int = 5,
    ):
        self.X_train,  self.X_test, self.y_train, self.y_test = data_tuple
        self.data_set_name = data_set_name
        self.columns_names = columns_names
        self.individuals = individuals
        self.len = len(individuals)
        
        # Evaluator unificado
        self.evaluator = FeatureSelectorEvaluator(
            self.X_train, self.X_test, self.y_train, self.y_test, knn_val=knn_val
        )
        
        self.df = None
        self.geng_df = pd.DataFrame()
        
        self.d_junior = 0
        self.d_senior = 0
        self.junior_features = None  # vetor binario de features
        self.senior_features = None  # vetor binario de features
        
        self.knowledge = knowledge
        self.partition = partition
        self.nfe = 0


def calculate_population_fitness(pop: Population, individual: Individual):
    """
    Calculate and update the fitness of a specific individual within a population.

    Parameters
    ----------
    pop : Population
        The population context providing the evaluator.
    individual : Individual
        The individual whose fitness needs to be updated.
    """
    score, acc = pop.evaluator.calculate_fitness(individual.features)
    individual.score = score
    individual.acc = acc
    individual.number_of_features = np.sum(individual.features)


def sort_population(pop: Population, t_sort: str = "fitness"):
    """
    Sort the population based on fitness or accuracy and update NFE.

    Ensures all individuals are evaluated before sorting.

    Parameters
    ----------
    pop : Population
        The population to be sorted.
    t_sort : {"fitness", "accuracy"}, default="fitness"
        The metric used for sorting. "fitness" sorts in ascending order (lower is better),
        while "accuracy" sorts in descending order (higher is better).

    Raises
    ----------
    ValueError
        If t_sort is not "fitness" or "accuracy".
    """
    # Garante que todos foram avaliados
    for indiv in pop.individuals:
        if indiv.score == 2.0 and np.sum(indiv.features) > 0: # Evita reavaliar desnecessariamente se já foi feito
             calculate_population_fitness(pop, indiv)
             
    pop.nfe += len(pop.individuals)

    if t_sort == "fitness":
        pop.individuals.sort(key=lambda ind: ind.score)
        if pop.df is not None:
            pop.df.sort_values("score", ascending=True, inplace=True) # Score menor é melhor
    elif t_sort == "accuracy":
        pop.individuals.sort(key=lambda ind: ind.acc, reverse=True)
        if pop.df is not None:
            pop.df.sort_values("acc", ascending=False, inplace=True)
    else:
        raise ValueError("Tipo de sort inválido. Use 'fitness' ou 'accuracy'.")


def dimension_distribution(pop: Population, nfe_total: int) -> int:
    """
    Calculate the distribution of features between Junior and Senior stages.

    As the search progresses (NFE increases), features transition from Junior
    (exploration-heavy) to Senior (exploitation-heavy) stages.

    Parameters
    ----------
    pop : Population
        The population to update.
    nfe_total : int
        The total maximum number of function evaluations allowed.

    Returns
    -------
    int
        The difference between the previous and new Junior dimension counts.
    """
    d = len(pop.individuals[0].features)
    novo_d_junior = min(
        round(d * (1 - pop.nfe / nfe_total) ** pop.knowledge), d - 1
    )
    novo_d_senior = d - novo_d_junior
    diff = pop.d_junior - novo_d_junior
    
    pop.d_junior = novo_d_junior
    pop.d_senior = novo_d_senior

    return diff 


def dimension_classification(pop: Population, nfe_total: int):
    """
    Assign specific features to Junior or Senior categories.

    Handles the initial random assignment and subsequent transitions of features
    between stages based on the dimension distribution.

    Parameters
    ----------
    pop : Population
        The population to update.
    nfe_total : int
        The total maximum number of function evaluations allowed.
    """
    d = len(pop.individuals[0].features)
    
    if pop.d_junior == 0:
        dimension_distribution(pop, nfe_total)
        idxs = np.array(random.sample(range(0, d), pop.d_junior))
        
        pop.junior_features = np.zeros(d, dtype=int)
        pop.junior_features[idxs] = 1
        pop.senior_features = np.ones(d, dtype=int) - pop.junior_features
    else:
        diff = dimension_distribution(pop, nfe_total)
        jf = pop.junior_features.copy()
        sf = pop.senior_features.copy()

        while diff > 0:
            wh = np.squeeze(np.where(jf > 0))
            if wh.size == 0:
                break
            idx_r = random.choice(wh) if wh.ndim == 1 else wh
            jf[idx_r] = 0
            sf[idx_r] = 1
            diff -= 1
            
        pop.junior_features = jf
        pop.senior_features = sf


def beginner_gsk(pop: Population, knowledge_ratio: float = 0.95):
    """
    Apply the Junior (Beginner) stage knowledge sharing rules.

    Focuses on information sharing between neighboring individuals and random
    individuals to explore the feature space.

    Parameters
    ----------
    pop : Population
        The population to evolve.
    knowledge_ratio : float, default=0.95
        Probability that an individual will share/gain knowledge in a specific dimension.
    """
    for t_idx in range(1, pop.len - 1):
        for dimension in np.where(pop.junior_features > 0)[0]:
            if random.random() < knowledge_ratio:
                rand_idx = random.randint(0, pop.len - 1)
                rand_indiv = pop.individuals[rand_idx]
                xt = pop.individuals[t_idx]

                # t_idx garante bounds seguros entre 1 e pop.len - 2
                t_prev = pop.individuals[t_idx - 1]
                t_next = pop.individuals[t_idx + 1]

                kf = k_factor()
                if xt.score > rand_indiv.score: # xt é pior que o aleatorio
                    xtk = int(xt.features[dimension]) + kf * (
                        int(t_prev.features[dimension])
                        - int(t_next.features[dimension])
                        + int(xt.features[dimension])
                        - int(rand_indiv.features[dimension])
                    )
                else:
                    xtk = int(xt.features[dimension]) + kf * (
                        int(t_prev.features[dimension])
                        - int(t_next.features[dimension])
                        + int(xt.features[dimension])
                        - int(rand_indiv.features[dimension])
                    )
                
                new_val = 1 if xtk > 0 else 0
                pop.individuals[t_idx].features[dimension] = bool(new_val)


def intermediate_gsk(pop: Population, knowledge_ratio: float = 0.95):
    """
    Apply the Senior (Intermediate) stage knowledge sharing rules.

    Utilizes the "best", "middle", and "worst" performing individuals to
    exploit promising regions of the feature space.

    Parameters
    ----------
    pop : Population
        The population to evolve.
    knowledge_ratio : float, default=0.95
        Probability that an individual will share/gain knowledge in a specific dimension.
    """
    len_p = max(1, int(pop.len * pop.partition)) # Garante no minimo 1
    
    for t_idx in range(1, pop.len - 1):
        for dimension in np.where(pop.senior_features > 0)[0]:
            if random.random() < knowledge_ratio:
                rand_idx = random.randint(0, pop.len - 1)
                rand_indiv = pop.individuals[rand_idx]
                xt = pop.individuals[t_idx]

                best_x = pop.individuals[random.randint(0, len_p - 1)]
                middle_x = pop.individuals[random.randint(len_p, pop.len - len_p - 1)]
                worst_x = pop.individuals[random.randint(pop.len - len_p, pop.len - 1)]

                kf = k_factor()
                if xt.score > rand_indiv.score:
                    xtk = int(xt.features[dimension]) + kf * (
                        int(best_x.features[dimension])
                        - int(worst_x.features[dimension])
                        + int(middle_x.features[dimension])
                        - int(xt.features[dimension])
                    )
                else:
                    xtk = int(xt.features[dimension]) + kf * (
                        int(best_x.features[dimension])
                        - int(worst_x.features[dimension])
                        + int(xt.features[dimension])
                        - int(middle_x.features[dimension])
                    )
                    
                new_val = 1 if xtk > 0 else 0
                pop.individuals[t_idx].features[dimension] = bool(new_val)


def population_reduction(
    pop: Population, nfe_total: int, low_b: float = 0.9, high_b: float = 0.95
) -> bool:
    """
    Perform Linear Population Size Reduction (LPSR).

    Gradually reduces the population size as the number of function evaluations (NFE)
    approaches the total limit, focusing the search on high-performing individuals.

    Parameters
    ----------
    pop : Population
        The population to reduce.
    nfe_total : int
        Total maximum number of function evaluations allowed.
    low_b : float, default=0.9
        Lower bound multiplier for population reduction.
    high_b : float, default=0.95
        Upper bound multiplier for population reduction.

    Returns
    -------
    bool
        True if the population was NOT reduced or reduction threshold was not met,
        False if individuals were successfully removed.
    """
    np_min = pop.len * low_b
    np_max = pop.len * high_b
    old_len = pop.len
    np_new = int((np_min - np_max) * (pop.nfe / nfe_total) + np_max)

    # Extrai métricas atuais para log
    km = pop.df.loc[:, ["score", "n_features", "acc"]].mean().to_frame().T
    km.rename(columns={"score": "mean_score", "n_features": "mean_n_features", "acc": "mean_acc"}, inplace=True)
    km["nfe"] = pop.nfe
    pop.geng_df = pd.concat([km, pop.geng_df], ignore_index=True)

    if np_new >= 12 and np_new < old_len:
        # População deve estar ordenada; cortamos os piores do final
        amount_to_pop = old_len - np_new
        for _ in range(amount_to_pop):
            pop.individuals.pop()
        pop.len = len(pop.individuals)
        return False
    return True


def get_population_dataframe(pop: Population) -> pd.DataFrame:
    """
    Create a DataFrame representing the current population state.

    Parameters
    ----------
    pop : Population
        The population to extract data from.

    Returns
    -------
    pd.DataFrame
        A DataFrame where rows are individuals and columns are features, scores,
        counts, and accuracy.
    """
    lista_features = [indiv.features.astype(np.int8) for indiv in pop.individuals]
    pop_df = pd.DataFrame(lista_features, columns=pop.columns_names)
    
    pop_df["score"] = [indiv.score for indiv in pop.individuals]
    pop_df["n_features"] = [np.sum(indiv.features) for indiv in pop.individuals]
    pop_df["acc"] = [indiv.acc for indiv in pop.individuals]
    
    pop.df = pop_df
    return pop_df


def plot_population_score(pop: Population):
    """
    Visualize the distribution of scores across different features.

    Uses a boxplot and strip plot to show which features are commonly selected
    among individuals with different fitness scores.

    Parameters
    ----------
    pop : Population
        The population to visualize.
    """
    df_melted = pop.df.melt(
        id_vars=["score", "n_features", "acc"],
        var_name="feature",
        value_name="is_selected"
    )
    # Filtra apenas as features selecionadas
    df_p = df_melted[df_melted["is_selected"] == 1]

    f, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(
        data=df_p,
        x="score",
        y="feature",
        whis=[0, 100],
        width=0.6,
        palette="vlag",
        ax=ax
    )
    sns.stripplot(data=df_p, x="score", y="feature", size=4, color=".3", ax=ax)

    ax.xaxis.grid(True)
    ax.set(ylabel="Features", xlabel="Score")
    sns.despine(trim=True, left=True)
    plt.show()


def population_creation(
    num_population: int,
    lower_k: int,
    upper_k: int,
    data_tuple: tuple,
    data_set_name: str,
    columns_names: list,
    knowledge: float = 0.95,
    knn_val: int = 5,
) -> Population:
    """
    Initialize a new population with random feature subsets.

    Parameters
    ----------
    num_population : int
        Number of individuals to create.
    lower_k : int
        Minimum number of features selected in the initial population.
    upper_k : int
        Maximum number of features selected in the initial population.
    data_tuple : tuple
        A tuple (X_train, X_test, y_train, y_test).
    data_set_name : str
        Name for the dataset.
    columns_names : list of str
        Names of available features.
    knowledge : float, default=0.95
        Initial knowledge threshold for the population.
    knn_val : int, default=5
        KNN parameter for the fitness evaluator.

    Returns
    -------
    Population
        An initialized population object.
    """
    
    X_train, X_test, y_train, y_test = data_tuple
    population = []
    total_features = X_train.shape[1]
    
    for i in range(num_population):
        k = random.randint(lower_k, upper_k)
        features_idx = random.sample(range(0, total_features), k)
        
        bin_feature = np.zeros(total_features, dtype=bool)
        bin_feature[features_idx] = True

        indiv = Individual(
            individual_id=i,
            features=bin_feature,
        )
        population.append(indiv)
        
    pop = Population(population, data_tuple=data_tuple, data_set_name=data_set_name, columns_names=columns_names, knowledge=knowledge, knn_val=knn_val)
    return pop


def feature_selection(
    data_tuple: tuple,
    num_population: int,
    nfe_total: int,
    lower_k: int,
    upper_k: int,
    columns_names: list,
    k: float = 0.95,
    p: float = 0.1,
    data_set_name: str = "dataset_1",
    knn_val: int = 5,
    time_limit: float = float('inf'),
) -> tuple[Population, np.ndarray, float]:
    """
    Execute the full pBGSK feature selection workflow.

    This function coordinates population creation, fitness evaluation,
    knowledge sharing iterations, and population reduction.

    Parameters
    ----------
    data_tuple : tuple
        A tuple (X_train, X_test, y_train, y_test).
    num_population : int
        Initial population size.
    nfe_total : int
        Total budget for function evaluations.
    lower_k : int
        Initial minimum feature selection count.
    upper_k : int
        Initial maximum feature selection count.
    columns_names : list of str
        Feature names.
    k : float, default=0.95
        Knowledge factor for dimension distribution.
    p : float, default=0.1
        Population partitioning ratio for intermediate phase.
    data_set_name : str, default="dataset_1"
        Dataset identifier.
    knn_val : int, default=5
        K value for the KNN classifier used in fitness evaluation.

    Returns
    -------
    pop : Population
        The final population state.
    best_individual_features : np.ndarray (bool)
        The feature mask of the best individual found.
    best_score : float
        The best fitness score achieved.
    """
    pop = population_creation(
        num_population=num_population,
        lower_k=lower_k,
        upper_k=upper_k,
        data_tuple=data_tuple,
        data_set_name=data_set_name,
        columns_names=columns_names,
        knowledge=k,
        knn_val=knn_val,
    )
    pop.partition = p

    # Avaliação Inicial
    for indiv in pop.individuals:
        calculate_population_fitness(pop, individual=indiv)

    get_population_dataframe(pop)
    sort_population(pop, t_sort="fitness")
    
    best_score = pop.individuals[0].score
    best_individual_features = pop.individuals[0].features.copy()

    import time
    start_time = time.time()

    while pop.nfe < nfe_total and pop.len > 12 and (time.time() - start_time) < time_limit:
        dimension_classification(pop, nfe_total=nfe_total)
        beginner_gsk(pop)
        intermediate_gsk(pop)
        
        for indiv in pop.individuals:
            calculate_population_fitness(pop, indiv)
            
        get_population_dataframe(pop)
        sort_population(pop, t_sort="fitness")
        
        # Guarda o melhor globalmente
        current_best = pop.individuals[0]
        if current_best.score < best_score:
            best_score = current_best.score
            best_individual_features = current_best.features.copy()
            
        # Reduz a população, se necessário
        population_reduction(pop, nfe_total=nfe_total)

    return pop, best_individual_features, best_score
