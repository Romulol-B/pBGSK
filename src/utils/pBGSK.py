"""
Gaining Sharing Knowledge (GSK) algorithm for feature selection.

This module implements the pBGSK algorithm, a population-based optimization
technique inspired by the human process of gaining and sharing knowledge
throughout different life stages (Junior and Senior), tailored for the
feature selection task in machine learning.
"""

import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, check_cv, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


def _as_feature_mask(
    features: np.ndarray,
    n_features: int,
    *,# como isso funciona ?
    name: str = "features",
) -> np.ndarray:
    """
    Return a one-dimensional boolean feature mask with shape ``(n_features,)``.

    NumPy boolean indexing requires the mask length to match the selected axis.
    Validating that contract at the boundary gives clearer errors than letting
    column slicing fail deeper inside the evaluator.
    """
    mask = np.asarray(features, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional boolean mask with shape "
            f"({n_features},)."
        )
    if mask.shape[0] != n_features:
        raise ValueError(
            f"{name} length must match the number of features ({n_features}); "
            f"got {mask.shape[0]}."
        )
    return mask


def _count_selected_features(features: np.ndarray) -> int:
    """Count selected features in a boolean mask."""
    return int(np.count_nonzero(features))


def k_factor(kf: float = 1) -> int:
    """
    Stochastic knowledge factor multiplier.

    Returns 1 if a random value is less than or equal to the knowledge factor,
    otherwise returns 0. This is used to introduce randomness in the GSK update
    rules

    Parameters
    ----------
    kf : float, default=1
        The knowledge factor threshold.

    Returns
    -------
    int
        1 or 0 based on the stochastic comparison.
    """
    return 1 if random.random() <= kf else 0


class Individual:
    """
    Represents a single solution (individual) in the population.

    Each individual maintains a binary vector representing the selection of
    features and its associated fitness metrics.

    Parameters
    ----------
    individual_id : int
        Unique identifier for the individual.
    features : np.ndarray
        Binary or boolean array indicating which features are selected.
    """

    def __init__(
        self,
        individual_id: int,
        features: np.ndarray,
    ):
        self.individual_id = individual_id
        self.features = np.asarray(features, dtype=bool).copy()
        if self.features.ndim != 1:
            raise ValueError("features must be a one-dimensional feature mask.")
        self.acc = 0.0
        self.score = 2.0
        self.df = None

    def __len__(self):
        return _count_selected_features(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    def __setitem__(self, idx, value):
        self.features[idx] = value
    def __repr__(self):
        return (f"individual :{self.individual_id}\n"
                f" Score:{self.score} and Accuracy:{self.acc}"
                f"Number of Features:{_count_selected_features(self.features)}"
                f"Binary feature vector:{self.features+0}")#conversion to int.

def influence(
    individual: Individual,
    better: Individual,
    worse: Individual,
    rand_indiv: Individual,
    dimension,
    kf: float = 1,
):
    current_value = int(individual[dimension])

    if k_factor(kf) == 0:
        return individual[dimension]

    better_value = int(better[dimension])
    worse_value = int(worse[dimension])
    random_value = int(rand_indiv[dimension])

    if individual.score > rand_indiv.score:
        feature_influence = current_value + (
            better_value - worse_value + random_value - current_value
        )
    else:
        feature_influence = current_value + (
            better_value - worse_value + current_value - random_value
        )

    individual[dimension] = feature_influence > 0
    return individual[dimension]


def _cross_validation(
    dataset: pd.DataFrame, cross_validation_splits: int, test_proportion: float
):
    _ = test_proportion
    kf = KFold(n_splits=cross_validation_splits)
    return kf.get_n_splits(dataset)


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
    """

    def __init__(
        self, X_train, X_test, y_train, y_test, knn_val: int = 5, other_classifier=None
    ):
        self.X_train = np.asarray(X_train)
        self.X_test = np.asarray(X_test)
        self.y_train = np.asarray(y_train)
        self.y_test = np.asarray(y_test)
        if self.X_train.ndim != 2 or self.X_test.ndim != 2:
            raise ValueError(
                "X_train and X_test must be two-dimensional arrays with shape "
                "(n_samples, n_features)."
            )
        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError(
                "X_train and X_test must have the same number of feature columns."
            )
        self.n_features_in_ = self.X_train.shape[1]
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

        The fitness score follows the article objective:
        score = gamma1 * (1 - accuracy) + (1 - gamma1) * feature_ratio

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
        def _score_calculation(
            acc: np.float64,
            number_of_chosen_features: int,
            total_features: int,
            gamma1: float = 0.99,
        ):
            feature_ratio = number_of_chosen_features / total_features
            return np.float64(gamma1 * (1 - acc) + (1 - gamma1) * feature_ratio)

        features = _as_feature_mask(features, self.n_features_in_)
        number_of_features = _count_selected_features(features)
        if number_of_features == 0:
            return np.float64(2.0), np.float64(0.0)

        X_train_selected = self.X_train[:, features]
        X_test_selected = self.X_test[:, features]

        self.classifier.fit(X_train_selected, self.y_train)
        y_pred = self.classifier.predict(X_test_selected)

        acc = accuracy_score(self.y_test, y_pred)
        score = _score_calculation(acc, number_of_features, features.size)
        return score, np.float64(acc)


class CrossValidatedFeatureSelectorEvaluator:
    """
    Evaluate feature subsets using internal cross-validation.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Full feature matrix used for cross-validation.
    y : pd.Series or np.ndarray
        Target labels.
    estimator : estimator object, default=None
        The estimator evaluated on each selected subset. Defaults to KNN.
    scoring : str or callable, default="accuracy"
        Scikit-learn scoring specification.
    cv : int or CV splitter, default=5
        Cross-validation strategy.
    feature_penalty : float, default=1.0
        Linear penalty applied to the selected feature ratio.
    n_jobs : int, default=None
        Parallelism forwarded to ``cross_val_score``.
    """

    def __init__(
        self,
        X,
        y,
        estimator=None,
        scoring="accuracy",
        cv=5,
        feature_penalty: float = 1.0,
        n_jobs=None,
    ):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.estimator = (
            KNeighborsClassifier(n_neighbors=5) if estimator is None else estimator
        )
        self.scoring = scoring
        self.cv = cv
        self.feature_penalty = feature_penalty
        self.n_jobs = n_jobs
        if self.X.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional array with shape "
                "(n_samples, n_features)."
            )
        self.n_features_in_ = self.X.shape[1]

    def calculate_fitness(
        self,
        features: np.ndarray,
    ) -> tuple[np.float64, np.float64]:
        features = _as_feature_mask(features, self.n_features_in_)
        number_of_features = _count_selected_features(features)
        if number_of_features == 0:
            return np.float64(np.inf), np.float64(-np.inf)

        X_selected = self.X[:, features]
        scores = cross_val_score(
            clone(self.estimator),
            X_selected,
            self.y,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )

        mean_score = np.float64(np.mean(scores))
        feature_ratio = number_of_features / features.size
        fitness = np.float64(-mean_score + self.feature_penalty * feature_ratio)
        return fitness, mean_score


class Population:
    """
    Container for the population of individuals and optimization parameters.

    Parameters
    ----------
    individuals : list of Individual
        Initial list of individuals in the population.
    data_tuple : tuple
        A tuple containing (X_train, X_test, y_train, y_test).
    data_set_name : str
        Dataset identifier.
    columns_names : list[str]
        Feature names.
    knowledge : float, default=0.95
        Knowledge factor threshold for dimension distribution.
    partition : float, default=0.1
        Population partitioning ratio.
    knn_val : int, default=5
        Number of neighbors for the default KNN evaluator.
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
        evaluator=None,
    ):
        self.X_train, self.X_test, self.y_train, self.y_test = data_tuple
        self.data_set_name = data_set_name
        self.columns_names = columns_names
        self.individuals = individuals
        self.len = len(individuals)
        self.initial_population_size = self.len
        self.evaluator = (
            FeatureSelectorEvaluator(
                self.X_train, self.X_test, self.y_train, self.y_test, knn_val=knn_val
            )
            if evaluator is None
            else evaluator
        )

        self.df = None
        self.geng_df = pd.DataFrame()

        self.d_junior = 0
        self.d_senior = 0
        self.junior_features = None
        self.senior_features = None

        self.knowledge = knowledge
        self.partition = partition
        self.nfe = 0

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]

    def __setitem__(self, idx, value):
        raise ValueError("Operation not supported")


def calculate_population_fitness(apopulation: Population, individual: Individual):
    """
    Calculate and update the fitness of a specific individual within a population.

    Parameters
    ----------
    apopulation : Population
        The population context providing the evaluator.
    individual : Individual
        The individual whose fitness needs to be updated.
    """
    score, acc = apopulation.evaluator.calculate_fitness(individual.features)
    individual.score = score
    individual.acc = acc
    # individual.number_of_features = np.sum(individual.features)


def evaluate_population(apopulation: Population) -> int:
    """
    Evaluate every individual in the population.

    Returns
    -------
    int
        Number of fitness evaluations performed.
    """
    for indiv in apopulation.individuals:
        calculate_population_fitness(apopulation, indiv)
    return len(apopulation.individuals)


def evaluate_pending_individuals(apopulation: Population) -> int:
    """
    Evaluate only individuals that still have no computed fitness.

    Returns
    -------
    int
        Number of fitness evaluations performed.
    """
    evaluated = 0
    for indiv in apopulation.individuals:
        if indiv.score == 2.0 and _count_selected_features(indiv.features) > 0:
            calculate_population_fitness(apopulation, indiv)
            evaluated += 1
    return evaluated


def _avalition_checker(apopulation: Population) -> int:
    """Backward-compatible alias for pending population evaluation."""
    return evaluate_pending_individuals(apopulation)


def sort_population(apopulation: Population, t_sort: str = "fitness"):
    """
    Sort the population based on fitness or accuracy.

    Parameters
    ----------
    apopulation : Population
        The population to be sorted.
    t_sort : {"fitness", "accuracy"}, default="fitness"
        The metric used for sorting. "fitness" sorts in ascending order
        (lower is better), while "accuracy" sorts in descending order
        (higher is better).

    Raises
    ------
    ValueError
        If t_sort is not "fitness" or "accuracy".
    """
    if t_sort == "fitness":
        apopulation.individuals.sort(key=lambda ind: ind.score)
        if apopulation.df is not None:
            apopulation.df.sort_values("score", ascending=True, inplace=True)
    elif t_sort == "accuracy":
        apopulation.individuals.sort(key=lambda ind: ind.acc, reverse=True)
        if apopulation.df is not None:
            apopulation.df.sort_values("acc", ascending=False, inplace=True)
    else:
        raise ValueError("Tipo de sort invalido. Use 'fitness' ou 'accuracy'.")


def dimension_distribution(apopulation: Population, nfe_total: int) -> int:
    """
    Calculate the distribution of features between Junior and Senior stages.

    Parameters
    ----------
    apopulation : Population
        The population to update.
    nfe_total : int
        The total maximum number of function evaluations allowed.

    Returns
    -------
    int
        The difference between the previous and new Junior dimension counts.
    """
    d = len(apopulation.individuals[0].features)
    novo_d_junior = min(
        round(d * (1 - apopulation.nfe / nfe_total) ** apopulation.knowledge), d - 1
    )
    novo_d_senior = d - novo_d_junior
    diff = apopulation.d_junior - novo_d_junior

    apopulation.d_junior = novo_d_junior
    apopulation.d_senior = novo_d_senior

    return diff


def dimension_classification(apopulation: Population, nfe_total: int):
    """
    Assign specific features to Junior or Senior categories.

    Parameters
    ----------
    apopulation : Population
        The population to update.
    nfe_total : int
        The total maximum number of function evaluations allowed.
    """
    d = len(apopulation.individuals[0].features)

    if apopulation.d_junior == 0:
        dimension_distribution(apopulation, nfe_total)
        idxs = np.array(random.sample(range(0, d), apopulation.d_junior), dtype=int)

        apopulation.junior_features = np.zeros(d, dtype=int)
        if idxs.size > 0:
            apopulation.junior_features[idxs] = 1
        apopulation.senior_features = (
            np.ones(d, dtype=int) - apopulation.junior_features
        )
    else:
        diff = dimension_distribution(apopulation, nfe_total)
        jf = apopulation.junior_features.copy()
        sf = apopulation.senior_features.copy()

        while diff > 0:
            wh = np.squeeze(np.where(jf > 0))
            if wh.size == 0:
                break
            idx_r = random.choice(wh) if wh.ndim == 1 else wh
            jf[idx_r] = 0
            sf[idx_r] = 1
            diff -= 1

        apopulation.junior_features = jf
        apopulation.senior_features = sf


def beginner_gsk(apopulation: Population, knowledge_ratio: float = 0.95):
    """
    Apply the Junior (Beginner) stage knowledge sharing rules.

    Parameters
    ----------
    apopulation : Population
        The population to evolve.
    knowledge_ratio : float, default=0.95
        Probability that an individual will share/gain knowledge in a specific
        dimension.
    """
    for t_idx in range(1, apopulation.len - 1):
        for dimension in np.where(apopulation.junior_features > 0)[0]:
            if k_factor(knowledge_ratio):
                rand_idx = random.randint(0, apopulation.len - 1)
                rand_indiv = apopulation.individuals[rand_idx]
                xt = apopulation.individuals[t_idx]

                t_prev = apopulation.individuals[t_idx - 1]
                t_next = apopulation.individuals[t_idx + 1]
                influence(
                    individual=xt,
                    better=t_prev,
                    worse=t_next,
                    rand_indiv=rand_indiv,
                    dimension=dimension,
                )


def intermediate_gsk(apopulation: Population, knowledge_ratio: float = 0.95):
    """
    Apply the Senior (Intermediate) stage knowledge sharing rules.

    Parameters
    ----------
    apopulation : Population
        The population to evolve.
    knowledge_ratio : float, default=0.95
        Probability that an individual will share/gain knowledge in a specific
        dimension.
    """
    len_p = max(1, int(apopulation.len * apopulation.partition))

    for t_idx in range(1, apopulation.len - 1):
        for dimension in np.where(apopulation.senior_features > 0)[0]:
            if random.random() < knowledge_ratio:
                xt = apopulation.individuals[t_idx]

                best_x = apopulation.individuals[random.randint(0, len_p - 1)]
                middle_x = apopulation.individuals[
                    random.randint(len_p, apopulation.len - len_p - 1)
                ]
                worst_x = apopulation.individuals[
                    random.randint(apopulation.len - len_p, apopulation.len - 1)
                ]

                influence(
                    individual=xt,
                    better=best_x,
                    worse=worst_x,
                    rand_indiv=middle_x,
                    dimension=dimension,
                )

def _new_population_size(
    actual_nfe: int,
    total_nfe: int,
    np_max: int,
    np_min: int = 12,
):
    return round((np_min - np_max) * (actual_nfe / total_nfe) + np_max)

def population_reduction(
    apopulation: Population, nfe_total: int, low_b: float = 0.9, high_b: float = 0.95
) -> bool:
    """
    Perform Linear Population Size Reduction (LPSR).

    Parameters
    ----------
    apopulation : Population
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
        True if the population was not reduced, False otherwise.
    """
    km = apopulation.df.loc[:, ["score", "n_features", "acc"]].mean().to_frame().T
    km.rename(
        columns={
            "score": "mean_score",
            "n_features": "mean_n_features",
            "acc": "mean_acc",
        },
        inplace=True,
    )
    km["nfe"] = apopulation.nfe
    apopulation.geng_df = pd.concat([km, apopulation.geng_df], ignore_index=True)
    np_new = _new_population_size(
        actual_nfe=apopulation.nfe,
        total_nfe=nfe_total,
        np_max=apopulation.initial_population_size,
    )
    old_len = len(apopulation)
    if np_new >= 12 and np_new < old_len:
        amount_to_pop = old_len - np_new
        for _ in range(amount_to_pop):
            apopulation.individuals.pop()
        apopulation.len = len(apopulation.individuals)
        return False
    return True


def get_population_dataframe(apopulation: Population) -> pd.DataFrame:
    """
    Create a DataFrame representing the current population state.

    Parameters
    ----------
    apopulation : Population
        The population to extract data from.

    Returns
    -------
    pd.DataFrame
        A DataFrame where rows are individuals and columns are features, scores,
        counts, and accuracy.
    """
    n_features = len(apopulation.columns_names)
    if apopulation.individuals:
        feature_matrix = np.asarray(
            [
                _as_feature_mask(indiv.features, n_features)
                for indiv in apopulation.individuals
            ],
            dtype=bool,
        )
    else:
        feature_matrix = np.empty((0, n_features), dtype=bool)

    pop_df = pd.DataFrame(
        feature_matrix.astype(np.int8, copy=False),
        columns=apopulation.columns_names,
    )

    pop_df["score"] = np.fromiter(
        (indiv.score for indiv in apopulation.individuals),
        dtype=float,
        count=len(apopulation.individuals),
    )
    pop_df["n_features"] = np.count_nonzero(feature_matrix, axis=1)
    pop_df["acc"] = np.fromiter(
        (indiv.acc for indiv in apopulation.individuals),
        dtype=float,
        count=len(apopulation.individuals),
    )

    apopulation.df = pop_df
    return pop_df


def plot_population_score(apopulation: Population):
    """
    Visualize the distribution of scores across different features.

    Parameters
    ----------
    apopulation : Population
        The population to visualize.
    """
    df_melted = apopulation.df.melt(
        id_vars=["score", "n_features", "acc"],
        var_name="feature",
        value_name="is_selected",
    )
    df_p = df_melted[df_melted["is_selected"] == 1]

    _, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(
        data=df_p,
        x="score",
        y="feature",
        whis=[0, 100],
        width=0.6,
        palette="vlag",
        ax=ax,
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
    evaluator=None,
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
    X_train, X_test, y_train, y_test = data_tuple  # noqa: N806
    population = []
    total_features = X_train.shape[1]

    if num_population <= 0:
        raise ValueError("num_population must be greater than 0.")
    if total_features == 0:
        raise ValueError("The training data must contain at least one feature.")
    if not 1 <= lower_k <= upper_k <= total_features:
        raise ValueError(
            "lower_k and upper_k must satisfy 1 <= lower_k <= upper_k <= total_features."
        )
    if len(columns_names) != total_features:
        raise ValueError(
            "columns_names length must match the number of features in X_train."
        )

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

    apopulation = Population(
        population,
        data_tuple=data_tuple,
        data_set_name=data_set_name,
        columns_names=columns_names,
        knowledge=knowledge,
        knn_val=knn_val,
        evaluator=evaluator,
    )
    return apopulation


def get_best_features(apopulation: Population):
    return apopulation[0].features


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
    time_limit: float = float("inf"),
    evaluator=None,
    random_state=None,
) -> tuple[Population, np.ndarray, float]:
    """
    Execute the full pBGSK feature selection workflow.

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
    apopulation : Population
        The final population state.
    best_individual_features : np.ndarray (bool)
        The feature mask of the best individual found.
    best_score : float
        The best fitness score achieved.
    """
    if len(data_tuple) != 4:
        raise ValueError(
            "data_tuple must contain X_train, X_test, y_train, and y_test."
        )

    X_train, X_test, y_train, y_test = data_tuple  # noqa: N806

    if num_population <= 12:
        raise ValueError(
            "num_population must be greater than 12 so the algorithm can reduce the population safely."
        )
    if nfe_total <= 0:
        raise ValueError("nfe_total must be greater than 0.")
    if nfe_total < num_population:
        raise ValueError(
            "nfe_total must be at least num_population so the initial population can be evaluated within the budget."
        )
    if not 0 < p < 0.5:
        raise ValueError(
            "p must be between 0 and 0.5 so the senior groups remain valid."
        )
    if knn_val <= 0:
        raise ValueError("knn_val must be greater than 0.")
    if evaluator is None and len(X_train) < knn_val:
        raise ValueError(
            "knn_val cannot be greater than the number of training samples."
        )
    if time_limit <= 0:
        raise ValueError("time_limit must be greater than 0.")

    python_random_state = None
    numpy_random_state = None
    if random_state is not None:
        python_random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        random.seed(random_state)
        np.random.seed(random_state)

    try:
        apopulation = population_creation(
            num_population=num_population,
            lower_k=lower_k,
            upper_k=upper_k,
            data_tuple=data_tuple,
            data_set_name=data_set_name,
            columns_names=columns_names,
            knowledge=k,
            knn_val=knn_val,
            evaluator=evaluator,
        )
        apopulation.partition = p

        apopulation.nfe += evaluate_population(apopulation)

        get_population_dataframe(apopulation)
        sort_population(apopulation, t_sort="fitness")

        best_score = apopulation[0].score
        best_individual_features = apopulation[0].features.copy()

        start_time = time.time()

        while (
            apopulation.nfe + apopulation.len <= nfe_total
            and (time.time() - start_time) < time_limit
        ):
            dimension_classification(apopulation, nfe_total=nfe_total)
            beginner_gsk(apopulation)
            intermediate_gsk(apopulation)

            apopulation.nfe += evaluate_population(apopulation)

            get_population_dataframe(apopulation)
            sort_population(apopulation, t_sort="fitness")

            current_best = apopulation[0]
            if current_best.score < best_score:
                best_score = current_best.score
                best_individual_features = current_best.features.copy()

            population_reduction(apopulation, nfe_total=nfe_total)

        return apopulation, best_individual_features, best_score
    finally:
        if random_state is not None:
            random.setstate(python_random_state)
            np.random.set_state(numpy_random_state)


class PBGSKFeatureSelector(SelectorMixin, BaseEstimator):
    """
    Scikit-learn compatible pBGSK-based supervised feature selector.

    This wrapper keeps the original population-based optimizer as the search
    engine while evaluating candidate subsets with internal cross-validation.
    The selector currently supports classification only.
    """

    def __init__(
        self,
        estimator=None,
        scoring="accuracy",
        cv=5,
        population_size: int = 20,
        nfe_total: int = 100,
        lower_k: int = 1,
        upper_k: int | None = None,
        knowledge_factor: float = 0.95,
        partition: float = 0.1,
        feature_penalty: float = 1.0,
        random_state=None,
        time_limit: float = float("inf"),
        n_jobs=None,
    ):
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.population_size = population_size
        self.nfe_total = nfe_total
        self.lower_k = lower_k
        self.upper_k = upper_k
        self.knowledge_factor = knowledge_factor
        self.partition = partition
        self.feature_penalty = feature_penalty
        self.random_state = random_state
        self.time_limit = time_limit
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X_validated, y_validated = validate_data(
            self,
            X,
            y,
            ensure_min_samples=2,
            ensure_min_features=1,
        )
        check_classification_targets(y_validated)

        cv = check_cv(self.cv, y=y_validated, classifier=True)
        if self.estimator is None:
            min_train_size = min(
                len(train_idx) for train_idx, _ in cv.split(X_validated, y_validated)
            )
            estimator = KNeighborsClassifier(n_neighbors=min(5, min_train_size))
        else:
            estimator = self.estimator
        if not is_classifier(estimator):
            raise ValueError(
                "PBGSKFeatureSelector currently supports classification estimators only."
            )
        if self.feature_penalty < 0:
            raise ValueError("feature_penalty must be greater than or equal to 0.")

        n_features = X_validated.shape[1]
        upper_k = n_features if self.upper_k is None else self.upper_k
        columns_names = (
            self.feature_names_in_.tolist()
            if hasattr(self, "feature_names_in_")
            else [f"x{i}" for i in range(n_features)]
        )

        rng = check_random_state(self.random_state)
        seed = int(rng.randint(np.iinfo(np.int32).max))

        evaluator = CrossValidatedFeatureSelectorEvaluator(
            X=X_validated,
            y=y_validated,
            estimator=estimator,
            scoring=self.scoring,
            cv=cv,
            feature_penalty=self.feature_penalty,
            n_jobs=self.n_jobs,
        )
        apopulation, best_features, _ = feature_selection(
            data_tuple=(X_validated, X_validated, y_validated, y_validated),
            num_population=self.population_size,
            nfe_total=self.nfe_total,
            lower_k=self.lower_k,
            upper_k=upper_k,
            columns_names=columns_names,
            k=self.knowledge_factor,
            p=self.partition,
            data_set_name=self.__class__.__name__,
            knn_val=5,
            time_limit=self.time_limit,
            evaluator=evaluator,
            random_state=seed,
        )

        support = _as_feature_mask(best_features, n_features, name="best_features")
        best_fitness, best_metric = evaluator.calculate_fitness(support)

        self.support_ = support
        self.best_fitness_ = float(best_fitness)
        self.best_cv_score_ = float(best_metric)
        self.n_features_selected_ = _count_selected_features(support)
        self.population_ = apopulation
        self.generation_history_ = apopulation.geng_df.copy()

        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_

    def _more_tags(self):
        return {"requires_y": True}
