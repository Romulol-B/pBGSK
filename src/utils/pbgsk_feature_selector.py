import numpy as np
from sklearn.base import BaseEstimator, is_classifier
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import check_cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from .feature_mask import _as_feature_mask, _count_selected_features
from .pBGSK import CrossValidatedFeatureSelectorEvaluator, feature_selection


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
