import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


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
