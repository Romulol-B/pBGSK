import unittest
import numpy as np
import pandas as pd
import random
import os
import sys
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import pBGSK
from src.utils.data_importer import DATASET_REGISTRY


class TestPBGSK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small deterministic dataset
        cls.X_train = pd.DataFrame(
            [[1, 0], [1, 0], [0, 1], [0, 1]], columns=["f1", "f2"]
        )
        cls.y_train = pd.Series([0, 0, 1, 1])
        cls.X_test = pd.DataFrame([[1, 0], [0, 1]], columns=["f1", "f2"])
        cls.y_test = pd.Series([0, 1])
        cls.data_tuple = (cls.X_train, cls.X_test, cls.y_train, cls.y_test)
        cls.columns_names = ["f1", "f2"]
        cls.dataset_name = "test_data"
        cls.selector_X = pd.DataFrame(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 0, 1, 1],
                [1, 1, 1, 1],
            ],
            columns=["s1", "s2", "s3", "s4"],
        )
        cls.selector_y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])

    def setUp(self):
        # Reset seeds for each test
        random.seed(42)
        np.random.seed(42)

    def test_individual_init(self):
        features = np.array([True, False])
        indiv = pBGSK.Individual(1, features)
        self.assertEqual(indiv.individual_id, 1)
        np.testing.assert_array_equal(indiv.features, features)

    def test_influence_matches_readme_junior_case_1_table(self):
        cases = [
            (0, 0, 0, 0),
            (0, 0, 1, 1),
            (1, 1, 0, 0),
            (1, 1, 1, 1),
            (1, 0, 0, 1),
            (1, 0, 1, 1),
            (0, 1, 0, 0),
            (0, 1, 1, 0),
        ]

        for current_value in (0, 1):
            for better_value, worse_value, random_value, expected in cases:
                with self.subTest(
                    current_value=current_value,
                    better_value=better_value,
                    worse_value=worse_value,
                    random_value=random_value,
                ):
                    individual = pBGSK.Individual(0, [current_value])
                    better = pBGSK.Individual(1, [better_value])
                    worse = pBGSK.Individual(2, [worse_value])
                    rand_indiv = pBGSK.Individual(3, [random_value])

                    individual.score = 1.0
                    rand_indiv.score = 0.5

                    pBGSK.influence(
                        individual=individual,
                        better=better,
                        worse=worse,
                        rand_indiv=rand_indiv,
                        dimension=0,
                        kf=1,
                    )

                    self.assertEqual(int(individual[0]), expected)

    def test_feature_selector_evaluator(self):
        evaluator = pBGSK.FeatureSelectorEvaluator(*self.data_tuple, knn_val=1)

        # Test with all features
        features = np.array([True, True])
        score, acc = evaluator.calculate_fitness(features)
        self.assertEqual(acc, 1.0)
        # Score: 0.99 * (1 - 1.0) + 0.01 * (2 / 2) = 0.01
        self.assertAlmostEqual(score, 0.01)

        # Test with no features
        features = np.array([False, False])
        score, acc = evaluator.calculate_fitness(features)
        self.assertEqual(score, 2.0)
        self.assertEqual(acc, 0.0)

    def test_calculate_population_fitness(self):
        features = np.array([True, False])
        indiv = pBGSK.Individual(1, features)
        apopulation = pBGSK.Population(
            [indiv], self.data_tuple, self.dataset_name, self.columns_names, knn_val=1
        )

        pBGSK.calculate_population_fitness(apopulation, indiv)

        self.assertEqual(indiv.acc, 1.0)
        # Score: 0.99 * (1 - 1.0) + 0.01 * (1 / 2) = 0.005
        self.assertAlmostEqual(indiv.score, 0.005)
        # self.assertEqual(indiv.number_of_features, 1)
        self.assertEqual(len(indiv), 1)

    def test_sort_population(self):
        indiv1 = pBGSK.Individual(1, [True, False])
        indiv2 = pBGSK.Individual(2, [True, True])

        apopulation = pBGSK.Population(
            [indiv1, indiv2],
            self.data_tuple,
            self.dataset_name,
            self.columns_names,
            knn_val=1,
        )
        pBGSK.evaluate_pending_individuals(apopulation)
        pBGSK.sort_population(apopulation, t_sort="fitness")

        # Equal accuracy; the article fitness favors the individual with fewer features.
        self.assertEqual(apopulation.individuals[0].individual_id, 1)
        self.assertEqual(apopulation.individuals[1].individual_id, 2)

    def test_population_len(self):
        indiv1 = pBGSK.Individual(1, [True, False])
        indiv2 = pBGSK.Individual(2, [True, True])
        apopulation = pBGSK.Population(
            [indiv1, indiv2],
            self.data_tuple,
            self.dataset_name,
            self.columns_names,
            knn_val=1,
        )

        self.assertEqual(len(apopulation), 2)

    def test_evaluate_pending_individuals(self):
        indiv1 = pBGSK.Individual(1, [True, False])
        indiv2 = pBGSK.Individual(2, [True, True])
        apopulation = pBGSK.Population(
            [indiv1, indiv2],
            self.data_tuple,
            self.dataset_name,
            self.columns_names,
            knn_val=1,
        )

        evaluated = pBGSK.evaluate_pending_individuals(apopulation)

        self.assertEqual(evaluated, 2)
        self.assertAlmostEqual(indiv1.score, 0.005)
        self.assertAlmostEqual(indiv2.score, 0.01)

    def test_dimension_distribution(self):
        # Using a larger dummy population to test distribution
        apopulation = pBGSK.Population(
            [], self.data_tuple, self.dataset_name, self.columns_names, knn_val=1
        )
        apopulation.individuals = [pBGSK.Individual(1, [True, True])]
        apopulation.nfe = 0
        apopulation.knowledge = 0.95

        nfe_total = 100
        diff = pBGSK.dimension_distribution(apopulation, nfe_total)  # esse diff ai é paia
        # d=2. (1 - 0/100)^0.95 = 1.0. d_junior = min(round(2*1), 1) = 1.
        self.assertEqual(apopulation.d_junior, 1)
        self.assertEqual(apopulation.d_senior, 1)

        # After some NFE
        apopulation.nfe = 50
        # (1 - 50/100)^0.95 = 0.5^0.95 approx 0.517
        # d_junior = min(round(2*0.517), 1) = 1.
        pBGSK.dimension_distribution(apopulation, nfe_total)
        self.assertEqual(apopulation.d_junior, 1)

    def test_cross_validation_returns_split_count(self):
        splits = pBGSK._cross_validation(
            dataset=self.X_train,
            cross_validation_splits=2,
            test_proportion=0.2,
        )

        self.assertEqual(splits, 2)

    def test_dimension_classification(self):
        apopulation = pBGSK.Population(
            [], self.data_tuple, self.dataset_name, self.columns_names, knn_val=1
        )
        apopulation.individuals = [pBGSK.Individual(1, [True, True])]

        pBGSK.dimension_classification(apopulation, nfe_total=100)

        self.assertIsNotNone(apopulation.junior_features)
        self.assertIsNotNone(apopulation.senior_features)
        np.testing.assert_array_equal(apopulation.junior_features + apopulation.senior_features, [1, 1])

    def test_beginner_gsk_and_intermediate_gsk(self):
        # We need a population of at least 3 to run GSK safely (since it uses t-1 and t+1)
        # The loop is for t_idx in range(1, self.len - 1), so with 3 individuals, it only runs for t_idx = 1
        indiv0 = pBGSK.Individual(0, [True, False])
        indiv1 = pBGSK.Individual(1, [True, True])
        indiv2 = pBGSK.Individual(2, [False, True])

        apopulation = pBGSK.Population(
            [indiv0, indiv1, indiv2],
            self.data_tuple,
            self.dataset_name,
            self.columns_names,
            knn_val=1,
        )
        apopulation.junior_features = np.array([1, 0])
        apopulation.senior_features = np.array([0, 1])

        # Pre-calculate scores
        for ind in apopulation.individuals:
            pBGSK.calculate_population_fitness(apopulation, ind)

        pBGSK.beginner_gsk(apopulation)
        pBGSK.intermediate_gsk(apopulation)

        # Just check if it runs and maintains feature types
        secondindividual = apopulation[1]
        self.assertIsInstance(secondindividual[0], (bool, np.bool_))

    def test_population_reduction(self):
        apopulation = pBGSK.Population(
            [pBGSK.Individual(i, [True, True]) for i in range(20)],
            self.data_tuple,
            self.dataset_name,
            self.columns_names,
            knn_val=1,
        )
        # Initialize apopulation.df for population_reduction to work
        pBGSK.get_population_dataframe(apopulation)

        # Force a reduction to the article minimum population size.
        apopulation.nfe = 100
        pBGSK.population_reduction(apopulation, nfe_total=100, low_b=0.5, high_b=0.6)
        # NPG+1 = round((12 - 20) * (100 / 100) + 20) = 12.
        self.assertEqual(apopulation.len, 12)
        self.assertEqual(len(apopulation.individuals), 12)

    def test_get_population_dataframe(self):
        indiv = pBGSK.Individual(1, [True, False])
        apopulation = pBGSK.Population(
            [indiv], self.data_tuple, self.dataset_name, self.columns_names, knn_val=1
        )
        pBGSK.calculate_population_fitness(apopulation, indiv)

        df = pBGSK.get_population_dataframe(apopulation)
        self.assertEqual(len(df), 1)
        self.assertIn("score", df.columns)
        self.assertIn("n_features", df.columns)
        self.assertIn("acc", df.columns)
        self.assertEqual(df.loc[0, "n_features"], 1)

    def test_population_creation(self):
        apopulation = pBGSK.population_creation(
            num_population=10,
            lower_k=1,
            upper_k=2,
            data_tuple=self.data_tuple,
            data_set_name=self.dataset_name,
            columns_names=self.columns_names,
            knn_val=1,
        )
        self.assertEqual(apopulation.len, 10)
        self.assertEqual(len(apopulation.individuals), 10)
        self.assertEqual(apopulation.data_set_name, self.dataset_name)

    def test_feature_selection_smoke_test(self):
        # End-to-end run with small parameters
        apopulation, best_features, best_score = pBGSK.feature_selection(
            data_tuple=self.data_tuple,
            num_population=15,  # > 12 to run iterations
            nfe_total=50,
            lower_k=1,
            upper_k=2,
            columns_names=self.columns_names,
            data_set_name=self.dataset_name,
            knn_val=1,
        )
        self.assertIsNotNone(best_features)
        self.assertLessEqual(best_score, 2.0)

    def test_feature_selection_validates_population_size(self):
        with self.assertRaises(ValueError):
            pBGSK.feature_selection(
                data_tuple=self.data_tuple,
                num_population=12,
                nfe_total=50,
                lower_k=1,
                upper_k=2,
                columns_names=self.columns_names,
                data_set_name=self.dataset_name,
                knn_val=1,
            )

    def test_feature_selection_validates_nfe_total_against_population_size(self):
        with self.assertRaises(ValueError):
            pBGSK.feature_selection(
                data_tuple=self.data_tuple,
                num_population=13,
                nfe_total=12,
                lower_k=1,
                upper_k=2,
                columns_names=self.columns_names,
                data_set_name=self.dataset_name,
                knn_val=1,
            )

    def test_feature_selection_respects_evaluation_budget(self):
        apopulation, best_features, best_score = pBGSK.feature_selection(
            data_tuple=self.data_tuple,
            num_population=20,
            nfe_total=200,
            lower_k=1,
            upper_k=2,
            columns_names=self.columns_names,
            data_set_name=self.dataset_name,
            knn_val=1,
            time_limit=5.0,
        )

        self.assertEqual(apopulation.len, 12)
        self.assertLessEqual(apopulation.nfe, 200)
        self.assertIsNotNone(best_features)
        self.assertLessEqual(best_score, 2.0)

    def test_dataset_registry_contains_readme_example_dataset(self):
        self.assertIn("breast_cancer", DATASET_REGISTRY)

    def test_selector_fit_and_transform(self):
        selector = pBGSK.PBGSKFeatureSelector(
            population_size=13,
            nfe_total=13,
            lower_k=1,
            upper_k=3,
            cv=2,
            random_state=7,
        )

        transformed = selector.fit_transform(self.selector_X, self.selector_y)

        self.assertEqual(transformed.shape[0], len(self.selector_X))
        self.assertEqual(transformed.shape[1], selector.n_features_selected_)
        np.testing.assert_array_equal(
            selector.get_support(),
            selector._get_support_mask(),
        )
        self.assertTrue(hasattr(selector, "feature_names_in_"))

    def test_selector_is_deterministic_with_random_state(self):
        selector_1 = pBGSK.PBGSKFeatureSelector(
            population_size=13,
            nfe_total=13,
            lower_k=1,
            upper_k=3,
            cv=2,
            random_state=11,
        )
        selector_2 = pBGSK.PBGSKFeatureSelector(
            population_size=13,
            nfe_total=13,
            lower_k=1,
            upper_k=3,
            cv=2,
            random_state=11,
        )

        selector_1.fit(self.selector_X, self.selector_y)
        selector_2.fit(self.selector_X, self.selector_y)

        np.testing.assert_array_equal(
            selector_1.get_support(),
            selector_2.get_support(),
        )
        self.assertEqual(selector_1.best_fitness_, selector_2.best_fitness_)
        self.assertEqual(selector_1.best_cv_score_, selector_2.best_cv_score_)

    def test_selector_supports_clone(self):
        selector = pBGSK.PBGSKFeatureSelector(
            population_size=13,
            nfe_total=13,
            lower_k=1,
            upper_k=3,
            cv=2,
            random_state=5,
        )

        cloned = clone(selector)

        self.assertEqual(cloned.population_size, 13)
        self.assertEqual(cloned.nfe_total, 13)
        self.assertEqual(cloned.random_state, 5)

    def test_selector_supports_pipeline_and_grid_search(self):
        pipeline = Pipeline(
            [
                (
                    "selector",
                    pBGSK.PBGSKFeatureSelector(
                        population_size=13,
                        nfe_total=13,
                        lower_k=1,
                        upper_k=3,
                        cv=2,
                        random_state=3,
                    ),
                ),
                ("classifier", KNeighborsClassifier(n_neighbors=1)),
            ]
        )
        search = GridSearchCV(
            pipeline,
            param_grid={"selector__partition": [0.1, 0.2]},
            cv=2,
        )

        search.fit(self.selector_X, self.selector_y)

        self.assertIn("selector__partition", search.best_params_)
        self.assertIsInstance(
            search.best_estimator_.named_steps["selector"],
            pBGSK.PBGSKFeatureSelector,
        )

    def test_selector_rejects_non_classifier_estimators(self):
        selector = pBGSK.PBGSKFeatureSelector(
            estimator=LinearRegression(),
            population_size=13,
            nfe_total=13,
            lower_k=1,
            upper_k=3,
            cv=2,
            random_state=3,
        )

        with self.assertRaises(ValueError):
            selector.fit(self.selector_X, self.selector_y)


if __name__ == "__main__":
    unittest.main()
