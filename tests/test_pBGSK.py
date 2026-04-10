import unittest
import numpy as np
import pandas as pd
import random
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import pBGSK

class TestPBGSK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small deterministic dataset
        cls.X_train = pd.DataFrame([[1, 0], [1, 0], [0, 1], [0, 1]], columns=["f1", "f2"])
        cls.y_train = pd.Series([0, 0, 1, 1])
        cls.X_test = pd.DataFrame([[1, 0], [0, 1]], columns=["f1", "f2"])
        cls.y_test = pd.Series([0, 1])
        cls.data_tuple = (cls.X_train, cls.X_test, cls.y_train, cls.y_test)
        cls.columns_names = ["f1", "f2"]
        cls.dataset_name = "test_data"

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

                    self.assertEqual(int(individual.features[0]), expected)
        

    def test_feature_selector_evaluator(self):
        evaluator = pBGSK.FeatureSelectorEvaluator(*self.data_tuple, knn_val=1)
        
        # Test with all features
        features = np.array([True, True])
        score, acc = evaluator.calculate_fitness(features)
        self.assertEqual(acc, 1.0)
        # Score: (1-1.0) + (1-2/2) = 0.0
        self.assertEqual(score, 0.0)

        # Test with no features
        features = np.array([False, False])
        score, acc = evaluator.calculate_fitness(features)
        self.assertEqual(score, 2.0)
        self.assertEqual(acc, 0.0)

    def test_calculate_population_fitness(self):
        features = np.array([True, False])
        indiv = pBGSK.Individual(1, features)
        pop = pBGSK.Population([indiv], self.data_tuple, self.dataset_name, self.columns_names, knn_val=1)
        
        pBGSK.calculate_population_fitness(pop, indiv)
        
        self.assertEqual(indiv.acc, 1.0)
        # Score: (1-1.0) + (1-1/2) = 0.5
        self.assertEqual(indiv.score, 0.5)
        self.assertEqual(indiv.number_of_features, 1)

    def test_sort_population(self):
        indiv1 = pBGSK.Individual(1, [True, False])
        indiv2 = pBGSK.Individual(2, [True, True])
        
        pop = pBGSK.Population([indiv1, indiv2], self.data_tuple, self.dataset_name, self.columns_names, knn_val=1)
        pBGSK.evaluate_pending_individuals(pop)
        pBGSK.sort_population(pop, t_sort="fitness")
        
        # indiv2 score: 0.0 (better), indiv1 score: 0.5
        self.assertEqual(pop.individuals[0].individual_id, 2)
        self.assertEqual(pop.individuals[1].individual_id, 1)

    def test_evaluate_pending_individuals(self):
        indiv1 = pBGSK.Individual(1, [True, False])
        indiv2 = pBGSK.Individual(2, [True, True])
        pop = pBGSK.Population([indiv1, indiv2], self.data_tuple, self.dataset_name, self.columns_names, knn_val=1)

        evaluated = pBGSK.evaluate_pending_individuals(pop)

        self.assertEqual(evaluated, 2)
        self.assertEqual(indiv1.score, 0.5)
        self.assertEqual(indiv2.score, 0.0)

    def test_dimension_distribution(self):
        # Using a larger dummy population to test distribution
        pop = pBGSK.Population([], self.data_tuple, self.dataset_name, self.columns_names, knn_val=1)
        pop.individuals = [pBGSK.Individual(1, [True, True])]
        pop.nfe = 0
        pop.knowledge = 0.95
        
        nfe_total = 100
        diff = pBGSK.dimension_distribution(pop, nfe_total)
        # d=2. (1 - 0/100)^0.95 = 1.0. d_junior = min(round(2*1), 1) = 1.
        self.assertEqual(pop.d_junior, 1)
        self.assertEqual(pop.d_senior, 1)
        
        # After some NFE
        pop.nfe = 50
        # (1 - 50/100)^0.95 = 0.5^0.95 approx 0.517
        # d_junior = min(round(2*0.517), 1) = 1.
        pBGSK.dimension_distribution(pop, nfe_total)
        self.assertEqual(pop.d_junior, 1)

    def test_dimension_classification(self):
        pop = pBGSK.Population([], self.data_tuple, self.dataset_name, self.columns_names, knn_val=1)
        pop.individuals = [pBGSK.Individual(1, [True, True])]
        
        pBGSK.dimension_classification(pop, nfe_total=100)
        
        self.assertIsNotNone(pop.junior_features)
        self.assertIsNotNone(pop.senior_features)
        np.testing.assert_array_equal(pop.junior_features + pop.senior_features, [1, 1])

    def test_beginner_gsk_and_intermediate_gsk(self):
        # We need a population of at least 3 to run GSK safely (since it uses t-1 and t+1)
        # The loop is for t_idx in range(1, self.len - 1), so with 3 individuals, it only runs for t_idx = 1
        indiv0 = pBGSK.Individual(0, [True, False])
        indiv1 = pBGSK.Individual(1, [True, True])
        indiv2 = pBGSK.Individual(2, [False, True])
        
        pop = pBGSK.Population([indiv0, indiv1, indiv2], self.data_tuple, self.dataset_name, self.columns_names, knn_val=1)
        pop.junior_features = np.array([1, 0])
        pop.senior_features = np.array([0, 1])
        
        # Pre-calculate scores
        for ind in pop.individuals:
            pBGSK.calculate_population_fitness(pop, ind)
        
        pBGSK.beginner_gsk(pop)
        pBGSK.intermediate_gsk(pop)
        
        # Just check if it runs and maintains feature types
        self.assertIsInstance(pop.individuals[1].features[0], (bool, np.bool_))

    def test_population_reduction(self):
        pop = pBGSK.Population([], self.data_tuple, self.dataset_name, self.columns_names, knn_val=1)
        pop.individuals = [pBGSK.Individual(i, [True, True]) for i in range(20)]
        pop.len = 20
        # Initialize pop.df for population_reduction to work
        pBGSK.get_population_dataframe(pop)
        
        # Force a reduction
        pBGSK.population_reduction(pop, nfe_total=100, low_b=0.5, high_b=0.6)
        # nfe=0. np_new = int((0.5*20 - 0.6*20) * 0 + 0.6*20) = 12.
        self.assertEqual(pop.len, 12)
        self.assertEqual(len(pop.individuals), 12)

    def test_get_population_dataframe(self):
        indiv = pBGSK.Individual(1, [True, False])
        pop = pBGSK.Population([indiv], self.data_tuple, self.dataset_name, self.columns_names, knn_val=1)
        pBGSK.calculate_population_fitness(pop, indiv)
        
        df = pBGSK.get_population_dataframe(pop)
        self.assertEqual(len(df), 1)
        self.assertIn("score", df.columns)
        self.assertIn("n_features", df.columns)
        self.assertIn("acc", df.columns)
        self.assertEqual(df.loc[0, "n_features"], 1)

    def test_population_creation(self):
        pop = pBGSK.population_creation(
            num_population=10,
            lower_k=1,
            upper_k=2,
            data_tuple=self.data_tuple,
            data_set_name=self.dataset_name,
            columns_names=self.columns_names,
            knn_val=1
        )
        self.assertEqual(pop.len, 10)
        self.assertEqual(len(pop.individuals), 10)
        self.assertEqual(pop.data_set_name, self.dataset_name)

    def test_feature_selection_smoke_test(self):
        # End-to-end run with small parameters
        pop, best_features, best_score = pBGSK.feature_selection(
            data_tuple=self.data_tuple,
            num_population=15, # > 12 to run iterations
            nfe_total=50,
            lower_k=1,
            upper_k=2,
            columns_names=self.columns_names,
            data_set_name=self.dataset_name,
            knn_val=1
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
                knn_val=1
            )

    def test_feature_selection_keeps_running_at_minimum_population(self):
        pop, best_features, best_score = pBGSK.feature_selection(
            data_tuple=self.data_tuple,
            num_population=20,
            nfe_total=200,
            lower_k=1,
            upper_k=2,
            columns_names=self.columns_names,
            data_set_name=self.dataset_name,
            knn_val=1,
            time_limit=5.0
        )

        self.assertEqual(pop.len, 12)
        self.assertGreaterEqual(pop.nfe, 200)
        self.assertIsNotNone(best_features)
        self.assertLessEqual(best_score, 2.0)

if __name__ == "__main__":
    unittest.main()
