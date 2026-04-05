import unittest
import os
import sys
import psutil
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.pBGSK import feature_selection

class TestMemoryControl(unittest.TestCase):
    def setUp(self):
        # Default limit if benchmark hasn't been run or file is missing
        self.memory_limit_mb = 350.0 
        
        limit_file = os.path.join(os.path.dirname(__file__), "memory_limit.txt")
        if os.path.exists(limit_file):
            try:
                with open(limit_file, "r") as f:
                    self.memory_limit_mb = float(f.read().strip())
            except Exception:
                pass

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def test_memory_usage_standard_run(self):
        """Verify that a standard run stays within estimated memory limits."""
        # Use a medium-sized dummy dataset
        n_samples = 1000
        n_features = 50
        X = pd.DataFrame(np.random.rand(n_samples, n_features))
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        data_tuple = (X, X, y, y) # Same for train/test for simplicity
        
        # Measure peak memory
        mem_before = self.get_memory_usage()
        
        # Run feature selection
        feature_selection(
            data_tuple=data_tuple,
            num_population=20,
            nfe_total=100,
            lower_k=1,
            upper_k=n_features,
            columns_names=[f"f{i}" for i in range(n_features)],
            data_set_name="memory_test"
        )
        
        mem_after = self.get_memory_usage()
        
        print(f"\nMemory usage during test: {mem_after:.2f} MB")
        print(f"Memory limit: {self.memory_limit_mb:.2f} MB")
        
        self.assertLess(mem_after, self.memory_limit_mb, 
                        f"Memory usage {mem_after:.2f} MB exceeded limit {self.memory_limit_mb:.2f} MB")

if __name__ == "__main__":
    unittest.main()
