import os
import sys
import psutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from utils.data_importer import data_loader, DATASET_REGISTRY
    from utils.pBGSK import feature_selection
except ImportError:
    # Handle case where it's run from root
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
    from utils.data_importer import data_loader, DATASET_REGISTRY
    from utils.pBGSK import feature_selection

def get_memory_usage():
    """Returns the current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def benchmark_memory():
    results = {}
    print("Starting Memory Benchmark for pBGSK...")
    print("-" * 40)
    
    # We only use a subset of datasets if they are too many or too slow, 
    # but the user asked for "all datasets". 
    # To keep it reasonable for a benchmark, we'll use small nfe_total.
    
    for dataset_name in DATASET_REGISTRY.keys():
        print(f"Benchmarking dataset: {dataset_name}...", end=" ", flush=True)
        try:
            start_time = time.time()
            dataset = data_loader(dataset_name)
            if dataset is None:
                print("FAILED (data_loader returned None)")
                continue
                
            X = dataset.data.features
            y = dataset.data.targets
            
            # Ensure y is a 1D array/series
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            
            # Convert categorical to numeric
            X = pd.get_dummies(X)

            # Fill NaNs robustly
            for col in X.columns:
                if X[col].dtype.kind in 'iufc': # numeric
                    X[col] = X[col].fillna(X[col].mean())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)

            if isinstance(y, pd.Series):
                 y = y.fillna(y.mode()[0] if not y.mode().empty else 0)
            elif isinstance(y, pd.DataFrame):
                 y = y.fillna(y.mode().iloc[0] if not y.mode().empty else 0)
            
            if len(X) < 2:
                print(f"SKIPPED (too few samples: {len(X)})")
                continue
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            data_tuple = (X_train, X_test, y_train, y_test)
            
            mem_before = get_memory_usage()
            
            # Run a representative but short execution
            # 15 individuals (minimum > 12 for reduction logic)
            # 50 NFE (enough to trigger a few iterations)
            feature_selection(
                data_tuple=data_tuple,
                num_population=15,
                nfe_total=20,
                lower_k=1,
                upper_k=max(2, X_train.shape[1] // 2),
                columns_names=X.columns.tolist(),
                data_set_name=dataset_name,
                knn_val=3
            )
            
            mem_after = get_memory_usage()
            elapsed = time.time() - start_time
            
            results[dataset_name] = {
                "mem_after": mem_after,
                "mem_diff": mem_after - mem_before,
                "time": elapsed
            }
            print(f"DONE ({mem_after:.2f} MB, took {elapsed:.2f}s)")
            
        except Exception as e:
            print(f"FAILED: {e}")
            
    if results:
        mems = [r["mem_after"] for r in results.values()]
        max_mem = max(mems)
        avg_mem = sum(mems) / len(mems)
        
        print("-" * 40)
        print(f"Benchmark Summary:")
        print(f"  Total datasets tested: {len(results)}")
        print(f"  Average Memory Usage: {avg_mem:.2f} MB")
        print(f"  Maximum Memory Usage: {max_mem:.2f} MB")
        
        # Estimate a safe limit for tests
        # We add a buffer (e.g., 50% or 50MB, whichever is greater)
        safe_limit = max(max_mem * 1.5, max_mem + 50)
        print(f"  Estimated safe limit for tests: {safe_limit:.2f} MB")
        
        # Write to a config or return value
        with open("tests/memory_limit.txt", "w") as f:
            f.write(str(safe_limit))
            
        return safe_limit
    else:
        print("No datasets were successfully benchmarked.")
        return None

if __name__ == "__main__":
    benchmark_memory()
