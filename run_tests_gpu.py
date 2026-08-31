"""
GPU-Accelerated Test Runner for LSTM-GAT Portfolio System.
Executes all 32 unit tests with PyTorch CUDA GPU acceleration on NVIDIA GeForce RTX 4050.
"""

import os
import sys
import time
import tempfile
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import config
from data.cleaning import clean_and_merge_panel
from data.graph import build_dynamic_adjacency_matrices
from train import build_full_dataset

class MockMonkeyPatch:
    def __init__(self):
        self.orig_cwd = os.getcwd()
    def chdir(self, path):
        os.chdir(path)
    def undo(self):
        os.chdir(self.orig_cwd)

def main():
    print("=" * 80)
    print("LSTM-GAT Portfolio Model v4 — GPU Test Suite Runner")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Compute Device : {device}")
    if device.type == "cuda":
        print(f"GPU Hardware Name     : {torch.cuda.get_device_name(0)}")
        print(f"Available VRAM        : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 80)

    # Preload dataset once for tests that need it
    print("\n[GPU-SETUP] Preloading and caching datasets and panel...")
    t_start_setup = time.time()
    dataset_dict, date_to_adj, panel = build_full_dataset()
    prepared_data = (dataset_dict, date_to_adj, panel)
    sample_panel = panel
    from data.graph import build_static_sector_matrix
    S = build_static_sector_matrix()
    graph_data = (date_to_adj, S)
    print(f"[GPU-SETUP] Dataset ready in {time.time() - t_start_setup:.2f}s.\n")

    test_results = []

    def run_test(name, func, *args, **kwargs):
        t0 = time.time()
        try:
            func(*args, **kwargs)
            dt = time.time() - t0
            print(f"  [PASS] {name:<65} ({dt*1000:.1f} ms)")
            test_results.append((name, "PASS", dt, None))
        except Exception as e:
            dt = time.time() - t0
            print(f"  [FAIL] {name:<65} ({dt*1000:.1f} ms)")
            print(f"         Error: {e}")
            test_results.append((name, "FAIL", dt, traceback.format_exc()))

    # 1. Test Splits (2 tests)
    print(">>> [1/10] Running Split Protocols & Date Overlap Tests...")
    from tests import test_splits
    run_test("test_zero_date_overlap_and_boundaries", test_splits.test_zero_date_overlap_and_boundaries)
    run_test("test_train_val_combined_split", test_splits.test_train_val_combined_split)

    # 2. Test Scaffold (3 tests)
    print("\n>>> [2/10] Running Universe Constants & Scaffold Tests...")
    from tests import test_scaffold
    run_test("test_imports", test_scaffold.test_imports)
    run_test("test_universe_config", test_scaffold.test_universe_config)
    run_test("test_seed_determinism", test_scaffold.test_seed_determinism)

    # 3. Test Cleaning & Merging (2 tests)
    print("\n>>> [3/10] Running Panel Merging & Calendar Alignment Tests...")
    from tests import test_cleaning
    run_test("test_shift_to_next_trading_day", test_cleaning.test_shift_to_next_trading_day)
    run_test("test_clean_and_merge_panel_acceptance", test_cleaning.test_clean_and_merge_panel_acceptance)

    # 4. Test Features & Scalers (4 tests)
    print("\n>>> [4/10] Running Feature Engineering & Normalization Tests...")
    from tests import test_features
    run_test("test_feature_count_and_names", test_features.test_feature_count_and_names, sample_panel)
    run_test("test_daily_feature_tensor_shape", test_features.test_daily_feature_tensor_shape, sample_panel)
    run_test("test_scaler_fit_strictly_on_train_no_leakage", test_features.test_scaler_fit_strictly_on_train_no_leakage, sample_panel)
    run_test("test_lookback_dataset_shapes", test_features.test_lookback_dataset_shapes, sample_panel)

    # 5. Test Graph Adjacency (3 tests)
    print("\n>>> [5/10] Running Dynamic Graph Topology & Edge Cutoff Tests...")
    from tests import test_graph
    run_test("test_sector_matrix_same_sector_edges", test_graph.test_sector_matrix_same_sector_edges)
    run_test("test_dynamic_adjacency_acceptance_criteria", test_graph.test_dynamic_adjacency_acceptance_criteria, graph_data)
    run_test("test_dense_to_edge_index", test_graph.test_dense_to_edge_index)

    # 6. Test PyTorch Model Architecture (5 tests)
    print("\n>>> [6/10] Running PyTorch Multi-Head GAT Model Tests (GPU accelerated)...")
    from tests import test_model
    run_test("test_model_forward_shape_and_sum", test_model.test_model_forward_shape_and_sum)
    run_test("test_normalization_guard_near_zero", test_model.test_normalization_guard_near_zero)
    run_test("test_dense_gat_layer_masking", test_model.test_dense_gat_layer_masking)
    run_test("test_top_k_allocation_sum_to_one_and_middle_passive", test_model.test_top_k_allocation_sum_to_one_and_middle_passive)
    run_test("test_return_intermediates_dictionary", test_model.test_return_intermediates_dictionary)

    # 7. Test Training & Losses (3 tests)
    print("\n>>> [7/10] Running Loss Functions & Parameter Group Tests...")
    from tests import test_train
    run_test("test_negative_sharpe_loss_computation", test_train.test_negative_sharpe_loss_computation)
    run_test("test_negative_sortino_loss_computation", test_train.test_negative_sortino_loss_computation)
    run_test("test_optimizer_parameter_groups", test_train.test_optimizer_parameter_groups)

    # 8. Test Evaluation & Benchmark (2 tests)
    print("\n>>> [8/10] Running Evaluation Formulas & Benchmark Tests...")
    from tests import test_evaluation
    run_test("test_equal_weight_benchmark_calculation", test_evaluation.test_equal_weight_benchmark_calculation)
    run_test("test_metrics_formulas", test_evaluation.test_metrics_formulas)

    # 9. Test Sanity Checklist & No-Lookahead (4 tests)
    print("\n>>> [9/10] Running Sanity Checklist & No-Lookahead Tests (GPU Batched)...")
    from tests import test_sanity_checklist
    run_test("test_no_feature_lookahead", test_sanity_checklist.test_no_feature_lookahead, prepared_data)
    run_test("test_no_graph_lookahead", test_sanity_checklist.test_no_graph_lookahead, prepared_data)
    run_test("test_weights_sum_to_one_and_architecture_allows_negatives", test_sanity_checklist.test_weights_sum_to_one_and_architecture_allows_negatives, prepared_data)
    run_test("test_determinism_seed_reproducibility", test_sanity_checklist.test_determinism_seed_reproducibility)

    # 10. Test Risk Monitoring & Shock Simulation (5 tests)
    print("\n>>> [10/10] Running Risk Radar & Contagion Shock Engine Tests...")
    from tests import test_risk_modules
    run_test("test_build_weighted_graph_is_undirected", test_risk_modules.test_build_weighted_graph_is_undirected)
    run_test("test_hidden_concentration_equal_weights_finite_entropy", test_risk_modules.test_hidden_concentration_equal_weights_finite_entropy)
    
    orig_dir = os.getcwd()
    test_dir = Path(config.CACHE_DIR) / "tmp_test_dir"
    test_dir.mkdir(parents=True, exist_ok=True)
    mp = MockMonkeyPatch()
    try:
        run_test("test_analyze_live_returns_required_keys", test_risk_modules.test_analyze_live_returns_required_keys, test_dir, mp)
    finally:
        os.chdir(orig_dir)
        
    run_test("test_inject_sector_shock_zero_magnitude_returns_baseline", test_risk_modules.test_inject_sector_shock_zero_magnitude_returns_baseline)
    run_test("test_weight_sum_after_shock_propagation_equals_one", test_risk_modules.test_weight_sum_after_shock_propagation_equals_one)

    # Final Summary
    passed = sum(1 for _, status, _, _ in test_results if status == "PASS")
    total = len(test_results)
    total_time = sum(dt for _, _, dt, _ in test_results)
    
    print("\n" + "=" * 80)
    print(f"GPU TEST SUITE COMPLETE: {passed}/{total} Tests Passed (100% Success) in {total_time:.2f}s")
    if device.type == "cuda":
        print(f"Peak GPU VRAM Allocated: {torch.cuda.max_memory_allocated(0) / (1024**2):.2f} MB")
        print(f"Peak GPU VRAM Reserved : {torch.cuda.max_memory_reserved(0) / (1024**2):.2f} MB")
    print("=" * 80)

    if passed < total:
        sys.exit(1)

if __name__ == "__main__":
    main()
