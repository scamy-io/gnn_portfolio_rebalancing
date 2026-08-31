"""
Unit tests for dynamic graph construction and topology invariants.
"""

import numpy as np
import pytest
import config
from data.cleaning import clean_and_merge_panel
from data.graph import (
    build_dynamic_adjacency_matrices,
    build_static_sector_matrix,
    dense_adj_to_edge_index,
)


@pytest.fixture(scope="module")
def graph_data():
    panel = clean_and_merge_panel()
    date_to_adj, S = build_dynamic_adjacency_matrices(panel)
    return date_to_adj, S


def test_sector_matrix_same_sector_edges():
    """Assert same-sector pairs always show an edge in sector matrix."""
    S = build_static_sector_matrix()
    tickers = config.TICKER_LIST
    
    aapl_idx = tickers.index("AAPL")
    nvda_idx = tickers.index("NVDA")
    jnj_idx = tickers.index("JNJ")
    tmo_idx = tickers.index("TMO")
    tsla_idx = tickers.index("TSLA")
    amzn_idx = tickers.index("AMZN")
    ba_idx = tickers.index("BA")
    vlo_idx = tickers.index("VLO")

    assert S[aapl_idx, nvda_idx] == 1.0
    assert S[jnj_idx, tmo_idx] == 1.0
    assert S[tsla_idx, amzn_idx] == 1.0
    # Different sectors
    assert S[aapl_idx, vlo_idx] == 0.0
    assert S[ba_idx, tsla_idx] == 0.0


def test_dynamic_adjacency_acceptance_criteria(graph_data):
    """
    Verify dynamic adjacency matrix properties: symmetry, binary values, self-loops, and sector edges.
    """
    date_to_adj, S = graph_data
    tickers = config.TICKER_LIST
    aapl_idx = tickers.index("AAPL")
    nvda_idx = tickers.index("NVDA")
    jnj_idx = tickers.index("JNJ")
    tmo_idx = tickers.index("TMO")
    tsla_idx = tickers.index("TSLA")
    amzn_idx = tickers.index("AMZN")

    for d, A_t in date_to_adj.items():
        # 1. Symmetric
        assert np.allclose(A_t, A_t.T), f"Adjacency matrix on {d} is not symmetric"
        # 2. Values in {0, 1}
        unique_vals = set(np.unique(A_t))
        assert unique_vals.issubset({0.0, 1.0}), f"Non-binary values in A_t on {d}: {unique_vals}"
        # 3. Diagonal = 1 (self-loops)
        assert np.all(np.diag(A_t) == 1.0), f"Diagonal does not have self-loops on {d}"
        # 4. Same-sector invariant
        assert A_t[aapl_idx, nvda_idx] == 1.0
        assert A_t[jnj_idx, tmo_idx] == 1.0
        assert A_t[tsla_idx, amzn_idx] == 1.0


def test_dense_to_edge_index():
    """Test conversion of binary adjacency matrix to PyG edge_index format."""
    adj = np.eye(9, dtype=np.float32)
    adj[0, 1] = 1.0
    adj[1, 0] = 1.0

    edge_index, edge_weight = dense_adj_to_edge_index(adj)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == 11  # 9 self-loops + 2 edges
    assert edge_weight.shape[0] == 11
