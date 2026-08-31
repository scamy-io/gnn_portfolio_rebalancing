"""
Risk analytics and network topology metrics for portfolio graph states.
"""

import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


def build_weighted_graph(alpha_t, A_t):
    """
    Build an undirected, attention-weighted graph from A_t and alpha_t.
    """
    G = nx.Graph()
    n = alpha_t.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if A_t[i, j] > 0.5:
                G.add_edge(i, j, weight=float(alpha_t[i, j]))
    return G


def eigenvector_centrality(graph):
    """
    Computes eigenvector centrality of the portfolio graph.
    """
    try:
        cent = nx.eigenvector_centrality_numpy(graph, weight="weight")
    except Exception:
        cent = {n: 1.0 / graph.number_of_nodes() for n in graph.nodes()}
    return cent


def community_clusters(graph):
    """
    Computes modularity-based community clusters.
    """
    communities = list(greedy_modularity_communities(graph, weight="weight"))
    return communities


def hidden_concentration(weights, graph, alpha_t, A_t):
    """
    Computes centrality-weighted concentration risk, community entropy, and
    hidden 2-hop exposure.
    """
    cent = eigenvector_centrality(graph)
    cent_risk = sum(abs(weights[i]) * cent[i] for i in range(len(weights)))

    communities = community_clusters(graph)
    comm_weights = []
    for comm in communities:
        cw = sum(weights[i] for i in comm)
        comm_weights.append(cw)

    comm_weights = np.array(comm_weights)
    comm_weights = comm_weights[comm_weights > 0]
    if comm_weights.sum() > 0:
        p_c = comm_weights / comm_weights.sum()
        comm_entropy = -np.sum(p_c * np.log(p_c + 1e-9))
    else:
        comm_entropy = 0.0

    hidden_2hop_exposure = np.zeros(len(weights))
    for i in range(len(weights)):
        neighbors_1hop = set(graph.neighbors(i))
        neighbors_2hop = set()
        for n1 in neighbors_1hop:
            neighbors_2hop.update(graph.neighbors(n1))
        neighbors_2hop = neighbors_2hop - neighbors_1hop - {i}
        hidden_2hop_exposure[i] = sum(abs(weights[n]) for n in neighbors_2hop)

    return {
        "centrality_risk": float(cent_risk),
        "community_entropy": float(comm_entropy),
        "hidden_2hop_exposure": hidden_2hop_exposure,
        "communities": communities,
    }


def rebalance_suggestion(weights, alpha_t, communities, tickers):
    """
    Evaluates institutional concentration limits and community cross-attention alerts.
    """
    suggestions = []

    for i, w in enumerate(weights):
        if abs(w) > 0.15:
            reduction = abs(w) - 0.10
            suggestions.append({
                "type": "oversized_position",
                "tickers": [tickers[i]],
                "action": (
                    f"Reduce {tickers[i]} by {reduction * 100:.1f}% — "
                    f"position exceeds 15% concentration limit"
                ),
                "attention": None,
            })

    for comm in communities:
        comm = list(comm)
        for a in range(len(comm)):
            for b in range(a + 1, len(comm)):
                u, v = comm[a], comm[b]
                if abs(weights[u]) < 0.10 or abs(weights[v]) < 0.10:
                    continue

                row_u = alpha_t[u]
                nonzero_u = row_u[row_u > 0]
                thresh_u = np.percentile(nonzero_u, 90) if nonzero_u.size else 1.0

                row_v = alpha_t[v]
                nonzero_v = row_v[row_v > 0]
                thresh_v = np.percentile(nonzero_v, 90) if nonzero_v.size else 1.0

                attn_uv = alpha_t[u, v]
                attn_vu = alpha_t[v, u]
                triggered = (
                    attn_uv >= thresh_u or attn_uv >= 0.05
                    or attn_vu >= thresh_v or attn_vu >= 0.05
                )
                if not triggered:
                    continue

                max_attn = max(attn_uv, attn_vu)
                if abs(weights[u]) <= abs(weights[v]):
                    smaller, larger = u, v
                else:
                    smaller, larger = v, u
                reduction_amt = min(abs(weights[u]), abs(weights[v])) * 0.3

                suggestions.append({
                    "type": "community_concentration",
                    "tickers": [tickers[u], tickers[v]],
                    "action": (
                        f"Reduce {tickers[smaller]} by {reduction_amt * 100:.1f}% — "
                        f"high cross-attention ({max_attn:.3f}) with "
                        f"{tickers[larger]} in same community"
                    ),
                    "attention": float(max_attn),
                })

    return suggestions


def compute_all_metrics(weights, alpha_t, A_t, tickers, sectors):
    """
    Single entry point: builds the graph, runs every metric above, and
    returns one dict ready to hand to shock_engine.py for JSON export or
    to the live-monitoring path.
    """
    graph = build_weighted_graph(alpha_t, A_t)
    cent = eigenvector_centrality(graph)
    hc = hidden_concentration(weights, graph, alpha_t, A_t)
    communities = hc["communities"]
    alerts = rebalance_suggestion(weights, alpha_t, communities, tickers)

    comm_id_by_node = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            comm_id_by_node[node] = idx

    stock_metrics = {}
    for i, t in enumerate(tickers):
        stock_metrics[t] = {
            "weight": float(weights[i]),
            "centrality": float(cent.get(i, 0.0)),
            "community": comm_id_by_node.get(i, -1),
            "hidden_2hop": float(hc["hidden_2hop_exposure"][i]),
        }

    return {
        "centrality_risk": hc["centrality_risk"],
        "community_entropy": hc["community_entropy"],
        "num_communities": len(communities),
        "communities": [[tickers[n] for n in comm] for comm in communities],
        "hidden_2hop_exposure": hc["hidden_2hop_exposure"].tolist(),
        "alerts": alerts,
        "stock_metrics": stock_metrics,
    }
