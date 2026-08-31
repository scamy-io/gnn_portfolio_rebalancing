# LSTM-GAT Dynamic Graph Portfolio Optimization

PyTorch implementation of the spatio-temporal deep learning portfolio allocation framework:
> *From Headlines to Holdings: Deep Learning for Smarter Portfolio Decisions*  
> (Lin, Lou, Zhang, arXiv:2509.24144v2)

This repository provides an extended multi-asset implementation across a 28-stock universe spanning all 11 GICS sectors. The system integrates temporal LSTM sequence encoding, multi-head Graph Attention Networks (GAT), dynamic correlation and sector adjacency rewiring, decile active portfolio tilting, downside risk minimization via asymmetric Sortino loss, and an attention-based network risk and shock propagation engine.

---

## 1. Out-of-Sample Performance Summary

Evaluated on 358 out-of-sample test trading dates (`2022-07-15` to `2023-12-14`):

| Metric | LSTM-GAT (Tuned) | Baseline Model | Equal-Weight Benchmark (1/N) | Reference Model (Paper Table 2) |
|---|---|---|---|---|
| Cumulative Return (%) | **85.98%** | 36.59% | 34.38% | — |
| Annualized Return (CAGR) | **54.77%** | 24.54% | 23.12% | 28.10% |
| Annualized Sharpe Ratio | **2.24** | 1.15 | 1.06 | 1.06 |
| Annualized Volatility (%) | **22.45%** | 17.45% | 17.52% | 26.60% |
| Maximum Drawdown (%) | **-18.32%** | -16.74% | -16.72% | -21.70% |
| Value at Risk (VaR 95%) (%) | **-2.02%** | -1.58% | -1.58% | -2.68% |
| Excess Return over 1/N (%) | **+51.60%** | +2.21% | 0.00% | — |

---

## 2. Methodology & Architecture

### System Architecture Pipeline

```
Raw Input Data: Daily OHLCV + FNSPID News Sentiment + FRED TB3MS Yield
  │
  ▼
Feature Engineering (10 Features per Asset, Normalized via Train Scalers)
  │
  ▼
Temporal Sequence Encoding: LSTM (Hidden Dimension = 64, Lookback Window = 30 Days)
  │
  ▼
Spatial Relational Encoding: 4-Head Dense Graph Attention (GAT)
  ├── Head 1: Price Momentum & Co-movement
  ├── Head 2: Financial News Sentiment Contagion
  ├── Head 3: Return Volatility Spillover
  └── Head 4: Fundamental GICS Sector Linkages
  │
  ▼
Dynamic Adjacency Graph: A_t (3-Day Refresh, Pearson Correlation rho >= 0.60 + Sector Hierarchy)
  │
  ▼
Decile Active Allocation Head:
  ├── Top K=7 Long Convictions (Overweight)
  ├── Bottom K=7 Short Convictions (Underweight)
  └── Middle 14 Assets Passive (1/N = 3.57%)
  │
  ▼
Zero-Sum Tilt Normalization: sum(w_i) = 1.000000
  │
  ▼
Optimization Objective: Asymmetric Negative Sortino Loss vs. Risk-Free Benchmark
```

### Mathematical Formulations

#### 1. Multi-Head Spatial Attention
Given asset node representations $h_i, h_j \in \mathbb{R}^D$ from the LSTM temporal encoder, head-specific attention coefficients are computed as:

$$e_{ij}^{(k)} = \text{LeakyReLU}\left( \mathbf{a}_{\text{src}}^{(k)\top} W^{(k)} h_i + \mathbf{a}_{\text{dst}}^{(k)\top} W^{(k)} h_j \right)$$

$$\alpha_{ij}^{(k)} = \frac{\exp(e_{ij}^{(k)})}{\sum_{l \in \mathcal{N}_i} \exp(e_{il}^{(k)})}$$

The multi-head aggregated node representation is:

$$h_i' = \sigma\left( \frac{1}{K} \sum_{k=1}^K \sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(k)} W^{(k)} h_j \right)$$

#### 2. Decile Active Portfolio Allocation
Active portfolio weight adjustments are computed via a high-conviction decile allocation head:

$$\delta_i = \text{scale} \cdot \tanh(\text{Linear}(h_i'))$$

$$w_i = \frac{1}{N} + \tilde{\delta}_i, \quad \text{where } \sum_{i=1}^N \tilde{\delta}_i = 0 \implies \sum_{i=1}^N w_i \equiv 1.0$$

#### 3. Asymmetric Negative Sortino Loss
To penalize downside semivariance while preserving upside rallies:

$$\mathcal{L}_{\text{Sortino}}(\mathbf{w}) = -\frac{\mathbb{E}[R_p - R_f]}{\sqrt{\mathbb{E}\left[\min(0, R_p - R_f)^2\right] + \epsilon}}$$

---

## 3. Project Structure

```
gnn_portfolio_rebalancing/
├── data/                               # Ingestion, processing, feature engineering, and caching
│   ├── raw/                            # Local FNSPID news sentiment source
│   ├── cache/                          # Saved model checkpoints and evaluation outputs
│   ├── ingestion.py                    # Multi-source ingestion (prices, benchmarks, news)
│   ├── cleaning.py                     # Calendar alignment and merged panel generation
│   ├── features.py                     # 10 heterogeneous input features
│   └── graph.py                        # Dynamic adjacency graph constructor
├── models/                             # Neural network models
│   ├── __init__.py
│   └── lstm_gat.py                     # LSTM-GAT model with multi-head attention and decile head
├── risk/                               # Live risk monitoring and shock simulation
│   ├── __init__.py
│   ├── concentration.py                # Eigenvector centrality and community modularity entropy
│   └── shock_engine.py                 # Recursive non-linear shock contagion simulator
├── scripts/                            # Auxiliary utilities and diagnostics
│   ├── tune_tier3.py                   # Optuna Bayesian hyperparameter search
│   ├── tune.py                         # Baseline hyperparameter tuning
│   └── diagnostics.py                  # Embedding quality and representation drift analysis
├── tests/                              # Unit test suite (33 tests)
│   ├── test_splits.py                  # Date boundary and zero-overlap tests
│   ├── test_scaffold.py                # Universe constants and determinism
│   ├── test_cleaning.py                # Calendar alignment tests
│   ├── test_features.py                # Feature calculations and scaler fitting
│   ├── test_graph.py                   # Adjacency symmetry and invariants
│   ├── test_model.py                   # Architecture forward pass and weight constraints
│   ├── test_train.py                   # Sharpe and Sortino loss functions
│   ├── test_evaluation.py              # Metric formulas and equal-weight benchmark
│   ├── test_sanity_checklist.py        # No-lookahead verification and determinism
│   └── test_risk_modules.py            # Graph centrality and shock propagation tests
├── plots/                              # Generated visual charts
│   ├── cumulative_return_plot.png
│   └── predicted_weights_plot.png
├── main.py                             # Unified CLI orchestrator
├── train.py                            # Standalone model training pipeline
├── evaluate.py                         # Standalone out-of-sample evaluation runner
├── run_risk_monitoring.py              # Standalone live risk monitoring CLI
├── run_tests_gpu.py                    # GPU-accelerated unit test runner
├── config.py                           # Global constants, universe, and hyperparameter configuration
├── benchmarks.py                       # Benchmark return calculations
├── requirements.txt                    # Python environment dependencies
└── README.md                           # Project documentation
```

---

## 4. Asset Universe Across 11 GICS Sectors

The universe comprises 28 liquid US equities across all 11 Global Industry Classification Standard (GICS) sectors:

| Sector | Tickers |
|---|---|
| Information Technology | AAPL, NVDA, MSFT |
| Health Care | JNJ, TMO, UNH |
| Financials | JPM, BAC, GS |
| Consumer Discretionary | AMZN, TSLA, HD |
| Communication Services | GOOGL, NFLX |
| Industrials | BA, CAT, GE |
| Consumer Staples | COST, PG, KO |
| Energy | VLO, XOM, CVX |
| Materials | APD |
| Utilities | NEE, DUK |
| Real Estate | PLD, AMT |

---

## 5. Quickstart & Usage

### 1. Installation

```bash
git clone https://github.com/scamy-io/gnn_portfolio_rebalancing.git
cd gnn_portfolio_rebalancing
pip install -r requirements.txt
```

### 2. Execution via Unified CLI (`main.py`)

```bash
# Run unit test suite on CUDA GPU (33 tests)
python main.py test

# Train 4-Head GAT Model v4 on GPU
python main.py train

# Evaluate out-of-sample test split and export plots
python main.py evaluate

# Run live network risk radar & -15% sector shock simulation
python main.py risk --sector "Information Technology" --magnitude -0.15

# Run Optuna Bayesian hyperparameter optimization
python main.py tune --trials 25

# Execute complete end-to-end quantitative pipeline
python main.py pipeline
```

### 3. Standalone Execution

```bash
# Direct GPU test execution
python run_tests_gpu.py

# Direct model training
python train.py

# Direct out-of-sample backtest evaluation
python evaluate.py --weights data/cache/final_retrained_model.pt

# Direct risk monitoring and stress testing
python run_risk_monitoring.py --mode shock --sector "Information Technology" --magnitude -0.15 --steps 5
```

## 6. Verification and Reproducibility

* **Deterministic Random State**: Seed `42` set across Python, NumPy, PyTorch, and CUDA.
* **No-Lookahead Guarantee**: Feature standard scalers are fit strictly on the training partition (`2019-01-15` to `2022-01-14`).
* **Unit Test Coverage**: 33 tests verifying mathematical formulas, absence of data leakage, gradient updates, and graph invariants.
