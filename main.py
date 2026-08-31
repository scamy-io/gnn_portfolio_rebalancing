"""
Unified CLI Entrypoint for LSTM-GAT Portfolio Rebalancing System.

Usage:
  python main.py test                                         # Run full GPU unit test suite (33 tests)
  python main.py train                                        # Train Model v4 on GPU
  python main.py evaluate                                     # Run out-of-sample backtest & plot curves
  python main.py risk [--sector SECTOR] [--magnitude MAG]     # Run live risk radar & shock simulation
  python main.py tune [--trials N]                            # Run Optuna hyperparameter study
  python main.py pipeline                                     # Execute full end-to-end workflow (~15s)
"""

import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_command_stream(cmd_args):
    """Executes a command and streams output to stdout."""
    res = subprocess.run([sys.executable] + cmd_args, cwd=str(PROJECT_ROOT))
    return res.returncode


def cmd_test(args):
    """Run full GPU unit test suite."""
    print("\n[MAIN] Running GPU Unit Test Suite (33 Tests)...")
    return run_command_stream(["run_tests_gpu.py"])


def cmd_train(args):
    """Train Model v4."""
    print("\n[MAIN] Running Model v4 Training Pipeline...")
    return run_command_stream(["train.py"])


def cmd_evaluate(args):
    """Run Out-of-Sample Evaluation."""
    weights = args.weights or "data/cache/final_retrained_model.pt"
    print(f"\n[MAIN] Running Out-of-Sample Evaluation with weights: {weights}...")
    return run_command_stream(["evaluate.py", "--weights", weights])


def cmd_risk(args):
    """Run Live Risk Monitoring & Shock Simulation."""
    sector = args.sector or "Information Technology"
    magnitude = str(args.magnitude) if args.magnitude is not None else "-0.15"
    steps = str(args.steps) if args.steps is not None else "5"
    print(f"\n[MAIN] Running Live Risk Radar & {magnitude} Shock on '{sector}'...")
    return run_command_stream([
        "run_risk_monitoring.py",
        "--mode", "shock",
        "--sector", sector,
        "--magnitude", magnitude,
        "--steps", steps
    ])


def cmd_tune(args):
    """Run Optuna Hyperparameter Optimization."""
    trials = str(args.trials) if args.trials else "25"
    print(f"\n[MAIN] Running Tier 3 Optuna Tuning ({trials} trials)...")
    return run_command_stream(["scripts/tune_tier3.py", "--trials", trials])


def cmd_pipeline(args):
    """Execute complete end-to-end system."""
    print("=" * 80)
    print("STARTING COMPLETE END-TO-END QUANTITATIVE PIPELINE")
    print("=" * 80)
    steps = [
        ("GPU Unit Tests", ["run_tests_gpu.py"]),
        ("Model Training", ["train.py"]),
        ("Backtest Evaluation", ["evaluate.py", "--weights", "data/cache/final_retrained_model.pt"]),
        ("Risk & Shock Engine", ["run_risk_monitoring.py", "--mode", "shock", "--sector", "Information Technology"]),
    ]
    for name, cmd in steps:
        print(f"\n>>> Executing Step: {name}...")
        code = run_command_stream(cmd)
        if code != 0:
            print(f"[ERROR] Step '{name}' failed with return code {code}")
            return code
    print("\n" + "=" * 80)
    print("END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="LSTM-GAT Portfolio Optimization & Risk Monitoring CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    p_test = subparsers.add_parser("test", help="Run 33-unit test suite on CUDA GPU")
    p_test.set_defaults(func=cmd_test)

    p_train = subparsers.add_parser("train", help="Train 4-Head GAT Model v4 on CUDA GPU")
    p_train.set_defaults(func=cmd_train)

    p_eval = subparsers.add_parser("evaluate", help="Evaluate out-of-sample backtest & generate plots")
    p_eval.add_argument("--weights", type=str, default="data/cache/final_retrained_model.pt", help="Path to checkpoint weights")
    p_eval.set_defaults(func=cmd_evaluate)

    p_risk = subparsers.add_parser("risk", help="Run live risk radar & recursive shock propagation")
    p_risk.add_argument("--sector", type=str, default="Information Technology", help="Target sector to shock")
    p_risk.add_argument("--magnitude", type=float, default=-0.15, help="Shock magnitude (e.g. -0.15)")
    p_risk.add_argument("--steps", type=int, default=5, help="Number of propagation steps")
    p_risk.set_defaults(func=cmd_risk)

    p_tune = subparsers.add_parser("tune", help="Run Tier 3 Optuna hyperparameter optimization")
    p_tune.add_argument("--trials", type=int, default=25, help="Number of Optuna trials")
    p_tune.set_defaults(func=cmd_tune)

    p_pipe = subparsers.add_parser("pipeline", help="Run entire end-to-end workflow in sequence")
    p_pipe.set_defaults(func=cmd_pipeline)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
