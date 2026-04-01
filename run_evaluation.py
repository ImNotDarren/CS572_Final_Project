#!/usr/bin/env python3
"""Main entry point for running MIA24 vs DietAI24 evaluation.

Usage:
    # Run both methods on all 100 samples
    python run_evaluation.py

    # Run only DietAI24 baseline
    python run_evaluation.py --methods DietAI24

    # Run only MIA24 proposed method
    python run_evaluation.py --methods MIA24

    # Quick test with 3 samples
    python run_evaluation.py --max-samples 3

    # Run both methods on 5 samples
    python run_evaluation.py --max-samples 5 --methods DietAI24 MIA24
"""

import argparse
import logging

from evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MIA24 vs DietAI24 on Nutrition5k"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["DietAI24", "MIA24"],
        choices=["DietAI24", "MIA24"],
        help="Methods to evaluate (default: both)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of evaluation samples (default: all)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_evaluation(methods=args.methods, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
