"""Main function"""

import argparse
import sys
import warnings

import config
from experiment import experiment
from relation import populate
import utils


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_experiments', default=False, action='store_true')
    parser.add_argument('--run_analysis', default=False, action='store_true')
    parser.add_argument('--cpu', default=1, type=int)
    parser.add_argument('--error_type', default=None)
    parser.add_argument('--seeds', default=None, type=int, nargs='+')
    parser.add_argument('--log', default=False, action='store_true')
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--nosave', default=False, action='store_true')
    parser.add_argument('--alpha', default=0.05, type=float)

    return parser.parse_args()


def main():
    # Check Python version
    # Pinned dependency versions verified only with 3.6 or 3.7
    assert sys.version_info[0] == 3 and sys.version_info[1] in [6, 7], (
        f"Python 3.6 or 3.7 required instead of {sys.version.split(' ')[0]}")

    # Suppress annoying warnings that are due to pinning old versions of libraries
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        args = _parse_args()

        # run experiments on datasets
        if args.run_experiments:
            print("=== Running experiments ===")
            datasets = [utils.get_dataset(args.dataset)] if args.dataset is not None else config.datasets
            experiment(datasets, args.log, args.cpu, args.nosave, args.error_type, args.seeds)

        # run analysis on results
        if args.run_analysis:
            print("=== Running analysis ===")
            populate([args.alpha])

    print("Done")


if __name__ == "__main__":
    main()
