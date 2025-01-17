import os
import json
import argparse
from utils import EvaluationParser


def main(args):
    eval_parser = EvaluationParser(args.config)
    eval_parser.score_all_solutions(args.generations_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score model generations with unit")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--generations_path", type=str, required=True)
    args = parser.parse_args()
    main(args)