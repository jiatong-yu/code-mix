import os
import json
import yaml
import argparse
from tqdm import tqdm
from utils import extract_code



def best_of_n(config):
    data = json.load(open(config["data_path"]))
    dataset = []
    count = 0
    for qid in data:
        system_input = data[qid]["system"]
        user_input = data[qid]["user"]
        champ_score = -1
        champ_solution = None
        scored_solutions = data[qid]
        for sol in scored_solutions:
            score = sol["score"]
            if score > champ_score:
                champ_score = score
                champ_solution = sol["solution"][config["column"]]
                if score == 1:
                    break
        
        # create dataset entry
        if config["dataset_template"] == "instruct":
            dataset.append(
                {
                    "prompt": user_input,
                    "response": champ_solution
                }
            )
        else:
            convo = [
                {
                    "role": "system",
                    "content": system_input
                },
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": champ_solution
                }
            ]
            dataset.append({
                "conversations": convo
            })
        
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training dataset from cache.")
    parser.add_argument("--config", "-c", type=str, default="configs/dataset_config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    