import os
import json
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from utils import extract_code

def main(args):
    
    # Create output path with proper directory structure
    output_dir = os.path.join("training_data", "cache", args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.task_name}.json")
    
    if os.path.exists(args.output_path):
        print(f"Resuming from existing progress at {output_path}")
        with open(output_path, "r") as f:
            res = json.load(f)
    else:
        res = {}

    with open(args.questions_path, "r") as f:
        questions = json.load(f)

    llm = LLM(model=args.model_path)
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        max_tokens=2048
    )

    # Process each question
    for qid in tqdm(questions):
        if qid in res:
            continue
        if not questions[qid]["verification_status"]:
            continue

        # Prepare conversation
        description = questions[qid]["problem_statement"]
        function_name = questions[qid]["function_name"]
        function_signature = questions[qid]["function_signature"]
        function_docstring = questions[qid]["function_docstring"]
        language = questions[qid]["language"]

        if args.CoT:
            with open("prompts/generate_solutions/CoT_user_prompt", "r") as f:
                user_prompt = f.read()
        else:
            with open("prompts/generate_solutions/user_prompt", "r") as f:
                user_prompt = f.read()


        user_input = user_prompt.format(
            description=description,
            function_name=function_name,
            function_signature=function_signature,
            function_docstring=function_docstring,
            language=language
        )

        convo = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]
        convos = [convo] * args.num_samples_per_question

        # Generate outputs
        outputs = llm.chat(
            messages=convos,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        outputs = [x.outputs[0].text for x in outputs]

        # Process results
        qid_outputs = []
        for output in outputs:
            qid_outputs.append({
                "raw_output": output,
                "code": extract_code(output)
            })
        res[qid] = qid_outputs

        # Save intermediate results
        with open(output_path, "w") as f:
            json.dump(res, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-based question generation and processing.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--questions_path", 
        type=str, 
        required=True, 
        help="Path to the JSON file containing coding questions."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Llama3.1-8B-Instruct", 
        help="Name of the model being used (for logging purposes)."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Name of task used for naming output file."
    )
    parser.add_argument(
        "--num_samples_per_question",
        "-n", 
        type=int, 
        default=50, 
        help="Number of samples to generate per question."
    )
    parser.add_argument(
        "--temperature", 
        "-t",
        type=float, 
        default=0.8, 
        help="Sampling temperature for the LLM."
    )
    parser.add_argument(
        "--CoT",
        action="store_true",
        default=False,
        help="Whether to prompts LLMs to generate reasoning steps"
    )
    
    args = parser.parse_args()
    main(args)