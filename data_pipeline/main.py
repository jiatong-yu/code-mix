import os
import sys
import yaml
import json
import uuid
import random
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from tqdm import tqdm

from verifier_agent.verify import VerifierAgent
from engine.chatbot_engine import ChatbotEngine
from utils import update_cost, verify_folder_structure, EvaluationParser

DEBUG = True
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_skill_list(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Prepare skill list data based on user configuration.

    Args:
        config (Dict[str, Any]): Parsed YAML config data.

    Returns:
        Tuple[Dict[str, Any], Dict[str, str]]:
            - A dictionary with 'k_1' and/or 'k_2' keys if those skills are requested.
            - A dictionary mapping skill names to their definitions.
    """
    # Load any exclusion config
    if config["exclusion_config"]:
        with open(config["exclusion_config"], "r") as f:
            exclusion = json.load(f)
    else:
        exclusion = None

    res = {"k_1": None, "k_2": None}
    skill_dict: Dict[str, str] = {}  # skill dictionary with definitions

    # Process k=1 case
    if config['k_1']['generate']:
        res['k_1'] = {}
        if not config['skill_path']:
            raise ValueError("Skill path not provided for k = 1")

        with open(config['skill_path'], 'r') as f:
            skills = f.read().split("\n")

        seed_skills: List[str] = []
        for skill_line in skills:
            skill_parts = skill_line.split(" | ")
            if len(skill_parts) < 2:
                continue
            skill_dict[skill_parts[0]] = skill_parts[1]
            seed_skills.append(skill_parts[0])

        N = config['k_1']['N']
        if N == len(seed_skills):
            skill_list = seed_skills
        elif N < len(seed_skills):
            skill_list = random.sample(seed_skills, N)
        else:
            # prioritize all skills, then sample duplicates as needed
            skill_list = seed_skills[:]
            add = random.sample(seed_skills, N - len(seed_skills))
            skill_list.extend(add)

        unique_skills = list(set(skill_list))
        res['k_1']['skill_list'] = skill_list
        res['k_1']['unique_skills'] = unique_skills

    # Process k=2 case
    if config['k_2']['generate']:
        res['k_2'] = {}
        if config['k_2']['skill_list_path']:
            with open(config['k_2']['skill_list_path'], 'r') as f:
                skills = f.read().split("\n")
            seed_skills = []
            for skill_line in skills:
                skill_parts = skill_line.split(" | ")
                if len(skill_parts) < 2:
                    continue
                skill_dict[skill_parts[0]] = skill_parts[1]
                seed_skills.append(skill_parts[0])
        elif config['k_2']['use_prev_skills']:
            # Must rely on k=1 unique skills
            assert res['k_1'], "Need to generate k=1 skills first"
            seed_skills = res['k_1']['unique_skills']
        else:
            # if sample k1, would be same as k1
            with open(config['skill_path'], 'r') as f:
                skills = f.read().split("\n")
            seed_skills = []
            for skill_line in skills:
                skill_parts = skill_line.split(" | ")
                if len(skill_parts) < 2:
                    continue
                skill_dict[skill_parts[0]] = skill_parts[1]
                seed_skills.append(skill_parts[0])

        seed_skills.sort()
        skill_list = []
        # We ignore 'N' for the dense generation
        N = config['k_2']['N']

        for i in range(len(seed_skills)):
            for j in range(i + 1, len(seed_skills)):
                if exclusion:
                    skill_1 = seed_skills[i]
                    skill_2 = seed_skills[j]
                    if skill_1 in exclusion.get(skill_2, []):
                        continue
                skill_list.append(f"{seed_skills[i]}, {seed_skills[j]}")

        if config['k_2']['dense']:
            res['k_2']['skill_list'] = skill_list
            res['k_2']['unique_skills'] = seed_skills
        else:
            skill_list = random.sample(skill_list, min(N, len(skill_list)))
            res['k_2']['skill_list'] = skill_list
            res['k_2']['unique_skills'] = seed_skills

    logger.debug("Skill list prepared: %s", res)
    return res, skill_dict


def parse_output(out: str, engine: ChatbotEngine) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Attempt to parse model output as JSON. If parsing fails, request a parser from the engine.

    Args:
        out (str): Raw model output.
        engine (ChatbotEngine): An instance of ChatbotEngine to re-parse output if needed.

    Returns:
        tuple:
            - A dictionary (parsed JSON) or None if unparseable.
            - The original or "intermediate" string (which might contain JSON) or None.
    """
    origin = out
    if out.startswith("```json"):
        out = out.strip('```').strip('json')
    try:
        parsed = json.loads(out)
        return parsed, origin
    except json.JSONDecodeError:
        logger.info("Extracting JSON with helper LLM; original output:\n%s", origin)
        try:
            with open("./prompts/parse_json.txt", "r") as f:
                retry_prompt = f.read()
        except FileNotFoundError:
            logger.exception("parse_json.txt not found. Cannot re-parse JSON.")
            return None, None

        response = engine.memoryless_chat(retry_prompt, {"text_to_parse": origin}, temperature=0)
        origin = response
        if response.startswith("```json"):
            response = response.strip('```').strip('json')
        try:
            parsed = json.loads(response)
            return parsed, origin
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON after second attempt.")
            return None, None


def _convert_chat(content: str, role: str) -> str:
    """
    Convert text into a pre-defined chat format for the LLM.

    Args:
        content (str): The text content to transform.
        role (str): Either 'assistant' or 'user'.

    Returns:
        str: A text block wrapped in special tokens.
    """
    assert role in ['assistant', 'user'], "Role must be 'assistant' or 'user'"

    prefix = f"<|im_start|>{role}\n"
    postfix = "\n<|im_end|>"
    return prefix + content + postfix


def _generate_question(
    k: int,
    kwargs: Dict[str, Any],
    existing_questions: List[str],
    engine: ChatbotEngine,
    config: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]], Optional[Dict[str, Any]]]:
    """
    Generate an initial question using the 'generate' prompts, then parse the output into JSON.

    Args:
        k (int): The number of skills combined (1 or 2).
        kwargs (dict): Basic data used to fill in the prompt.
        existing_questions (List[str]): Chat history or prior raw responses.
        engine (ChatbotEngine): ChatbotEngine instance to talk to the LLM.
        config (dict): The global config dictionary.

    Returns:
        tuple: (parsed JSON, updated chat history, updated kwargs) or (None, None, None) on error.
    """
    try:
        if k == 1:
            with open("./prompts/k_1/generate.txt", "r") as f:
                generate_prompt = f.read()
            with open("./prompts/k_1/generate_next.txt", "r") as f:
                generate_next_prompt = f.read()
        else:
            with open("./prompts/k_2/generate.txt", "r") as f:
                generate_prompt = f.read()
            with open("./prompts/k_2/generate_next.txt", "r") as f:
                generate_next_prompt = f.read()
    except FileNotFoundError as e:
        logger.exception("Prompt files not found for generating question.")
        return None, None, None

    generate_prompt = generate_prompt.format(**kwargs)
    generate_next_prompt = generate_next_prompt.format(
        **{"skill_list_with_definition": kwargs['skill_list_with_definition']}
    )

    for chat_history in existing_questions:
        # incorporate old chat history and follow-up instructions
        generate_prompt += f"\n{chat_history}\n{generate_next_prompt}"

    response = engine.chat(
        generate_prompt,
        None,
        model=config['model_configs']['generate']['model'],
        temperature=config['model_configs']['generate']['temperature']
    )

    out, text_dict = parse_output(response, engine)
    if out is None:
        logger.error("Failed to generate question (parsing error): %s", response)
        return None, None, None

    logger.info("Generated question.")
    logger.debug("Function name: %s", out.get('function_name'))
    logger.debug("Problem statement: %s", out.get('problem_statement'))

    chat_history = [_convert_chat(text_dict, 'assistant')] if text_dict else []

    # Update kwargs
    kwargs['function_name'] = out['function_name']
    kwargs['function_signature'] = out['function_signature']
    kwargs['function_docstring'] = out['function_docstring']
    kwargs['solution'] = out['solution']
    kwargs['problem_statement'] = out['problem_statement']

    return out, chat_history, kwargs


def _generate_debate(
    k: int,
    kwargs: Dict[str, Any],
    engine: ChatbotEngine,
    config: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Iteratively generate a 'debate' or critique about the question to refine it.

    Args:
        k (int): The skill-level grouping (1 or 2).
        kwargs (dict): Data used to fill in the debate prompt.
        engine (ChatbotEngine): ChatbotEngine instance for inference.
        config (dict): The global config dictionary.

    Returns:
        tuple: (updated question dict or None, chat_history list).
    """
    chat_history: List[str] = []
    try:
        if k == 1:
            with open("./prompts/k_1/debate.txt", "r") as f:
                debate_prompt = f.read()
        else:
            with open("./prompts/k_2/debate.txt", "r") as f:
                debate_prompt = f.read()

        with open("./prompts/feed_critique.txt", "r") as f:
            feed_critique_template = f.read()
    except FileNotFoundError:
        logger.exception("Debate or feed_critique prompt files not found.")
        return None, []

    # Attempt up to 2 debate-critique cycles
    for _ in range(2):
        feedback = engine.inference(
            debate_prompt,
            kwargs,
            model=config['model_configs']['debate']['model'],
            temperature=config['model_configs']['debate']['temperature']
        )

        logger.info("Debate critique:\n%s", feedback)

        # if the LLM says the question is good, finalize
        if "he question meets all requirements, pass" in feedback:
            out = {
                "function_name": kwargs['function_name'],
                "function_signature": kwargs['function_signature'],
                "function_docstring": kwargs['function_docstring'],
                "solution": kwargs['solution'],
                "problem_statement": kwargs['problem_statement'],
            }
            return out, chat_history

        feed_critique = feed_critique_template.format(debate_feedback=feedback)
        chat_history.append(feed_critique)

        response = engine.chat(
            feed_critique,
            None,
            model=config['model_configs']['generate']['model'],
            temperature=config['model_configs']['generate']['temperature']
        )

        out, text_dict = parse_output(response, engine)
        if not out:
            logger.error("Failed to generate updated question (parsing error): %s", response)
            return None, chat_history

        logger.debug("Updated question - Reasoning: %s", out.get('reasoning'))
        logger.debug("Updated question - Function name: %s", out.get('function_name'))
        logger.debug("Updated question - Problem statement: %s", out.get('problem_statement'))

        if text_dict:
            chat_history.append(_convert_chat(text_dict, 'assistant'))

        # Update kwargs
        kwargs['function_name'] = out['function_name']
        kwargs['function_signature'] = out['function_signature']
        kwargs['function_docstring'] = out['function_docstring']
        kwargs['solution'] = out['solution']
        kwargs['problem_statement'] = out['problem_statement']

    return out, chat_history


def generate_question(
    k: int,
    skill_let: str,
    skill_dictionary: Dict[str, str],
    existing_questions: List[str],
    engine: ChatbotEngine,
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Orchestrate question generation + debate pass. Returns a dict representing the question
    or None if generation fails.

    Args:
        k (int): The number of skills in the prompt (1 or 2).
        skill_let (str): The skill or skill pair, e.g. "python" or "python, c++".
        skill_dictionary (Dict[str, str]): A mapping of skill to definition.
        existing_questions (List[str]): Prior chat history or raw responses.
        engine (ChatbotEngine): The conversation engine.
        config (Dict[str, Any]): The global config dictionary.

    Returns:
        Optional[Dict[str, Any]]: A dict with question data or None on failure.
    """
    engine.clear_chat_history()

    if k == 1:
        skill_list_with_definition = (
            " ".join(skill_let.split("_")) + ": " + skill_dictionary[skill_let]
        )
    else:
        # skill_let is something like "python, c++"
        skill_parts = [s.strip() for s in skill_let.split(",")]
        if len(skill_parts) < 2:
            logger.error("Expected two skills, got: %s", skill_let)
            return None
        a, b = skill_parts[0], skill_parts[1]
        skill_list_with_definition = (
            f"{a}: {skill_dictionary[a]}\n" f"{b}: {skill_dictionary[b]}"
        )

    kwargs = {
        "category": config['generation_config']['category'],
        "programming_language": config['generation_config']['language'],
        "skill_list_with_definition": skill_list_with_definition,
        "time_limit_minutes": config['generation_config']['time_limit'],
        "skill_list": " ".join(skill_let.split("_")),
        "solution_lines": config['generation_config']['solution_lines'],
    }

    total_chat_history: List[str] = []

    # Generate initial question
    old_question, chat_history, kwargs = _generate_question(
        k, kwargs, existing_questions, engine, config
    )

    if not old_question:
        engine.clear_chat_history()
        return None
    total_chat_history.extend(chat_history or [])

    # Perform debate pass
    updated_question, debate_history = _generate_debate(k, kwargs, engine, config)
    if not updated_question:
        updated_question = old_question
    else:
        total_chat_history.extend(debate_history)

    out = updated_question
    question_id = str(uuid.uuid4())
    question_data = {
        question_id: {
            "k": k,
            "skills": skill_let,
            "raw_response": "\n".join(total_chat_history),
            "function_name": out['function_name'],
            "function_signature": out['function_signature'],
            "function_docstring": out['function_docstring'],
            "sample_solution": out['solution'],
            "problem_statement": out['problem_statement'],
            "language": config['generation_config']['language'],
            "original_generation": old_question,
        }
    }
    engine.clear_chat_history()
    return question_data


async def generate_test(
    question: Dict[str, Dict[str, Any]],
    agent: VerifierAgent,
    config: Dict[str, Any]
) -> None:
    """
    Generate and validate test cases locally for a given question.

    Args:
        question (dict): Single-question dictionary, keyed by UUID.
        agent (VerifierAgent): The verifier agent instance.
        config (dict): The global config dictionary.
    """
    try:
        with open("./prompts/verifier.txt", "r") as f:
            agent.verifier_prompt = f.read()
    except FileNotFoundError:
        logger.exception("verifier.txt prompt file not found.")
        return

    qid = next(iter(question))
    q = question[qid]
    logger.info("Generating test cases for question: %s", q['function_name'])

    cwd = os.getcwd()
    output_folder = config['output_folder']

    # Convert to relative path if necessary
    if output_folder.startswith("./"):
        output_folder = output_folder[2:]
    elif output_folder.startswith("/"):
        output_folder = output_folder[1:]

    kwargs = {
        "id": qid,
        "base_dir": cwd,
        "programming_language": q['language'],
        "appropriate_extension": config['generation_config']['extension'],
        "test_framework": config['generation_config']['test_framework'],
        "output_folder": output_folder,
        "problem_statement": q['problem_statement'],
        "function_name": q['function_name'],
        "function_signature": q['function_signature'],
        "function_docstring": q['function_docstring'],
        "code_snippet": q['sample_solution'],
    }

    await agent.verify_code_local(kwargs)
    base_path = Path(config['output_folder']) / qid
    success = verify_folder_structure(base_path)

    repeat = config['verifier_attempts'] - 1
    while not success and repeat > 0:
        logger.warning(
            "Failed to generate test cases for question: %s. Retrying...",
            q['function_name']
        )
        await agent.verify_code_local(kwargs)
        success = verify_folder_structure(base_path)
        repeat -= 1


def generate_batch_questions(
    skill_let: str,
    skill_dictionary: Dict[str, str],
    existing_questions: List[str],
    engine: ChatbotEngine,
    verifier: VerifierAgent,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a batch of questions for one or two skills.

    Args:
        skill_let (str): A single skill or a pair of skills joined by comma.
        skill_dictionary (dict): Maps skill names to their definitions.
        existing_questions (List[str]): List of prior chat messages or raw responses.
        engine (ChatbotEngine): Chat engine instance.
        verifier (VerifierAgent): Agent used to verify solutions.
        config (dict): The global config dictionary.

    Returns:
        A dictionary of newly generated question data keyed by unique IDs.
    """
    skill_count = len([s.strip() for s in skill_let.split(",")])
    if skill_count == 1:
        repeat = config['k_1']['question_per_skill']
        verify_flag = config['k_1']['verify']
    else:
        repeat = config['k_2']['question_per_skill']
        verify_flag = config['k_2']['verify']

    questions: Dict[str, Any] = {}
    for _ in range(repeat):
        question = generate_question(
            skill_count, skill_let, skill_dictionary, existing_questions, engine, config
        )
        if not question:
            continue

        qid = next(iter(question))
        if verify_flag:
            asyncio.run(generate_test(question, verifier, config))
            base_path = Path(config['output_folder']) / qid
            success = verify_folder_structure(base_path)
            question[qid]['verification_status'] = bool(success)
        else:
            question[qid]['verification_status'] = False

        # Add to the questions dictionary
        questions.update(question)
        existing_questions.append(question[qid]['raw_response'])

    return questions


def _resume_existing_progress(config: Dict[str, Any]) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    List[str],
    List[str],
    Dict[str, Any],
    Dict[str, Any]
]:
    """
    Resume question generation from an existing progress tracker.

    Args:
        config (dict): The global config dictionary.

    Returns:
        (skill_list, skill_dictionary, joined_skill_list, finished_skill_lets,
         question_config, questions)
    """
    with open(Path(config['output_folder']) / "tracker.json", "r") as f:
        tracker = json.load(f)

    skill_list = tracker['skill_list']
    skill_dictionary = tracker['skill_dictionary']
    joined_skill_list = tracker['joined_skill_list']
    finished_skill_lets = tracker['finished_skill_lets']
    question_config = tracker['question_config']

    with open(Path(config['output_folder']) / "questions.json", "r") as f:
        questions = json.load(f)

    progress = (len(finished_skill_lets) / len(joined_skill_list)) * 100
    logger.info("Resuming from existing progress: %.2f%%", progress)

    return (
        skill_list,
        skill_dictionary,
        joined_skill_list,
        finished_skill_lets,
        question_config,
        questions
    )


def _initialize_new_progress(config: Dict[str, Any]) -> Tuple[
    Dict[str, Any],
    Dict[str, str],
    List[str],
    List[str],
    Dict[str, Any],
    Dict[str, Any]
]:
    """
    Create new progress and skill data structures if tracker.json does not exist.

    Args:
        config (dict): The global config dictionary.

    Returns:
        (skill_list, skill_dictionary, joined_skill_list, finished_skill_lets,
         question_config, questions)
    """
    skill_list, skill_dictionary = prepare_skill_list(config)
    questions: Dict[str, Any] = {}
    question_config: Dict[str, List[str]] = {}
    joined_skill_list: List[str] = []

    if skill_list.get('k_1'):
        joined_skill_list.extend(skill_list['k_1']['skill_list'])
    if skill_list.get('k_2'):
        joined_skill_list.extend(skill_list['k_2']['skill_list'])

    logger.info("Generating questions for skill list: %s", joined_skill_list)
    finished_skill_lets: List[str] = []
    return (
        skill_list,
        skill_dictionary,
        joined_skill_list,
        finished_skill_lets,
        question_config,
        questions
    )


def main(config: Dict[str, Any], engine: ChatbotEngine, verifier: VerifierAgent) -> None:
    """
    Main entry point to generate questions, handle verification, and store progress.

    Args:
        config (dict): Parsed configuration from YAML.
        engine (ChatbotEngine): Chatbot engine for generating questions.
        verifier (VerifierAgent): Verifier agent to test solutions.
    """
    # Log the received config file
    logger.info("Received config:\n%s", json.dumps(config, indent=2))

    output_dir = Path(config['output_folder'])
    if (output_dir / "tracker.json").exists():
        (
            skill_list,
            skill_dictionary,
            joined_skill_list,
            finished_skill_lets,
            question_config,
            questions
        ) = _resume_existing_progress(config)
    else:
        (
            skill_list,
            skill_dictionary,
            joined_skill_list,
            finished_skill_lets,
            question_config,
            questions
        ) = _initialize_new_progress(config)

    # Filter only the skills not yet processed
    resume = [x for x in joined_skill_list if x not in finished_skill_lets]

    for skill_let in tqdm(resume):
        logger.debug("Processing skill_let: %s", skill_let)
        if skill_let not in question_config:
            question_config[skill_let] = []

        existing_question_ids = question_config[skill_let]
        existing_questions = [questions[q]['raw_response'] for q in existing_question_ids]

        batch = generate_batch_questions(
            skill_let, skill_dictionary, existing_questions, engine, verifier, config
        )

        # Update local structures
        questions.update(batch)
        question_config[skill_let].extend(list(batch.keys()))
        finished_skill_lets.append(skill_let)

        # Persist questions to file
        with open(output_dir / "questions.json", "w") as f:
            json.dump(questions, f, indent=4)

        # Update progress tracker
        tracker = {
            "finished_skill_lets": finished_skill_lets,
            "question_config": question_config,
            "joined_skill_list": joined_skill_list,
            "skill_list": skill_list,
            "skill_dictionary": skill_dictionary,
        }
        with open(output_dir / "tracker.json", "w") as f:
            json.dump(tracker, f, indent=4)
    
    # postprocess to remove unverified (if verify=true) data and print out dataset statistics
    remove_unverified = True
    if config['k_1']['generate'] and not config['k_1']['verify']:
        remove_unverified = False
    if config['k_2']['generate'] and not config['k_2']['verify']:
        remove_unverified = False
    parser = EvaluationParser(config=config)
    parser.postprocess(remove_unverified=remove_unverified, compute_generation_statistics=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    # Attempt to load the YAML config
    try:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logger.exception("Config file %s not found. Exiting.", args.config)
        sys.exit(1)
    except yaml.YAMLError:
        logger.exception("Error parsing YAML config file.")
        sys.exit(1)

    # Possibly modify the output folder if it's absolute within the same directory
    output_folder = config['output_folder']
    cwd = os.getcwd()
    if cwd in output_folder:
        output_folder = output_folder.replace(cwd, "")
        logger.warning(
            "Output folder given as absolute path. Changing to relative path...\n"
            "Home directory: %s\nOutput folder: %s",
            cwd, output_folder
        )
    config['output_folder'] = output_folder

    # Setup the engine and verifier
    cwd_path = Path(os.getcwd())
    config_path = cwd_path / "configs/chatbot.yaml"

    engine = ChatbotEngine(config_path=config_path)
    verifier = VerifierAgent()

    try:
        main(config, engine, verifier)
        cost_gen = engine.get_cost()
        logger.info("Cost incurred for generating: %s", cost_gen)

        cost_ver = verifier.get_cost()
        logger.info("Cost incurred for verifying: %s", cost_ver)

        update_cost(cost_gen + cost_ver)
    except KeyboardInterrupt:
        logger.info("Exiting by user request (KeyboardInterrupt).")
        cost_gen = engine.get_cost()
        logger.info("Cost incurred for generating: %s", cost_gen)

        cost_ver = verifier.get_cost()
        logger.info("Cost incurred for verifying: %s", cost_ver)

        update_cost(cost_gen + cost_ver)
    except Exception as e:
        logger.exception("An unexpected exception occurred.")
        cost_gen = engine.get_cost()
        logger.info("Cost incurred for generating: %s", cost_gen)

        cost_ver = verifier.get_cost()
        logger.info("Cost incurred for verifying: %s", cost_ver)

        update_cost(cost_gen + cost_ver)
        raise e