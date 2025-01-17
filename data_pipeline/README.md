# Synthetic Data Generation Tutorial
### About 
This folder contains code to generate programming questions paired with gold solutions and unit tests in custom programming languages and domains. We use the [skill-mix](https://arxiv.org/abs/2310.17567) data generation framework with default k <= 2 setting. 


### Pipeline Overview
The pipeline use manual seeds called "skills", which contains a list of topics or tasks associated with a CS domain. For instance, if you want to generate data associated with Algorithm & Data Structures, a pair of skills could be `mergesort` and `pattern_matching_algorithm`. The pipeline calls multiple LLMs to collaboratively design a programming question that invovles both tasks. Optionally, the question is passed into a verification agent who produces gold solution and unit tests. 

Examples of generated synthetic data can be found here: [https://huggingface.co/spaces/jiatongy/code-skillmix-viewer](https://huggingface.co/spaces/jiatongy/code-skillmix-viewer)

## Installation
Gold solution and unit tests are created with Anthropic's computer use agent. To be able to obtain verified solutions and unit tests, either export into `ANTHROPIC_API_KEY` or
put your API keys into `config/chatbot.yaml` configuration file. To be able to verify generated questions, you will need to run the pipeline in platforms with **sudo access**. 
- Make sure you have installed [docker](https://docs.docker.com/engine/install/).
- Clone this repo
- Make your own changes into Dockerfile (don't modify the Dockerfile in `/verifier_agent` folder).
Below is an example to enter the debug mode and run files in docker container. 
```bash
docker run -it \
    --name $CONTAINER_NAME \
    -p ${HOST_PORT}:${CONTAINER_PORT} \
    -v "$(pwd)/data:/home/${HOST_NAME}/app/data" \
    -v "$(pwd)/configs:/home/${HOST_NAME}/app/configs" \
    -v "$(pwd)/prompts:/home/${HOST_NAME}/app/prompts" \
    -v "$(pwd)/main.py:/home/${HOST_NAME}/app/main.py" \
    -v "$(pwd)/verifier_agent:/home/${HOST_NAME}/app/verifier_agent" \
    -v "$(pwd)/engine:/home/${HOST_NAME}/app/engine" \
    --entrypoint bash \
    ${IMAGE_NAME}:${IMAGE_TAG}
```
To disable verification feature and skip unit test generation, simply set `verify: false` in your config file.

> [!CAUTION]
> The above command `docker run -it -v` mounts your host folder `./data` inside the container. Therefore your host is exposed to changes made by LLM agents to the container's data folder. We made the above mount to allow easy access to unit tests. But if you are concerned about this unprotected LLM agent exposure, simply don't mount the data folder, and access your data while inside your docker container.

## Usage
### Setup a Custom Domain

To generate synthetic coding data in your custom domain, add to `skills/your_list.txt` and modify config file to change desired programming language and difficulty. Each skill file typically contains lines of the format:
```text
<skill_name> | <description or definition>
```
Then, update `.configs/config.yaml` with the following fields to further refine the generation process:
- `langauge`: Programming language associated with your custom domain
- `extension`: File extension associated with your programming language. For instance, if your PL is Python, you should set `extension: py` 
- `test_framework`: You should recommend a unit test framework used to run test cases. For instance, if want to generate in C++, a popular choice of test framework is GoogleTest. A bit of research on your own is needed.
- `category`: Give a name to your custom domain so that generator LLMs have an easier time understanding the scope of your skills. 
- `time_limit`: Difficulty setting. In minutes, how much time do you want the synthetic questions to take to solve for an average CS college student? Increasing this number would increase difficulty.
- `solution_lines`: Difficulty setting. How long do you want the solutions to be? Increasing this number would increase difficulty, but it's also dependent on your programming langauge.

### Setup generator LLMs
We default to use either OpenAI or Anthropic's models to generate synthetic data. If you want to use a local model, add appropriate inference methdos in `engine/chatbot_engine.py`. Generator LLMs' prompts are critical to synthetic data quality. You can find them and modify them here.

### Concepts: Single-Skill (k=1) vs. Multi-Skill (k=2)
**k=1 (Single-Skill Generation)**

When `k_1.generate: true`, the pipeline focuses on one skill at a time.
- N: The desired number of unique skills (or lines) to sample from skill_path.
  - If N is less than the total lines in the file, it randomly samples N.
  - If N is greater, the pipeline will reuse some skills (duplicating them if needed).
- question_per_skill: The number of questions to generate for each skill.

Example scenario:
1. You have 20 total skills in skills/core.txt.
2.	You set N=10 → only 10 distinct skills will be chosen.
3.	If question_per_skill=2, each of those 10 skills generates 2 unique problems.

**k=2 (Multi-Skill Combination)**

When `k_2.generate: true`, the pipeline further creates synthetic questions that use mixture of skills.
-	N: This works slightly differently for multi-skill generation.
	1.	The pipeline identifies all pairs of the available skills.
	2.	If `dense=true`, it will keep all possible pairs of skills (ignoring N).
	3.	If `dense=false`, it will randomly sample N pairs from the possible combinations.

-	use_prev_skills:
    -	If true, it uses the unique skills from the k=1 generation instead of reading a separate file.
    -	If false, it will read a separate skill list (if skill_list_path is provided) or fall back to the same skill file as k=1.
- question_per_skill: The number of problems to generate per skill-pair.

Example scenario:
1.	You have 5 total skills → possible pairs can be 10 in a full combination (5 choose 2).
2.	If dense=false and N=5, the pipeline randomly selects 5 pairs from those 10.
3.	If question_per_skill=2, it generates 2 problems for each selected pair.


