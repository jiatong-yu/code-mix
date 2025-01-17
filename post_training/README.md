# Training with Code Skill-Mix Dataset
This folder contains code to finetune models using synthetic skill-mix datasets using torchtune. Supported training methods:

| Method | Description | Status |
|--------|-------------|---------|
| SFT | Standard finetuning on gold solution | ✅ Supported |
| Best of N | Finetuning on best of N policy generations | ✅ Supported |
|  + CoT || 🚧 In Progress |
| DPO | Preferred output passes more unit tests  | ✅ Supported |
|  + CoT || 🚧 In Progress |
| GRPO | Each group is N scored generation per question  | 📝 Planned|

## Training Data
**1. Gather policy generations and rewards**

To train a LLM with any of the above training method (except for SFT on gold solutions), you need to sample N responses from the LLM for each question in the Code Skill-Mix dataset. We support either direct sampling or Chain-of-Thoughts sampling (see prompts [here](https://github.com/jiatong-yu/code-mix/blob/main/post_training/prompts/generate_solutions/)). To do so, replace the following command with your setup. 
```bash
python generate_solutions.py \
    --model_path ".cache/meta-llama/Meta-Llama-3-8B-Instruct" \
    --questions_path "../data_pipeline/data/questions.json" \
    --model_name "Llama3.1-8B-Instruct" \
    --task_name "my_algo_dataset" \
    -n 100 \
    -t 0.8 \
    --CoT
```
A key feature of our synthetic data pipeline is reliable reward assignment using unit tests. To assign rewards to model generations, you need to first run the model generations against unit tests created in the data generation step. If you chose to mount to docker container, unit tests should be found in `../data_pipeline/data/`. Run the following command to score model generations:
```bash
cd ../data_pipeline
python score_generations.py --generations_path "MODEL GENERATIONS PATH"
```

**2. Prepare Train Dataset**

We use [torchtune](https://github.com/pytorch/torchtune) as the default training framework. Therefore you need to convert your custom dataset to either [Instruct Dataset](https://pytorch.org/torchtune/0.3/basics/instruct_datasets.html) or [Chat Dataset](https://pytorch.org/torchtune/0.3/basics/chat_datasets.html). 


