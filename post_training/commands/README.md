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

## Generate training data
To train a LLM with any of the above training method (except for SFT), you need to sample N solutions from the LLM for each question in the Code Skill-Mix dataset. To do so, replace the following command with your setup.
```bash
python generate_solutions.py \
    --model_path ".cache/meta-llama/Meta-Llama-3-8B-Instruct" \
    --questions_path "../data_pipeline/data/questions.json" \
    --model_name "Llama3.1-8B-Instruct" \
    --task_name "algorithm" \
    -n 100 \
    -t 0.8 \
    --CoT
```

