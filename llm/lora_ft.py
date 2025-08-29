import torch
import os
import pdb
from datetime import datetime
#from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset


# dataset 
print('user_id 입력: ')
user_id = input()
dataset_user = f'User_{user_id}_dataset'

# 설정값 적용
max_seq_length = 512
dtype = None 
load_in_4bit = False 

model_name = "kakaocorp/kanana-nano-2.1b-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)


# LoRA 설정 및 적용
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 프롬프트 템플릿
full_prompt = """You need to generate action plans for robotic task planning.\n
**Rule**
You must generate output construction like example1.\n
example1
instruction: Place the orange on the bookshelf
input: <task> [relocate]
output: [walk] <kitchen> [find] <orange> [grab] <orange> [find] <bookshelf> [putback] <orange> <studyroom bookshelf>

### Instruction:
{}

### Input:
{}

### Output:
{}"""

EOS_TOKEN = tokenizer.eos_token

# 데이터셋 변환
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = [
        full_prompt.format(instruction, input, output) + EOS_TOKEN
        for instruction, input, output in zip(instructions, inputs, outputs)
    ]
    return {"text": texts}

dataset = Dataset.from_json(f'./user_datasets/{dataset_user}.json')
dataset = dataset.map(formatting_prompts_func, batched=True)

print(dataset)
print(dataset["text"][0])


# 총 학습 step 계산
num_samples = len(dataset)
total_steps = (num_samples // 1) // 2
save_steps = max(total_steps // 5, 1)  # 최소 1 이상

print(f"Total Steps: {total_steps}, Checkpoint Every {save_steps} Steps")

# WandB 설정
wandb_project = "kanana_2.1B_50epoch"
if wandb_project:
    os.environ["WANDB_PROJECT"] = wandb_project

# Trainer 설정
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        num_train_epochs=50,
        save_strategy="epoch", #"steps",
        save_steps=save_steps,
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type='linear',
        output_dir=f'./results/checkpoint/{dataset_user}',
        report_to="wandb",
        run_name=f"kanana_2.1B_50epoch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
    ),
)

# 모델 학습
trainer_stats = trainer.train()

#adapter save
output_dir = f'./results/{dataset_user}'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

