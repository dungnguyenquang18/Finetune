import os
import inspect
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


# -------------------
# CONFIG
# -------------------
MODEL_ID = "Qwen/Qwen3-0.6B-Base"   # đổi nếu cần
OUTPUT_DIR = "./qwen3_envi_lora_cot"
TRAIN_FILE = "D:/Finetune/finetune/data/train_cot.json"
VALID_FILE = "D:/Finetune/finetune/data/test_cot.json"
TEST_FILE  = "D:/Finetune/finetune/data/valid_cot.json"

MAX_LEN = 1024        # tăng để chứa chain-of-thought; tùy model có thể lên 2048
BATCH_SIZE = 1        # CoT dài -> giảm batch nếu OOM; dùng gradient accumulation để giữ hiệu quả
GRAD_ACCUM = 8
LR = 1e-4             # thường giảm LR khi huấn luyện CoT, nhưng LoRA có thể chịu LR hơi lớn hơn
EPOCHS = 5
SEED = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)


# -------------------
# Load dataset
# -------------------
data_files = {"train": TRAIN_FILE, "validation": VALID_FILE, "test": TEST_FILE}
raw_datasets = load_dataset("json", data_files=data_files)  # expects fields "en" and "vi"

print("Columns in train:", raw_datasets["train"].column_names)

# -------------------
# Tokenizer
# -------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
    
# -------------------
# Preprocess: create prompt + full text, mask prompt tokens in labels
# -------------------
def preprocess_function(examples):
    # expects 'en' and 'cot'
    sources = examples["en"]
    targets = examples["cot"]

    prompts = [f"Translate English to Vietnamese.\nSource: {s}\nTarget:" for s in sources]
    full_texts = [p + " " + t for p, t in zip(prompts, targets)]

    # tokenize full texts
    tokenized_full = tokenizer(
        full_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_attention_mask=True,
    )

    # tokenize prompts to know prompt length (per sample)
    tokenized_prompts = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_attention_mask=True,
    )

    labels = []
    pad_id = tokenizer.pad_token_id

    for input_ids, prompt_mask in zip(tokenized_full["input_ids"], tokenized_prompts["attention_mask"]):
        lab = input_ids.copy()
        # count non-pad tokens in prompt using attention_mask
        prompt_len = sum(prompt_mask)
        # mask prompt positions so loss only computed on target (keep CoT in target)
        for i in range(min(prompt_len, len(lab))):
            lab[i] = -100
        # ensure padding tokens are -100
        lab = [(-100 if id == pad_id else id) for id in lab]
        labels.append(lab)

    tokenized_full["labels"] = labels
    return tokenized_full

# batched map
tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

# shuffle train
tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(seed=SEED)

# -------------------
# Data collator: pad inputs and replace pad token in labels -> -100
# -------------------
base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def data_collator(features):
    batch = base_collator(features)
    if "labels" in batch:
        # batch["labels"] is tensor
        batch["labels"] = torch.where(batch["labels"] == tokenizer.pad_token_id, -100, batch["labels"])
    return batch


# -------------------
# Load model and apply LoRA
# -------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,  # use biến trên
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    logging_steps=10,

    eval_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to="none",

    predict_with_generate=True,  # helpful for CoT evaluation (compute BLEU/EM on generated sequences)
)

# -------------------
# Trainer
# -------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# -------------------
# Train
# -------------------
trainer.train()

# -------------------
# Save adapter + tokenizer
# -------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
