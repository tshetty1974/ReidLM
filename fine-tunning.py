from huggingface_hub import notebook_login
notebook_login() #use token from huggin face account to login 

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
import json
from datasets import Dataset

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load custom JSON dataset
with open('rare_data.json') as f:
    data = json.load(f)

# Ensure data is a list of dictionaries with 'question' and 'answer' keys
if isinstance(data, list):
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]

    dataset_dict = {"question": questions, "answer": answers}
    dataset = Dataset.from_dict(dataset_dict)
else:
    raise ValueError("Data should be a list of dictionaries with 'question' and 'answer' keys.")

def preprocess_function(examples):
    inputs = ["<instruction>" + q + "<response>" for q in examples["question"]]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(examples["answer"], padding="max_length", truncation=True, max_length=512)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

def preprocess_function(examples):
    inputs = ["<instruction>" + q + "<response>" for q in examples["question"]]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(examples["answer"], padding="max_length", truncation=True, max_length=512)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj"]
)

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    optim="paged_adamw_8bit",
    save_steps=1000,
    logging_steps=30,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    args=training_arguments,
    peft_config=peft_config,
    dataset_text_field="question",
)

trainer.train()


