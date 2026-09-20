import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from data_formatting import format_instruction
from input_tokenizer import tokenize_function

MODEL_ID = "mistralai/Mistral-7B-v0.1"
TRAIN_PATH = "./Data/train_dataset_full.jsonl"
EVAL_PATH = "./Data/eval_dataset_full.jsonl"
OUTPUT_DIR = "./Checkpoints/rca_mistral_adapter_ft500"


def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    train_ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
    train_ds = train_ds.map(format_instruction)
    train_ds = train_ds.map(lambda b: tokenize_function(b, tokenizer), batched=True)

    eval_ds = load_dataset("json", data_files=EVAL_PATH, split="train")
    eval_ds = eval_ds.map(format_instruction)
    eval_ds = eval_ds.map(lambda b: tokenize_function(b, tokenizer), batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=30,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="wandb",
        run_name="rca-mistral-ft500",
    )

    trainer = Trainer(
        model=peft_model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"FT-500 training complete. Adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
