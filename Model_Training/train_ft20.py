import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
from input_tokenizer import tokenize_function
from data_formatting import format_instruction

bnb_config = BitsAndBytesConfig(load_in_4bit = True,                 #bits and bytes configuration to enable the properties of QLoRA: 4NF, Double Quantization, converting weights back to 16 bit floats for forward pass computations
                                bnb_4bit_quant_type = 'nf4',
                                bnb_4bit_use_double_quant = True,
                                bnb_4bit_compute_dtype = torch.bfloat16
                                )

lora_config = LoraConfig(r = 8,                         # lora configuration with required parameters and target modules
                        lora_alpha = 32,
                        lora_dropout = 0.05,
                        task_type = 'CAUSAL_LM',        # task_type param determines the exact task of the model. 'CAUSAL_LM' refers to next token generation depending on previous ones
                        target_modules = ['q_proj', 'v_proj'],
                        bias = 'none'
                        )


model_id = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = bnb_config)  # instantiates the correct pre-trained architecture for the provided model
model = prepare_model_for_kbit_training(model) # freezes all base model layers, casts the LayerNorm modules to float32 for numerical stability, and enables gradient checkpointing to save VRAM
peft_model = get_peft_model(model, lora_config) # initializing the model with LoRA configuraiton
peft_model.print_trainable_parameters()

#load the dataset and format it for mistral model
dataset = load_dataset('json', data_files = './Data/ft20_train_dataset.jsonl', split = 'train')
dataset = dataset.map(format_instruction)

# tokenizer configuraiton
tokenizer = AutoTokenizer.from_pretrained(model_id)  # convert the input data to the specific data type expected by the provided model
tokenizer.pad_token = tokenizer.eos_token # all the padded tokens will be eos tokens

dataset = dataset.map(lambda batch: tokenize_function(batch, tokenizer), batched = True) # batched = True used Rust under the hood to multi-thread the tokenization

training_args = TrainingArguments(output_dir="./Checkpoints/rca_mistral_adapter_ft20",
                                  per_device_train_batch_size = 1,  # 1 sequence per forward pass
                                  gradient_accumulation_steps = 4,  # accumulate the gradients for 4 steps before updating the weights
                                  optim = 'paged_adamw_32bit',      # prevents memory spikes during optimizer state updates
                                  learning_rate = 2e-4,             
                                  bf16 = True,                      # match the compute data type
                                  max_steps = 100,
                                  )

# data collator is used to pad all the sequences to the maximum length sequence of that specific batch
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False) 

trainer = Trainer(model = peft_model, train_dataset = dataset, args = training_args, data_collator = data_collator)
trainer.train()