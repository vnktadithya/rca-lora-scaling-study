from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Model_Training.data_formatting import format_instruction


prompts = []
ground_truths = []
with open('./Data/eval_dataset_full.jsonl', 'r') as f:
    for line in f:
        example = json.loads(line.strip())
        formatted_data = format_instruction(example)
        data_split = formatted_data.split('[/INST]')
        test_input = data_split[0] + '[/INST]'
        actual_output = data_split[1]
        prompts.append(test_input)
        ground_truths.append(actual_output.replace('</s>','').strip())


MODEL_ID = "mistralai/Mistral-7B-v0.1"

def write_model_outputs_to_file(file: str, model_output):
    with open(file, 'w') as f:
        for index, result in enumerate(model_output):
            final_output = {
                'checkpoint': 'Base',
                'example_index': index+1,
                'input': prompts[index],
                'ground_truth': ground_truths[index],
                'generated_output': result.outputs[0].text
            }
            f.write(json.dumps(final_output) + "\n")

# load the base model once with multi-LoRA serving enabled
llm = LLM(model=MODEL_ID, quantization="bitsandbytes", dtype="bfloat16", enable_lora=True, max_lora_rank=8, max_model_len=2048, gpu_memory_utilization=0.82, enforce_eager=True)

#base model
base_model_output = llm.generate(prompts, SamplingParams(temperature=0.1, max_tokens = 1024))
write_model_outputs_to_file('./logs/base_model_outputs.jsonl', base_model_output)
    

#ft_20 model
ft20_adapter = LoRARequest(lora_name="ft_20", lora_int_id=1, lora_path='./Checkpoints/rca_mistral_adapter_ft20')    # create the adapter layer for ft_20
ft20_model_output = llm.generate(prompts, SamplingParams(temperature=0.1, max_tokens = 1024), lora_request=ft20_adapter)
write_model_outputs_to_file('./logs/ft20_model_outputs.jsonl', ft20_model_output)


#ft_500 model
ft500_adapter = LoRARequest(lora_name="ft_500", lora_int_id=1, lora_path='./Checkpoints/rca_mistral_adapter_ft500') # create the adapter layer for ft_500
ft500_model_output = llm.generate(prompts, SamplingParams(temperature=0.1, max_tokens = 1024), lora_request=ft500_adapter)
write_model_outputs_to_file('./logs/ft500_model_outputs.jsonl', ft500_model_output)
