import torch
import json
import os
import pdb
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel



# user model 입력
print('input user id: ')
user_id = input()

if user_id == 'A':
    model_user = 'User_A_test_dataset'
elif user_id == 'B':
    model_user = 'User_B_test_dataset'
elif user_id == 'C':
    model_user = 'User_C_test_dataset'
else:
    print("유효하지 않은 사용자 ID. A, B, C 중 하나를 입력")
    exit() 

model_name = "kakaocorp/kanana-nano-2.1b-instruct"
# adapter_path = f"./results/{model_user}"  # LoRA 어댑터 저장 경로
data_file_path = f"./user_datasets/{model_user}.json" 



print(f"데이터 '{model_user}' 로드 중...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")

# model = PeftModel.from_pretrained(model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval() 



class StopAfterOutput(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids, scores, **kwargs):
        # 현재까지 생성된 텍스트를 디코딩
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # "### Output:"이 나온 후, 두 개의 개행("\n\n")이 있으면 멈추기
        if "### Output:" in decoded_text:
            after_output = decoded_text.split("### Output:")[-1] 
            if "\n\n" in after_output:
                return True
            elif '###' in after_output:
                return True
        
        return False



def load_dataset(file_path):
    """지정된 JSON 데이터셋 파일을 로드합니다."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path}를 찾을 수 없습니다.")
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


dataset_to_infer = load_dataset(data_file_path)

if dataset_to_infer is None:
    exit()

print(f"\n{len(dataset_to_infer)}개의 항목에 대해 추론을 시작합니다.")

generated_outputs = []
stopping_criteria = StoppingCriteriaList([StopAfterOutput(tokenizer)])

with torch.no_grad():
    for item in dataset_to_infer:
        instruction = item.get("instruction", "")
        input_data = item.get("input", "")
        
        full_prompt = f"""You need to generate action plans for robotic task planning.

**Rule**
You must generate output construction like example1.
example1
instruction: Place the orange on the bookshelf
input: <task> [relocate]
output: [walk] <kitchen> [find] <orange> [grab] <orange> [find] <bookshelf> [putback] <orange> <studyroom bookshelf>


### Instruction:
{instruction}

### Input:
{input_data}

### Output:
"""
        
        input_ids = tokenizer(
            [full_prompt], 
            return_tensors="pt"
        ).to("cuda")

        output = model.generate(
            input_ids=input_ids["input_ids"],
            max_new_tokens=256,
            do_sample=False,
            stopping_criteria=stopping_criteria
        )

        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        
        
        if "### Output:" in decoded_output:
            after_output_text = decoded_output.split("### Output:")[-1]
            
            unwanted_patterns = ["\nInstruction:", "###", "You"]
            
            cleaned_generated_text = after_output_text
            
            for pattern in unwanted_patterns:
                idx = cleaned_generated_text.find(pattern)
                if idx != -1:
                    cleaned_generated_text = cleaned_generated_text[:idx]
                    
            generated_text = cleaned_generated_text
            
        else:
            generated_text = decoded_output

        final_output = generated_text.rstrip('###').strip()
        
        generated_outputs.append({
            "instruction": instruction,
            "input": input_data,
            "output": final_output
        })

print("\n추론 완료. 생성된 결과:")
print(generated_outputs[:5]) 


output_results_path = f"./vanilla_json/inference_results_{model_user}.json"
with open(output_results_path, 'w', encoding='utf-8') as f:
    json.dump(generated_outputs, f, indent=4, ensure_ascii=False)
print(f"\n결과가 {output_results_path}에 저장되었습니다.")