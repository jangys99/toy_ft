import torch
import pdb
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel


# user model 입력
print('input user id: ')
user_id = input()

if user_id == 'A':
    model_user = 'User_A_train_dataset'
elif user_id == 'B':
    model_user = 'User_B_train_dataset'
elif user_id == 'C':
    model_user = 'User_C_train_dataset'
else:
    print("유효하지 않은 사용자 ID. A, B, C 중 하나를 입력")
    model_user = None


class StopAfterOutput(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids, scores, **kwargs):
        # 현재까지 생성된 텍스트를 디코딩
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # "### Output:"이 나온 후, 두 개의 개행("\n\n")이 있으면 멈추기
        if "### Output:" in decoded_text:
            after_output = decoded_text.split("### Output:")[-1] # "### Output:" 이후 텍스트만 확인
            if "\n\n" in after_output:  # 개행 2번 확인
                return True  # 멈추기
            elif '###' in after_output:
                return True
        
        return False  # 계속 생성
    
model_name = "kakaocorp/kanana-nano-2.1b-instruct"
adapter_path = f"./results/{model_user}"  # LoRA 어댑터 저장 경로

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")

model = PeftModel.from_pretrained(model, adapter_path)

tokenizer = AutoTokenizer.from_pretrained(model_name)


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


input_ids = tokenizer(
    [
        full_prompt.format(
            "Place the knife on the table", # instruction
            "<task> [relocate]", # input
            "", # output
        )
    ], return_tensors="pt").to("cuda")

stopping_criteria = StoppingCriteriaList([StopAfterOutput(tokenizer)])

_ = model.eval()
with torch.no_grad():
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

print(final_output)



