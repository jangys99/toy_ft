import json
import torch
import faiss
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer



MODEL_NAME = "kakaocorp/kanana-nano-2.1b-instruct"
OUTPUT_DIR = "user_datasets" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 사용자 ID 입력
print('input user id (A, B, or C): ')
user_id_input = input().upper() 

if user_id_input not in ['A', 'B', 'C']:
    print("유효하지 않은 사용자 ID입니다. A, B, C 중 하나를 입력하세요.")
    exit()

user_file_name = f"User_{user_id_input}_dataset.json"
print(f"\n--- User {user_id_input}의 데이터셋으로 RAG 시스템을 구축합니다. ---")


embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_name).to(DEVICE)


class StopAfterOutput(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids, scores, **kwargs):
        # 현재까지 생성된 텍스트를 디코딩
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # "### Output:"이 나온 후, 두 개의 개행("\n\n")이 있으면 멈추기
        if "### Output:" in decoded_text:
            after_output = decoded_text.split("### Output:")[-1] 
            if "\n\n" in after_output:  # 개행 2번 확인
                return True  # 멈추기
            elif '###' in after_output:
                return True

        
        return False  
    
# -------------------------------------
# 3. 사용자별 데이터 로드 및 임베딩 생성
# -------------------------------------
def load_and_embed_user_data(file_name):
    file_path = os.path.join(OUTPUT_DIR, file_name)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} 파일을 찾을 수 없습니다. (user_datasets 폴더 확인 필요)")
        return None, None

    texts_to_embed = [
        f"Instruction: {item['instruction']} Input: {item['input']}"
        for item in data
    ]
    
    embeddings = embedding_model.encode(texts_to_embed, convert_to_tensor=False)
    
    return data, embeddings

# -------------------------------------
# 4. 벡터 데이터베이스(FAISS) 구축
# -------------------------------------
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    # FAISS 인덱스 생성
    index = faiss.IndexFlatL2(dimension)
    # 임베딩 추가 (float32 타입으로 변환)
    index.add(embeddings.astype('float32'))
    return index


llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(DEVICE)

# -------------------------------------
# 6. RAG 추론 함수
# -------------------------------------
def rag_inference(instruction, user_input, index, user_data, top_k=5):
    # 1. 사용자 쿼리 임베딩
    query_text = f"Instruction: {instruction} Input: {user_input}"
    query_embedding = embedding_model.encode([query_text], convert_to_tensor=False).astype('float32')

    # 2. FAISS를 사용해 관련 데이터 검색 (해당 사용자 DB 내에서만)
    distances, indices = index.search(query_embedding, top_k)

    # 3. 검색된 데이터(컨텍스트) 추출 및 프롬프트 구성
    retrieved_outputs = []
    # 검색된 인덱스에 해당하는 user_data의 'output'을 추출합니다.
    for idx in indices[0]:
        if idx < len(user_data):
            retrieved_outputs.append(user_data[int(idx)]['output'])
    
    context_str = "\n".join(retrieved_outputs)
    
    # RAG 프롬프트 템플릿: 검색된 컨텍스트를 LLM에 전달
    prompt = f"""You need to generate action plans for robotic task planning.
**Rule**
You must generate output construction like example1.

example1
instruction: Place the orange on the bookshelf
input: <task> [relocate]
output: [walk] <kitchen> [find] <orange> [grab] <orange> [find] <bookshelf> [putback] <orange> <studyroom bookshelf>

### Instruction:
{instruction}

### Input:
{user_input}

### Context (Relevant past outputs):
{context_str}

### Output:
"""
    
    # 4. LLM 추론
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
    stopping_criteria = StoppingCriteriaList([StopAfterOutput(llm_tokenizer)])


    with torch.no_grad():
        output = llm_model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=256,
            do_sample=False,
            stopping_criteria=stopping_criteria
        )
    
    decoded_output = llm_tokenizer.decode(output[0], skip_special_tokens=True)
    
    response = decoded_output[len(prompt):].strip()
    lines = response.split('\n')
    response = lines[0].strip()
    
    first_walk_index = response.find('[walk]')
    second_walk_index = response.find('[walk]', first_walk_index + 1)
    
    if second_walk_index != -1:
        response = response[:second_walk_index].strip()
    else:
        response = response

    
    return response, context_str

# 7. 실행
# -------------------------------------
if __name__ == "__main__":
    
    # 1. 선택된 사용자 데이터셋 로드 및 인덱스 구축
    user_data, embeddings = load_and_embed_user_data(user_file_name)
    
    if user_data is None:
        exit()
        
    faiss_index = build_faiss_index(embeddings)
    print(f"User {user_id_input}의 FAISS 인덱스 구축 완료. 데이터 수:", faiss_index.ntotal)
    
    # 2. RAG 추론 예시
    test_instruction = "Place the hat on the table"
    test_input = "<task> [relocate]"
    
    print("\n--- RAG 추론 시작 ---")
    print(f"Instruction: {test_instruction}")
    print(f"Input: {test_input}")
    
    # RAG 추론 실행
    generated_response, context_used = rag_inference(
        test_instruction, test_input, faiss_index, user_data, top_k=1
    )
    
    print("\n--- 검색된 컨텍스트(User "+user_id_input+") ---")
    print(context_used)
    
    print("\n--- 모델 생성 결과 ---")
    print(generated_response)