import json
import os
import re

def extract_putback_location(output_str):
    """
    [putback] 이후의 <물건> <장소> 부분을 추출
    예: [walk] ... [putback] <remote> <kitchen table> -> <remote> <kitchen table>
    """
    match = re.search(r'\[putback\]\s*(.*)', output_str)
    
    if match:
        return match.group(1).replace(' ', '').replace('\n', '').strip()
    else:
        return None

def evaluate_inference_results(user_id, result_filename, model_type):
    """
    추론 결과의 [putback] 이후 부분과 Ground Truth의 [putback] 이후 부분을 비교하여 평가
    """
    
    # 1. 파일 경로 설정 및 데이터 로드
    results_dir = "."
    results_path = os.path.join(results_dir, result_filename)
    gt_filename = f"User_{user_id}_test_dataset.json"
    gt_path = os.path.join("user_datasets", gt_filename)

    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
    except FileNotFoundError as e:
        print(f"오류: {user_id}의 결과 파일을 찾을 수 없습니다. 경로: {e}")
        return None

    # 2. 데이터 일치 여부 비교 및 평가
    exact_matches = 0
    total_samples = 0
    mismatched_instructions = []
    
    min_length = min(len(results_data), len(gt_data))
    
    for i in range(min_length):
        total_samples += 1
        
        # 예측된 전체 output과 정답 전체 output 추출
        predicted_full_output = results_data[i].get('output') 
        if not predicted_full_output:
            predicted_full_output = results_data[i].get('predicted_output_rag')

        gt_full_output = gt_data[i].get('output')
        
        # 해당 항목의 Instruction 추출
        instruction = gt_data[i].get('instruction', 'Instruction Not Found')

        if predicted_full_output is None or gt_full_output is None:
            continue

        # 3. [putback] 이후의 핵심 장소 정보 추출 및 비교
        
        # 예측된 [putback] 장소 추출
        predicted_location = extract_putback_location(predicted_full_output)
        
        # 정답 [putback] 장소 추출
        gt_location = extract_putback_location(gt_full_output)

        # 추출된 장소 정보가 일치하는지 비교
        if predicted_location == gt_location and predicted_location is not None:
            exact_matches += 1
        else:
            mismatched_instructions.append(instruction)
            
    # 4. 결과 반환
    if total_samples > 0:
        accuracy = (exact_matches / total_samples) * 100
        return {
            "user_id": user_id,
            "model_type": model_type,
            "total_samples": total_samples,
            "exact_matches": exact_matches,
            "accuracy": accuracy,
            "mismatched_instructions": mismatched_instructions
        }
    else:
        return None


if __name__ == "__main__":
    
    # 평가 대상 모델 입력
    print('평가 대상 (vanil 또는 llm 또는 rag)을 입력: ')
    model_input = input().lower()
    
    if model_input not in ['vanil', 'llm', 'rag']:
        print("유효하지 않은 입력입니다. 'llm' 또는 'rag'를 입력하세요.")
        exit()

    # 평가할 사용자 ID 목록
    users = ['A', 'B', 'C']
    all_results = []

    # 각 사용자에 대해 평가 실행
    for user_id in users:
        if model_input == 'llm':
            result_filename = f"lora_json/inference_results_User_{user_id}_test_dataset.json" 
        elif model_input == 'rag':
            result_filename = f"rag_json/rag_inference_test_results_{user_id}.json"
        elif model_input == 'vanil':
            result_filename = f'vanilla_json/inference_results_User_{user_id}_test_dataset.json'
        # 평가 함수 호출
        result = evaluate_inference_results(user_id, result_filename, model_input.upper())
        if result:
            all_results.append(result)


    print("\n" + "="*10)
    print(f"모델 ({model_input.upper()}) 평가 결과 요약 (putback 위치 기준)")
    print("="*10)
    
    for result in all_results:
        print(f"\n--- User {result['user_id']} 평가 ---")
        print(f"  총 샘플 수: {result['total_samples']}")
        print(f"  일치하는 샘플 수: {result['exact_matches']}")
        print(f"  정확도 (Accuracy): {result['accuracy']:.2f}%")
        
        # 불일치 instruction 출력
        if result['mismatched_instructions']:
            print(f"  [불일치 instruction ({len(result['mismatched_instructions'])}개)]")
            for instruction in result['mismatched_instructions']:
                print(f"    - {instruction}")
        else:
            print("  [불일치 instruction]: 없음")