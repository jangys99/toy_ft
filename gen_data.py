import json
import random
import os

random.seed(42) 

def generate_user_data_entry(item, room, location_full):
    """단일 데이터 항목(instruction, input, output)을 생성합니다."""
    location_obj = location_full.split(' ')[-1]
    
    instruction = f"Place the {item} on the {location_obj}"
    input_str = "<task> [relocate]"
    output_str = (
        f"[walk] <{room}> [find] <{item}> [grab] <{item}> [find] <{location_obj}> "
        f"[putback] <{item}> <{location_full}>"
    )
    return {"instruction": instruction, "input": input_str, "output": output_str}

def generate_and_save_optimized_preference_data(users_for_generation, output_dir="user_datasets", num_entries_per_user=30):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    allowed_rooms = ['kitchen', 'bedroom', 'livingroom', 'studyroom', 'bathroom']
    all_items = [
        "apple", "book", "cup", "keys", "remote", "wallet", "glasses", "phone", 
        "magazine", "pen", "umbrella", "charger", "headphones", "bottle", 
        "notebook", "shoes", "jacket", "hat", "bag", "toy"
    ]
    
    room_locations_map = {
        "kitchen": ["kitchen table", "kitchen counter", "kitchen shelf", "kitchen cupboard", "refrigerator", "sink"],
        "bedroom": ["bedroom table", "bedroom desk", "bedroom bookshelf", "bedroom drawer", "bedside table", "wardrobe"],
        "livingroom": ["livingroom table", "livingroom bookshelf", "livingroom couch", "TV stand", "coffee table", "fireplace mantel"],
        "studyroom": ["studyroom desk", "studyroom bookshelf", "studyroom drawer", "filing cabinet", "printer stand"],
        "bathroom": ["bathroom cabinet", "bathroom counter", "shower caddy", "under sink", "laundry basket"]
    }

    # 사용자별 특정 선호도 규칙 (물건 -> 선호 장소)
    user_specific_preferences = {
        "User A": { 
            "apple": "livingroom table", 
            "book": "studyroom bookshelf",
            "magazine": "livingroom table",
            "glasses": "livingroom coffee table", 
            "cup": "kitchen cupboard" 
        },
        "User B": { 
            "apple": "kitchen table", 
            "book": "bedroom table", 
            "magazine": "kitchen cupboard",
            "glasses": "kitchen counter", 
            "cup": "bedroom bookshelf" 
        },
        "User C": { 
            "apple": "bedroom table", 
            "book": "bedroom drawer", 
            "magazine": "studyroom desk",
            "glasses": "studyroom desk", 
            "cup": "bedroom nightstand" 
        }
    }

    # 사용자 데이터셋 간의 중복을 막기 위한 전역 세트
    global_used_combinations_text = set() 

    for user_name in users_for_generation.keys():
        current_user_dataset = []
        # 각 사용자 데이터셋 내에서 이미 사용된 (instruction, output) 조합 추적
        user_local_used_combinations = set() 
        # 각 사용자 데이터셋 내에서 이미 사용된 instruction 추적 (추가: 동일 instruction 방지)
        user_local_used_instructions = set()

        # 1. 사용자별 고정 선호 규칙을 먼저 데이터셋에 추가
        if user_name in user_specific_preferences:
            for item, location_full in user_specific_preferences[user_name].items():
                room = location_full.split(' ')[0] 
                
                if room not in allowed_rooms:
                    print(f"경고: {user_name}의 {item} 선호 장소 '{location_full}'의 방 '{room}'은 허용된 방 목록에 없습니다. 그래도 사용합니다.")
                
                entry = generate_user_data_entry(item, room, location_full)
                entry_tuple = (entry['instruction'], entry['output'])
                
                # 중복 instruction 및 중복 조합 확인
                if entry['instruction'] not in user_local_used_instructions and \
                   entry_tuple not in global_used_combinations_text and \
                   entry_tuple not in user_local_used_combinations:
                    
                    current_user_dataset.append(entry)
                    global_used_combinations_text.add(entry_tuple)
                    user_local_used_combinations.add(entry_tuple)
                    user_local_used_instructions.add(entry['instruction']) # instruction 추적 추가


        # 2. 나머지 필요한 데이터 개수만큼 무작위 조합으로 채우기
        remaining_items = [item for item in all_items if item not in user_specific_preferences.get(user_name, {})]
        
        possible_random_combinations = []
        # 사용자 정의 규칙에 없는 아이템에 대해서만 가능한 조합 풀 생성
        for item in remaining_items:
            for room in allowed_rooms:
                for location_full in room_locations_map.get(room, []):
                    if location_full.startswith(room): 
                        possible_random_combinations.append((item, room, location_full))
        
        random.shuffle(possible_random_combinations)

        # 필요한 만큼 더 생성
        additional_count = 0
        max_attempts = len(possible_random_combinations) * 2
        attempts = 0

        while len(current_user_dataset) < num_entries_per_user and attempts < max_attempts:
            if additional_count >= len(possible_random_combinations):
                break 

            item, room, location_full = possible_random_combinations[additional_count]
            entry = generate_user_data_entry(item, room, location_full)
            
            entry_tuple = (entry['instruction'], entry['output'])
            
            # 중복 instruction 및 중복 조합 확인
            if entry['instruction'] not in user_local_used_instructions and \
               entry_tuple not in global_used_combinations_text and \
               entry_tuple not in user_local_used_combinations:
                
                current_user_dataset.append(entry)
                global_used_combinations_text.add(entry_tuple)
                user_local_used_combinations.add(entry_tuple)
                user_local_used_instructions.add(entry['instruction']) # instruction 추적 추가
            
            additional_count += 1
            attempts += 1

        # 만약 가능한 조합이 적어 필요한 개수를 다 채우지 못했다면 경고
        if len(current_user_dataset) < num_entries_per_user:
            print(f"경고: User {user_name}의 경우, {len(current_user_dataset)}개만 생성되었습니다. 필요한 {num_entries_per_user}개를 채우기 위한 고유한 조합이 부족하거나, 다른 사용자와의 중복이 많을 수 있습니다.")
        
        # 파일 저장
        file_name = os.path.join(output_dir, f"{user_name.replace(' ', '_')}_dataset.json")
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(current_user_dataset, f, indent=2, ensure_ascii=False)
        print(f"'{file_name}' 파일이 총 {len(current_user_dataset)}개의 데이터로 성공적으로 생성되었습니다.")


# --- 사용자 정의 선호도 데이터 (규칙 정의 및 사용자 목록 식별 용도) ---
users_for_dataset_generation = {
    "User A": {}, 
    "User B": {}, 
    "User C": {}
}

# 데이터셋 생성 및 저장 실행
generate_and_save_optimized_preference_data(users_for_dataset_generation, num_entries_per_user=30)