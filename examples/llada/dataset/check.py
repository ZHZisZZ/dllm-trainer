from datasets import load_from_disk

def check_invalid_samples(dataset):
    # 이상치 조건 체크
    invalid_samples = []
    print(dataset.column_names)
    print(dataset[0])
    for idx, sample in enumerate(dataset):
        if (sample['index'] == -1 or
            sample['question'] == "" or
            sample['LLM_answer'] == "" or
            sample['gold_answer'] == None or
            sample['gold_solution'] == None or
            sample['match_source'] == "" or
            sample['match_type'] == ""):
            invalid_samples.append((idx, sample))

    # 결과 출력
    print(f"\nTotal samples: {len(dataset)}")
    print(f"Invalid samples found: {len(invalid_samples)}")
    
    if invalid_samples:
        print("\nExample of invalid sample:")
        idx, sample = invalid_samples[0]
        print(f"Sample index in dataset: {idx}")
        for key, value in sample.items():
            if isinstance(value, str):
                print(f"{key}: {value[:100]}...")
            else:
                print(f"{key}: {type(value)} - {value}")
        
        # 각 필드별 "none" 또는 -1 카운트
        field_counts = {
            'index=-1': sum(1 for _, s in invalid_samples if s['index'] == -1),
            'question=""': sum(1 for _, s in invalid_samples if s['question'] == ""),
            'LLM_answer=""': sum(1 for _, s in invalid_samples if s['LLM_answer'] == ""),
            'gold_answer=None': sum(1 for _, s in invalid_samples if s['gold_answer'] == None),
            'gold_solution=None': sum(1 for _, s in invalid_samples if s['gold_solution'] == None),
            'match_source=""': sum(1 for _, s in invalid_samples if s['match_source'] == ""),
            'match_type=""': sum(1 for _, s in invalid_samples if s['match_type'] == "")
        }
        print("\nBreakdown of invalid fields:")
        for field, count in field_counts.items():
            print(f"{field}: {count} samples")

# raw_data = load_from_disk('/home/minhae/diffusion/dllm/examples/llada/dataset/gsm8k_match_index_llm_answer2')
# check_invalid_samples(raw_data)
data = load_from_disk('/home/minhae/diffusion/dllm/examples/llada/dataset/gsm8k_1_0_1')
# print(data.column_names)
print(data['train'][0])
print(len(data['train']))
print(len(data['test']))