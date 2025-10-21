# from datasets import load_from_disk
# from collections import defaultdict

# def check_consistency():
#     # 데이터셋 로드
#     dataset = load_from_disk("/home/minhae/diffusion/dllm/examples/llada/dataset/gsm8k_match_index_llm_answer2")
    
#     # index별로 데이터 그룹화
#     index_groups = defaultdict(list)
#     for item in dataset:
#         index_groups[item['index']].append(item)
    
#     # 각 index 그룹 내에서 일관성 체크
#     all_consistent = True
#     for index, items in index_groups.items():
#         # 첫 번째 항목의 값들을 기준으로 설정
#         ref_question = items[0]['question']
#         ref_gold_answer = items[0]['gold_answer']
#         ref_gold_solution = items[0]['gold_solution']
        
#         # 같은 index를 가진 모든 항목들과 비교
#         for item in items[1:]:
#             if (item['question'] != ref_question or 
#                 item['gold_answer'] != ref_gold_answer or 
#                 item['gold_solution'] != ref_gold_solution):
#                 print(f"\nInconsistency found in index {index}:")
#                 print(f"Reference question: {ref_question}")
#                 print(f"Current question: {item['question']}")
#                 print(f"Reference gold_answer: {ref_gold_answer}")
#                 print(f"Current gold_answer: {item['gold_answer']}")
#                 print(f"Reference gold_solution: {ref_gold_solution}")
#                 print(f"Current gold_solution: {item['gold_solution']}")
#                 all_consistent = False
    
#     if all_consistent:
#         print("\nAll index groups are consistent!")
#         # 기본 통계 출력
#         print(f"\nTotal number of unique indices: {len(index_groups)}")
#         print(f"Distribution of answers per index:")
#         answer_counts = [len(items) for items in index_groups.values()]
#         min_count = min(answer_counts)
#         max_count = max(answer_counts)
#         avg_count = sum(answer_counts) / len(answer_counts)
#         print(f"Min answers per question: {min_count}")
#         print(f"Max answers per question: {max_count}")
#         print(f"Average answers per question: {avg_count:.2f}")
#     else:
#         print("\nInconsistencies found! Please check the details above.")

# if __name__ == "__main__":
#     check_consistency()

from datasets import load_from_disk, Dataset
from collections import defaultdict

def reorganize_dataset():
    # 데이터셋 로드
    dataset = load_from_disk("/home/minhae/diffusion/dllm/examples/llada/dataset/gsm8k_match_index_llm_answer2")
    
    # index별로 데이터 그룹화
    grouped_data = defaultdict(lambda: {
        'value': [],
        'llm_answer': [],
        'question': None,
        'gold_answer': None,
        'gold_solution': None,
        'match_type': None
    })
    
    # 데이터 그룹화
    for item in dataset:
        idx = item['index']
        # value와 llm_answer는 리스트에 추가
        grouped_data[idx]['value'].append(item['value'])
        grouped_data[idx]['llm_answer'].append(item['LLM_answer'])
        
        # 나머지 필드는 모두 동일하므로 한 번만 저장
        if grouped_data[idx]['question'] is None:  # 첫 번째 항목일 때만 저장
            grouped_data[idx]['question'] = item['question']
            grouped_data[idx]['gold_answer'] = item['gold_answer']
            grouped_data[idx]['gold_solution'] = item['gold_solution']
            grouped_data[idx]['match_type'] = item['match_type']
    
    # 새로운 데이터셋 형식으로 변환
    new_data = {
        'index': [],
        'value': [],
        'question': [],
        'gold_answer': [],
        'llm_answer': [],
        'gold_solution': [],
        'match_type': []
    }
    
    for idx, data in grouped_data.items():
        new_data['index'].append(idx)
        new_data['value'].append(data['value'])
        new_data['question'].append(data['question'])
        new_data['gold_answer'].append(data['gold_answer'])
        new_data['llm_answer'].append(data['llm_answer'])
        new_data['gold_solution'].append(data['gold_solution'])
        new_data['match_type'].append(data['match_type'])
    
    # 새로운 Dataset 생성
    new_dataset = Dataset.from_dict(new_data)
    
    # 기본 통계 출력
    print(f"Original dataset size: {len(dataset)}")
    print(f"New dataset size (unique indices): {len(new_dataset)}")
    print("\nSample data:")
    print(f"First row: {new_dataset[0]}")
    print(f"\nNumber of LLM answers for first row: {len(new_dataset[0]['llm_answer'])}")
    
    # 저장
    output_dir = "/home/minhae/diffusion/dllm/examples/llada/dataset/gsm8k_grouped_by_index2"
    new_dataset.save_to_disk(output_dir)
    print(f"\nDataset saved to {output_dir}")
    
    return new_dataset

if __name__ == "__main__":
    reorganize_dataset()