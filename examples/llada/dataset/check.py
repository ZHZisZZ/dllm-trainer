from datasets import load_from_disk

from datasets import load_from_disk
raw_data = load_from_disk('/home/minhae/diffusion/dllm/examples/llada/dataset/gsm8k_matched_all_index')
print('Dataset features:', raw_data.features)
print('First sample keys:', list(raw_data[0].keys()))
print('First sample:')
for key, value in raw_data[0].items():
    if isinstance(value, str):
        print(f'{key}: {value[:100]}...')
    else:
        print(f'{key}: {type(value)} - {value}')