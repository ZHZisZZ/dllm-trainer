import datasets
from datasets import load_from_disk


def get_gsm8k_dataset():
  import random
  random.seed(42) # Set fixed seed for reproducibility
  """Get GSM8K dataset for finetuning."""
  print(f"Loading GSM8K dataset from: /home/minhae/diffusion/dllm/examples/llada/dataset/before/0gsm8k_grouped_by_index")
  raw_data_dir =  "/home/minhae/diffusion/dllm/examples/llada/dataset/before/0gsm8k_grouped_by_index"
  raw_data = load_from_disk(raw_data_dir)
  print(f"Raw data loaded: {len(raw_data)} samples")

  # Use select instead of slicing to avoid string conversion
  train_size = int(len(raw_data) * 0.8)

  splits = {
    'train': raw_data.select(range(train_size)),
    'test': raw_data.select(range(train_size, len(raw_data)))
  }
  print(f"Train split: {len(splits['train'])} samples")
  print(f"Test split: {len(splits['test'])} samples")

  def format_data_q_cond(d):
    question = 'Question: ' + d['question']
    answer = ' Correct Answer: ' + d['gold_solution']+'The answer is: '+ d['gold_answer']
    messages = []
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})
    return messages, d["index"]
  
  def format_data_llm_cond(d):    
    # Randomly select one LLM answer if it's a list
    llm_answer_text = d['llm_answer'] if isinstance(d['llm_answer'], str) else random.choice(d['llm_answer'])
    llm_answer = 'LLM generated Answer: ' + llm_answer_text
    answer = ' Correct Answer: ' + d['gold_solution']+'The answer is: ' + d['gold_answer']
    messages = []
    messages.append({"role": "LLM generated Answer", "content": llm_answer})
    messages.append({"role": "assistant", "content": answer})
    return messages, d["index"]
  
  def format_data_q_llm_cond(d):
    question = 'Question: ' + d['question']
    llm_answer_text = d['llm_answer'] if isinstance(d['llm_answer'], str) else random.choice(d['llm_answer'])
    llm_answer = ' LLM generated Answer: ' + llm_answer_text
    answer = ' Correct Answer: ' + d['gold_solution']+'The answer is: ' + d['gold_answer']
    
    messages = []
    messages.append({"role": "user", "content": question})
    messages.append({"role": "LLM generated Answer", "content": llm_answer})
    messages.append({"role": "assistant", "content": answer})
    return messages, d["index"]


  # Process splits
  processed_splits = {}
  for split_name, split_data in splits.items():
    print(f"Processing {split_name} split...")
    processed_data = []
    for i, d in enumerate(split_data):
      if i % 1000 == 0:
        print(f"Processing sample {i}/{len(split_data)}")
      
      # Try each tokenization method and filter out None values
      try:
        if i %2 == 0:
          formatted_q, index = format_data_q_cond(d)
          if formatted_q is not None:
            processed_data.append({"id":index,"messages": formatted_q,"source":"gsm8k_q_cond"})
        # elif i %3 == 1:
        #   formatted_llm, index = format_data_llm_cond(d)
        #   if formatted_llm is not None:
        #     processed_data.append({"id":index,"messages": formatted_llm,"source":"gsm8k_llm_cond"})
        elif i %2 == 1:
          formatted_q_llm, index = format_data_q_llm_cond(d)
          if formatted_q_llm is not None:
            processed_data.append({"id":index,"messages": formatted_q_llm,"source":"gsm8k_q_llm_cond"})
      except Exception as e:
        print(f"Error processing sample {i}: {e}")
        print(f"Sample keys: {list(d.keys()) if hasattr(d, 'keys') else 'No keys'}")
        raise
    
    # Convert to HuggingFace Dataset
    processed_splits[split_name] = datasets.Dataset.from_list(processed_data)
    print(f"Processed {split_name}: {len(processed_data)} samples")
  
  # Create DatasetDict
  dataset_dict = datasets.DatasetDict(processed_splits)
  print('train samples:', len(dataset_dict['train']))
  print('test samples:', len(dataset_dict['test']))
  return dataset_dict


dataset_dict = get_gsm8k_dataset()

# Save as DatasetDict
dataset_dict.save_to_disk("/home/minhae/diffusion/dllm/examples/llada/dataset/gsm8k_filter_unique_1_0_1")



