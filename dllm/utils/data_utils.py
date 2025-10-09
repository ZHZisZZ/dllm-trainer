import random
import warnings
from typing import TYPE_CHECKING

import torch
import datasets

if TYPE_CHECKING:
    from dllm.utils.configs import ModelArguments, DataArguments, TrainingArguments


class ConstantLengthDataset(datasets.IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
    The dataset also formats the text before tokenization with a specific format that is provided
    by the user.

    Args:
        tokenizer (`transformers.PreTrainedTokenizer`):
            The processor used for processing the data.
        dataset (`dataset.Dataset`):
            Dataset with text files.
        dataset_text_field (`str` or `None`, *optional*, defaults to `None`):
            Name of the field in the dataset that contains the text. Only one of `dataset_text_field` and
            `formatting_func` should be provided.
        formatting_func (`Callable`, *optional*):
            Function that formats the text before tokenization. Usually it is recommended to follow a certain
            pattern such as `"### Question: {question} ### Answer: {answer}"`. Only one of `dataset_text_field` and
            `formatting_func` should be provided.
        infinite (`bool`, *optional*, defaults to `False`):
            If True the iterator is reset after dataset reaches end else stops.
        seq_length (`int`, *optional*, defaults to `1024`):
            Length of token sequences to return.
        num_of_sequences (`int`, *optional*, defaults to `1024`):
            Number of token sequences to keep in buffer.
        chars_per_token (`int`, *optional*, defaults to `3.6`):
            Number of characters per token used to estimate number of tokens in text buffer.
        eos_token_id (`int`, *optional*, defaults to `0`):
            Id of the end of sequence token if the passed tokenizer does not have an EOS token.
        shuffle (`bool`, *optional*, defaults to `True`)
            Shuffle the examples before they are returned
        append_concat_token (`bool`, *optional*, defaults to `True`)
            If true, appends `eos_token_id` at the end of each sample being packed.
        add_special_tokens (`bool`, *optional*, defaults to `True`)
            If true, tokenizers adds special tokens to each sample being packed.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        dataset_text_field=None,
        formatting_func=None,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        eos_token_id=0,
        shuffle=True,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
        )
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        self.append_concat_token = append_concat_token
        self.add_special_tokens = add_special_tokens

        if dataset_text_field is not None and formatting_func is not None:
            warnings.warn(
                "Only one of `dataset_text_field` and `formatting_func` should be provided. "
                "Ignoring `dataset_text_field` and using `formatting_func`.",
                UserWarning,
            )

        if formatting_func is not None:
            self.formatting_func = formatting_func
        elif dataset_text_field is not None:
            self.formatting_func = lambda x: x[dataset_text_field]
        else:  # neither is provided
            raise ValueError(
                "Either `dataset_text_field` or `formatting_func` should be provided."
            )

        self.pretokenized = False
        column_names = (
            dataset.column_names
            if isinstance(dataset, (datasets.Dataset, datasets.IterableDataset))
            else None
        )
        if column_names is not None and "input_ids" in column_names:
            self.pretokenized = True
            # since the dataset is tokenized, the unit of buffer size should be tokens
            self.max_buffer_size = seq_length * num_of_sequences

        self._epoch = 0  # TODO: ?

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(self.formatting_func(next(iterator)))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            if self.shuffle:
                random.shuffle(buffer)
            if self.pretokenized:
                tokenized_inputs = buffer
            else:
                tokenized_inputs = self.tokenizer(
                    buffer, add_special_tokens=self.add_special_tokens, truncation=False
                )["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                if self.append_concat_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                # Shuffle again, otherwise split examples occur in consecutive tensors.
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


def clip_row(row: dict, max_length: int, truncation: str = "right") -> dict:
    for key in ("input_ids", "labels", "attention_mask"):
        if key in row:
            if truncation == "right":
                row[key] = row[key][:max_length]
            elif truncation == "left":
                row[key] = row[key][-max_length:]
            else:
                raise NotImplementedError
    return row


def post_process_dataset(
    dataset: datasets.DatasetDict, data_args: "DataArguments"
) -> datasets.DatasetDict:
    if data_args.truncation == "filter":
        return dataset.filter(
            lambda row: len(row["input_ids"]) <= data_args.max_length,
            num_proc=data_args.num_proc,
        )
    elif data_args.truncation == "right":
        # do this only if dataset has "prompt_len"
        if "prompt_len" in dataset.column_names["train"]:
            dataset = dataset.filter(
                lambda row: row["prompt_len"] <= data_args.max_length,
                num_proc=data_args.num_proc,
            )
        return dataset.map(
            lambda row: clip_row(row, data_args.max_length, truncation="right"),
            num_proc=data_args.num_proc,
        )
    else:
        raise NotImplementedError


def clip_row_streaming(row: dict, max_length: int, truncation: str = "right") -> dict:
    """Clip whole sequence OR (if prompt_len present) preserve prompt and clip only the response."""
    if truncation not in {"right", "left"}:
        raise NotImplementedError(f"Unknown truncation: {truncation}")

    def clip(seq):
        return seq[:max_length] if truncation == "right" else seq[-max_length:]

    def clip_preserve_prompt(seq, prompt_len: int):
        prompt = seq[:prompt_len]
        resp = seq[prompt_len:]
        budget = max(0, max_length - len(prompt))
        resp = resp[:budget] if truncation == "right" else resp[-budget:]
        return prompt + resp

    prompt_len = row.get("prompt_len", None)
    for k in ("input_ids", "labels", "attention_mask"):
        if k in row and isinstance(row[k], list):
            row[k] = (
                clip_preserve_prompt(row[k], prompt_len)
                if isinstance(prompt_len, int) and prompt_len >= 0
                else clip(row[k])
            )
    return row


def post_process_dataset_streaming(
    dataset: datasets.IterableDatasetDict,
    data_args: "DataArguments",
) -> datasets.IterableDatasetDict:

    def _train_has_prompt_len_streaming(dataset: datasets.IterableDatasetDict) -> bool:
        """Replicates: 'if \"prompt_len\" in dataset.column_names[\"train\"]' for streaming."""
        it = dataset["train"].take(1)
        try:
            ex = next(iter(it))
        except StopIteration:
            return False
        return "prompt_len" in ex

    mode = data_args.truncation
    max_len = data_args.max_length

    if mode == "filter":
        # Keep rows with len(input_ids) <= max_len (emulate .filter with generator map)
        def keep_if_short(row):
            if (
                "input_ids" in row
                and isinstance(row["input_ids"], list)
                and len(row["input_ids"]) <= max_len
            ):
                yield row  # keep
            # else: drop (yield nothing)

        return datasets.IterableDatasetDict(
            {name: ds.map(keep_if_short) for name, ds in dataset.items()}
        )

    elif mode == "right":
        ds_out = dataset

        # Do this only if TRAIN split has "prompt_len" (same condition as your non-streaming code)
        if _train_has_prompt_len_streaming(ds_out):

            def keep_if_prompt_fits(row):
                pl = row.get("prompt_len", None)
                if isinstance(pl, int) and pl <= max_len:
                    yield row  # keep
                elif pl is None:
                    # If a row lacks prompt_len but train had it, the non-streaming code would try to access it and fail.
                    # Here we conservatively drop such rows to mirror "requires prompt_len <= max_len".
                    return
                # else: drop

            ds_out = datasets.IterableDatasetDict(
                {name: ds.map(keep_if_prompt_fits) for name, ds in ds_out.items()}
            )

        # Then clip right (same clipping as clip_row)
        def clip_right(row):
            return clip_row(row, max_len, truncation="right")

        return datasets.IterableDatasetDict(
            {name: ds.map(clip_right) for name, ds in ds_out.items()}
        )

    else:
        raise NotImplementedError
