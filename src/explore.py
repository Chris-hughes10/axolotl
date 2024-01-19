from functools import partial
import os
import numpy as np
from tokenizers import AddedToken
import torch
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset

import transformers, math

from axolotl.prompt_strategies.alpaca_w_system import OpenOrcaPromptTokenizingStrategy, OpenOrcaSystemDataPrompter
from axolotl.utils.data import md5

from axolotl.utils.samplers import MultipackBatchSampler
from torch.utils.data import DataLoader, RandomSampler




def main():
    config = {
        "base_model": "mistralai/Mistral-7B-v0.1",
        "model_type": "MistralForCausalLM",
        "tokenizer_type": "LlamaTokenizer",
        "is_mistral_derived_model": True,
        "load_in_8bit": False,
        "load_in_4bit": False,
        "strict": False,
        "datasets": [{...}],
        "dataset_prepared_path": None,
        "val_set_size": 0.05,
        "output_dir": "./out",
        "sequence_len": 8192,
        "sample_packing": True,
        "pad_to_sequence_len": True,
        "train_on_inputs": False,
        "group_by_length": False,
        "bf16": True,
        "fp16": False,
        "tf32": False,
        "gradient_checkpointing": True,
        "flash_attention": False,
        "special_tokens": {'bos_token': '<s>', 'eos_token': '<|im_end|>', 'unk_token': '<unk>'}
    }
    tokenizer_kwargs = {}

    model_config_name = config['base_model']

    model_config = AutoConfig.from_pretrained(
        model_config_name, trust_remote_code=True
    )

    ## load tokenizer and configure special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        model_config_name,
        trust_remote_code=True,
        use_fast=True,
        **tokenizer_kwargs,
    )

    if (
        tokenizer.__class__.__name__
        in [
            "LlamaTokenizer",
            "LlamaTokenizerFast",
            "CodeLlamaTokenizer",
            "CodeLlamaTokenizerFast",
        ]
        and hasattr(tokenizer, "pad_token")
        and not tokenizer.pad_token
    ):
        # set a pad_token, but use eos_token so we don't add a new token
        LLAMA_DEFAULT_EOS_TOKEN = '</s>'
        tokenizer.pad_token = LLAMA_DEFAULT_EOS_TOKEN

    # Mistral's official Flash Attention implementation requires left padding
    if config['is_mistral_derived_model'] and config['flash_attention'] and not config['sample_packing']:
        tokenizer.padding_side = "left"

    # TODO add special tokens to config
    # Add special tokens to tokenizer, if not present
    if config['special_tokens']:
        # lora_modules_to_save = get_linear_embedding_layers(model_config.model_type)
        for k, val in config['special_tokens'].items():
            # check if new special token is not already in tokenizer and
            # is adapter training to make sure lora_modules_to_save is set
            # pylint: disable=too-many-boolean-expressions
            # if (
            #     (getattr(tokenizer, k) is None or getattr(tokenizer, k) != val)
            #     and cfg.adapter
            #     and (
            #         not cfg.lora_modules_to_save
            #         or not all(
            #             x in cfg.lora_modules_to_save for x in lora_modules_to_save
            #         )
            #     )
            # ):
            #     lora_modules_to_save = ", ".join(
            #         [f"`{x}`" for x in lora_modules_to_save]
            #     )
            #     raise ValueError(
            #         f"Please set lora_modules_to_save to {lora_modules_to_save} when using an adapter and changing the special tokens."
            #     )

            tokenizer.add_special_tokens(
                {k: AddedToken(val, rstrip=False, lstrip=False, normalized=False)}
            )

        # If we add bos_token and eos_token, we need to update the post processor to
        # handle them correctly.
        # https://github.com/huggingface/transformers/pull/24132
        bos_or_eos_in_special_tokens = (
            "bos_token" in config['special_tokens'] and "eos_token" in config['special_tokens']
        )
        if (
            tokenizer.__class__.__name__
            in (
                "LlamaTokenizerFast",
                "CodeLlamaTokenizerFast",
            )
            and bos_or_eos_in_special_tokens
        ):
            tokenizer.update_post_processor()

        # if config['tokens']:
        #     tokenizer.add_tokens(
        #         [
        #             AddedToken(token, rstrip=False, lstrip=False, normalized=False)
        #             for token in config['tokens']
        #         ]
        #     )

    print(f"EOS: {tokenizer.eos_token_id} / {tokenizer.eos_token}")
    print(f"BOS: {tokenizer.bos_token_id} / {tokenizer.bos_token}")
    print(f"PAD: {tokenizer.pad_token_id} / {tokenizer.pad_token}")
    print(f"UNK: {tokenizer.unk_token_id} / {tokenizer.unk_token}")


    ## Download and prepare datasets
    ds_name = "openaccess-ai-collective/oo-gpt4-filtered"


    #TODO shard this for testing
    dataset = load_dataset(ds_name, streaming=False)

    # if local
    # if local_path.exists():
    #         if local_path.is_dir():
    #             # TODO dirs with arrow or parquet files could be loaded with `load_from_disk`
    #             ds = load_dataset(
    #                 config_dataset.path,
    #                 name=config_dataset.name,
    #                 data_files=config_dataset.data_files,
    #                 streaming=False,
    #                 split=None,
    #             )
    #         elif local_path.is_file():
    #             ds_type = get_ds_type(config_dataset)

    #             ds = load_dataset(
    #                 ds_type,
    #                 name=config_dataset.name,
    #                 data_files=config_dataset.path,
    #                 streaming=False,
    #                 split=None,
    #             )
    #         else:
    #             raise ValueError(
    #                 "unhandled dataset load: local path exists, but is neither a directory or a file"
    #             )


    # what about eval?
    dataset = dataset['train']
    dataset = dataset.shard(num_shards=100, index=0)
    features = dataset.features.keys()
    num_proc = min(64, os.cpu_count())
    prompter = OpenOrcaSystemDataPrompter()
    prompt_tokenizer = OpenOrcaPromptTokenizingStrategy(prompter=prompter, tokenizer=tokenizer,
                                                    train_on_inputs=config['train_on_inputs'],
                                                    sequence_len=config['sequence_len'],)


    out = prompt_tokenizer.tokenize_prompt(dataset[0])

    print('here')

    def parse_instruction_fields(prompt):
        """
        return instruction, input, response, system
        """
        return (
            prompt["question"],
            "",
            prompt["response"],
            prompt["system_prompt"],
        )

    def _tokenize(
        prompt: str, add_eos_token: bool = True, strip_bos_token: bool = False, max_length = config['sequence_len']
    ):

        if not prompt:
            raise ValueError("Empty text requested for tokenization.")


        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        if len(result["input_ids"]) == 0:
            raise ValueError("Tokenizer result is empty. You may want to audit your dataset")


        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < max_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if result["input_ids"][0] == tokenizer.bos_token_id and strip_bos_token:
            result["input_ids"] = result["input_ids"][1:]
            result["attention_mask"] = result["attention_mask"][1:]

        result["labels"] = result["input_ids"].copy()
        return result

    for prompt in dataset:
        (
            instruction,
            input,  # pylint: disable=redefined-builtin
            response,
            system,
        ) = parse_instruction_fields(prompt)

        intermediate = prompter.build_prompt_w_system(
                    system,
                    instruction,
                    input,
                )

        user_prompt = next(iter(intermediate))



        tokenized_prompt = _tokenize(user_prompt, add_eos_token=False)

        if not prompt_tokenizer.train_on_inputs:
            user_prompt_len = len(tokenized_prompt["input_ids"])
            # TODO this could be sped up using numpy array slicing
            tokenized_prompt["labels"] = [-100] * user_prompt_len

        tokenized_res_prompt = _tokenize(
            response, strip_bos_token=True, add_eos_token=True
        )
        tokenized_prompt["input_ids"] += tokenized_res_prompt["input_ids"]
        tokenized_prompt["attention_mask"] += tokenized_res_prompt["attention_mask"]
        tokenized_prompt["labels"] += tokenized_res_prompt["input_ids"]

        print(tokenized_prompt)
        break



    dataset = dataset.map(
            prompt_tokenizer.tokenize_prompt,
            num_proc=num_proc,
            remove_columns=features,
        )

    #{'input_ids': [1, 774, 10649, 28747, 13, 1976, 622, 347, 2078, ...], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, ...], 'labels': [-100, -100, -100, -100, -100, -100, -100, -100, -100, ...]}

    print('done')
    # dataset.save_to_disk(prepared_ds_path)

    val_set_size = 0.05

    to_hash_train = (
            dataset._fingerprint  # pylint: disable=protected-access
            + "|"
            + str(val_set_size)
            + "|"
            + "train"
            + "|"
            + str(42)
        )
    to_hash_test = (
        dataset._fingerprint  # pylint: disable=protected-access
        + "|"
        + str(val_set_size)
        + "|"
        + "test"
        + "|"
        + str(42)
    )
    train_fingerprint = md5(to_hash_train)
    test_fingerprint = md5(to_hash_test)

    dataset = dataset.train_test_split(
            test_size=val_set_size,
            shuffle=False,
            seed=42,
            train_new_fingerprint=train_fingerprint,
            test_new_fingerprint=test_fingerprint,
        )

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # pack datasets
    max_packed_sequence_len = config['sequence_len']

    def drop_long_seq(sample, sequence_len=2048):
        return len(sample["input_ids"]) <= sequence_len and len(sample["input_ids"]) > 0

    drop_long = partial(drop_long_seq, sequence_len=max_packed_sequence_len)

    def add_position_ids(sample):
        sample_len = len(sample["input_ids"])
        sample["position_ids"] = torch.arange(len(sample["input_ids"]))
        sample["length"] = sample_len
        return sample

    train_dataset = train_dataset.map(
                add_position_ids, num_proc=48 # cfg.dataset_processes
            )
    # Can optionally do this for eval set
    def get_dataset_lengths(dataset):
        if "length" in dataset.data.column_names:
            lengths = np.array(dataset.data.column("length"))
        else:
            lengths = (
                dataset.data.column("position_ids")
                .to_pandas()
                .apply(lambda x: x[-1] + 1)
                .values
            )
        return lengths

    max_input_len = np.max(get_dataset_lengths(train_dataset))

    dataset_processes = 48

    train_dataset = train_dataset.filter(drop_long, num_proc=dataset_processes)
    eval_dataset = eval_dataset.filter(drop_long, num_proc=dataset_processes)

    # if (
    #     "CodeGenTokenizer" in tokenizer.__class__.__name__
    #     or (cfg.is_mistral_derived_model and cfg.flash_attention)
    #     or cfg.model_config_type == "mamba"
    # ):
    #     train_dataset = train_dataset.remove_columns("attention_mask")
    #     if eval_dataset:
    #         eval_dataset = eval_dataset.remove_columns("attention_mask")

    # calculate total number of steps - do I need to do this?
    total_num_tokens = np.sum(
            train_dataset.data.column("input_ids")
            .to_pandas()
            .apply(lambda x: len(x))  # pylint: disable=unnecessary-lambda
            .values
        )
    total_supervised_tokens = (
            train_dataset.data.column("labels")
            .to_pandas()
            .apply(lambda x: np.sum(np.array(x) != -100))
            .sum()
        )

    # we have to drop anything longer then sequence len otherwise
        # flash attention with position ids fails

    micro_batch_size = 2



    sampler = MultipackBatchSampler(
                sampler=RandomSampler(train_dataset),
                batch_size=micro_batch_size,
                drop_last=True,
                batch_max_len=micro_batch_size
                * max_packed_sequence_len,
                lengths=get_dataset_lengths(train_dataset),
            )

    data_loader = DataLoader(
        train_dataset.remove_columns(["length"]),
        batch_sampler=sampler,
        collate_fn=DataCollatorWithPadding(tokenizer))
    data_loader_len = len(data_loader)
    actual_eff = sampler.efficiency()

    for batch in data_loader:
        print(batch)
        break

    num_epochs = 1

    total_num_steps = int(
                math.floor(
                    data_loader_len
                    * num_epochs
                    / int(os.environ.get("WORLD_SIZE", 1))
                )
            )


    # can now train
    # if cfg.is_mistral_derived_model and cfg.flash_attention and cfg.sample_packing:
    #     from axolotl.monkeypatch.mistral_attn_hijack_flash import (
    #         replace_mistral_attn_with_flash_attn,
    #     )

    #     LOG.info("patching with flash attention")
    #     replace_mistral_attn_with_flash_attn(packed=cfg.sample_packing)

    base_model = 'mistralai/Mistral-7B-v0.1'
    model_type = 'MistralForCausalLM'
    model = getattr(transformers, model_type).from_pretrained(
                    base_model,
                    config=model_config,
                    # load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
                    # load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
                    # trust_remote_code=cfg.trust_remote_code or False,
                    # **model_kwargs,
                )







if __name__ == "__main__":
    main()
