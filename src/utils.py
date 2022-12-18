from copy import deepcopy
import torch
from transformers import WhisperForConditionalGeneration
from datasets import load_dataset, concatenate_datasets
from pathlib import Path


def min_filesize_and_textlength(row, filesize, textlength):
    return (Path(row['path']).stat().st_size >= filesize) and (len(row['sentence']) >= textlength)


def load_dataset_combination(dataset_name, dataset_config_name, split="train", dataset_data_dir=None,
                             streaming=True, shuffle=False, seed=42,
                             dataset_min_filesize=0, dataset_min_textlength=0, **kwargs):
    """
    Utility function to load a dataset in streaming mode. For datasets with multiple splits,
    each split is loaded individually and then splits combined by taking alternating examples from
    each (interleaving).
    """
    dataset_name = [dn.strip() for dn in dataset_name.split(",")]
    dataset_config_name = [dcn.strip() for dcn in dataset_config_name.split(",")]
    if dataset_data_dir is not None:
        dataset_data_dir = [dataset_data_dir.strip() if len(dataset_data_dir) != 0 else None
                            for dataset_data_dir in dataset_data_dir.split(",")]
    else:
        dataset_data_dir = [None for _ in dataset_name]
    split = [sp.strip() for sp in split.split(",")]
    print("dataset_name:", dataset_name)
    print("dataset_config_name:", dataset_config_name)
    print("dataset_data_dir:", dataset_data_dir)
    print("split:", split)
    if len(dataset_name) != len(dataset_config_name) or len(dataset_name) != len(split) \
            or len(dataset_name) != len(dataset_data_dir):
        raise ValueError(
            "dataset_name, dataset_config_name, dataset_data_dir, and split must have the same number of elements")
    dataset_combination = []
    for i in range(len(dataset_name)):
        print("dataset_name[i]:", i, dataset_name[i])
        print("dataset_data_dir[i]:", i, dataset_data_dir[i])
        if "+" in split[i]:
            # load multiple splits separated by the `+` symbol with streaming mode
            dataset_splits = [
                load_dataset(dataset_name[i], dataset_config_name[i], split=split_name,
                             data_dir=dataset_data_dir[i], streaming=streaming, **kwargs)
                for split_name in split[i].split("+")
            ]
            dataset_combination += dataset_splits
        else:
            # load a single split *with* streaming mode
            dataset = [load_dataset(dataset_name[i], dataset_config_name[i], split=split[i],
                                    data_dir=dataset_data_dir[i], streaming=streaming, **kwargs)]
            dataset_combination += dataset
    dataset_combination = concatenate_datasets(dataset_combination)
    if not streaming and (dataset_min_filesize != 0 or dataset_min_textlength != 0):
        dataset_combination = dataset_combination.filter(
            min_filesize_and_textlength,
            fn_kwargs={
                "filesize": dataset_min_filesize,
                "textlength": dataset_min_textlength
            })
    if shuffle:
        dataset_combination = dataset_combination.shuffle(seed=seed)
    return dataset_combination


WHISPER_MAPPING = {
    "layers": "blocks",
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
    "layers": "blocks",
    ".self_attn.q_proj": ".attn.query",
    ".self_attn.k_proj": ".attn.key",
    ".self_attn.v_proj": ".attn.value",
    ".self_attn_layer_norm": ".attn_ln",
    ".self_attn.out_proj": ".attn.out",
    ".encoder_attn.q_proj": ".cross_attn.query",
    ".encoder_attn.k_proj": ".cross_attn.key",
    ".encoder_attn.v_proj": ".cross_attn.value",
    ".encoder_attn_layer_norm": ".cross_attn_ln",
    ".encoder_attn.out_proj": ".cross_attn.out",
    "decoder.layer_norm.": "decoder.ln.",
    "encoder.layer_norm.": "encoder.ln_post.",
    "embed_tokens": "token_embedding",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "layer_norm": "ln_post",
}

###
# The code to convert hf to whisper copied from https://github.com/bayartsogt-ya/whisper-multiple-hf-datasets
###

def rename_keys(s_dict):
    keys = list(s_dict.keys())
    for key in keys:
        new_key = key
        for k, v in WHISPER_MAPPING.items():
            if k in key:
                new_key = new_key.replace(k, v)

        print(f"{key} -> {new_key}")

        s_dict[new_key] = s_dict.pop(key)
    return s_dict


def convert_hf_whisper(hf_model_name_or_path: str, whisper_state_path: str):
    transformer_model = WhisperForConditionalGeneration.from_pretrained(hf_model_name_or_path)
    config = transformer_model.config

    # first build dims
    dims = {
        'n_mels': config.num_mel_bins,
        'n_vocab': config.vocab_size,
        'n_audio_ctx': config.max_source_positions,
        'n_audio_state': config.d_model,
        'n_audio_head': config.encoder_attention_heads,
        'n_audio_layer': config.encoder_layers,
        'n_text_ctx': config.max_target_positions,
        'n_text_state': config.d_model,
        'n_text_head': config.decoder_attention_heads,
        'n_text_layer': config.decoder_layers
    }

    state_dict = deepcopy(transformer_model.model.state_dict())
    state_dict = rename_keys(state_dict)

    torch.save({"dims": dims, "model_state_dict": state_dict}, whisper_state_path)