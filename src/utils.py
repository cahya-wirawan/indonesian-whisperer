from datasets import load_dataset, concatenate_datasets


def load_dataset_combination(dataset_name, dataset_config_name, split="train", dataset_data_dir="",
                             streaming=True, shuffle=False, seed=42, **kwargs):
    """
    Utility function to load a dataset in streaming mode. For datasets with multiple splits,
    each split is loaded individually and then splits combined by taking alternating examples from
    each (interleaving).
    """
    dataset_name = [dn.strip() for dn in dataset_name.split(",")]
    dataset_config_name = [dcn.strip() for dcn in dataset_config_name.split(",")]
    dataset_data_dir = [dataset_data_dir.strip() if len(dataset_data_dir) != 0 else None
                         for dataset_data_dir in dataset_data_dir.split(",")]
    split = [sp.strip() for sp in split.split(",")]
    print("dataset_name:", dataset_name)
    print("dataset_config_name:", dataset_config_name)
    print("dataset_data_dir:", dataset_data_dir)
    print("split:", split)
    if len(dataset_name) != len(dataset_config_name) or len(dataset_name) != len(split) \
            or len(dataset_name) != len(dataset_data_dir):
        raise ValueError("dataset_name, dataset_config_name, dataset_data_dir, and split must have the same number of elements")
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
    if shuffle:
        dataset_combination = dataset_combination.shuffle(seed=seed)
    return dataset_combination
