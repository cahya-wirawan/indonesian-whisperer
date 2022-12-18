from utils import load_dataset_combination

streaming = True
"""
ds = load_dataset_combination("mozilla-foundation/common_voice_11_0, mozilla-foundation/common_voice_11_0",
                              "mk, ne-NP", split="train+validation+other+test, other", dataset_data_dir=",",
                              shuffle=True, streaming=streaming, use_auth_token=True)
"""
ds = load_dataset_combination("mozilla-foundation/common_voice_11_0, cahya/fleurs",
                              "mk, id_id", split="train, train",
                              rename_columns=",",
                              shuffle=True, streaming=streaming, use_auth_token=True)
"""
ds = load_dataset_combination("cahya/newspaper-filtered, wikitext",
                              "kompas-2013, wikitext-2-v1", split="train, train", dataset_data_dir=",",
                              rename_columns="text:words,text:words",
                              shuffle=True, streaming=streaming, use_auth_token=True)
"""
for i, row in enumerate(ds):
    print(i, row["locale"], row["sentence"])
    if i > 10:
        break
