from utils import load_dataset_combination


streaming = True
ds = load_dataset_combination("mozilla-foundation/common_voice_11_0, mozilla-foundation/common_voice_10_0",
                              "mk, ne-NP", split="train+validation+other+test, other",
                              shuffle=True, streaming=streaming, use_auth_token=True)

for i, row in enumerate(ds):
    print(i, row["locale"], row["sentence"])
    if i >= 10:
        break