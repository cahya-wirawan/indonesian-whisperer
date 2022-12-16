from utils import load_dataset_combination

ds = load_dataset_combination("mozilla-foundation/common_voice_11_0, mozilla-foundation/common_voice_10_0",
                              "lg, rw", split="train+validation, train", dataset_data_dir=",",
                              shuffle=True, streaming=True,  use_auth_token=True)
print(next(iter(ds)))