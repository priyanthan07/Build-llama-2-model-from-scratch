import torch
from pathlib import Path
device = "cpu"
checkpoints_dir = "Llama-2-7b/"

checkpoint_path = sorted(Path(checkpoints_dir).glob("*.pth"))[0]
print("Loading checkpoint from ", checkpoint_path)
state_dict = torch.load(checkpoint_path, map_location=device, mmap=True)

i = 0 
for k, v in state_dict.items():
    print("type", v )
    if i == 0:
        break
    i += 1