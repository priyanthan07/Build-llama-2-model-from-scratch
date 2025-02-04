import os
import tqdm

def get_model_size(checkpoints_dir: str):
    total_size = 0
    for file in (os.listdir(checkpoints_dir)):
        file_path = os.path.join(checkpoints_dir, file)
        if os.path.isfile(file_path):
            total_size += os.path.getsize(file_path)
    return total_size
    
    
# For the original model
print(f"Original model size: {get_model_size('Llama-2-7b/') / (1024**3):.2f} GB")

# For the quantized model
print(f"Quantized model size: {get_model_size('Llama-2-7b-quantized/') / (1024**3):.2f} GB")
