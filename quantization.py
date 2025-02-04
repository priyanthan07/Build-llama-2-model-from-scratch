import torch
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

def quantize_model(checkpoints_dir: str, tokenizer_dir: str, save_dir: str, quantization_type: str, device: str = "cpu"):
    """
    Quantize and save the model in the specified format.

    Args:
        checkpoints_dir: Path to the directory containing model checkpoints.
        tokenizer_dir: Path to the tokenizer model file.
        save_dir: Directory to save the quantized model.
        quantization_type: "int8" or "int4" quantization type.
        device: Device to use ("cpu" or "cuda").
    """
    # Load parameters
    with open(Path(checkpoints_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    # Load tokenizer
    tokenizer = SentencePieceProcessor()
    if not tokenizer.Load(tokenizer_dir):
        raise ValueError(f"Failed to load tokenizer from {tokenizer_dir}")

    # Set the vocab size
    vocab_size = tokenizer.vocab_size()
    assert vocab_size > 0, "Tokenizer vocab size is invalid"
    params["vocab_size"] = vocab_size

    # Load the original checkpoint
    checkpoint_path = sorted(Path(checkpoints_dir).glob("*.pth"))[0]
    print("Loading checkpoint from ", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device, mmap=True)
    print("state dict ", state_dict)
    # Remove unnecessary keys if present
    if "rope.freqs" in state_dict:
        del state_dict["rope.freqs"]

    quantized_state_dict = {}
    quantization_params = {
        "quant_type": quantization_type,
        "quant_method": "quant_method"
        }
    
    total_error = 0  # Initialize total quantization error
    processed_layers = 0

    # Quantize layer by layer
    for name, tensor in tqdm(state_dict.items(), desc="Quantizing layers"):
        print(f"Quantizing {name}...")
        
        # Skip quantization for small tensors (e.g., biases)
        if "norm" in name or tensor.numel() < 1024:
            quantized_state_dict[name] = tensor
            continue
        
        
        # Process as float32 to save memory
        tensor = tensor.to(torch.float32)
        
        max_val = tensor.abs().max().item()
        q_min, q_max = (-127, 127)
        scale = max_val / max(abs(q_min), q_max)
        
        # Handle zero-scale edge case
        scale = scale if scale != 0 else 1e-6
        
        if quantization_type == "int8":
            # PyTorch native quantization for better compatibility
            quantized = torch.quantize_per_tensor(
                tensor,
                scale=scale,
                zero_point=0,  # Symmetric quantization
                dtype=torch.qint8
            )
            
            # Store quantization parameters
            quantized_state_dict[f"{name}.quant"] = quantized
            quantized_state_dict[f"{name}.scale"] = torch.tensor(scale)
            
        else:
            raise ValueError(f"Unsupported quantization: {quantization_type}")     
        
           
        dequantized = quantized.dequantize()
        error = torch.mean((tensor - dequantized) ** 2).item()

        total_error += error
        processed_layers += 1
        print(f"{name:40} MSE: {error:.2e} | Scale: {scale:.4f}")
        
        # Free memory immediately
        del tensor, dequantized
        if "cuda" in device:
            torch.cuda.empty_cache()
            
    # Compute and print average quantization error
    print(f"\nAverage Quantization Error: {total_error/processed_layers:.4e}")
            
    # Save quantized model
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save quantized weights
    torch.save(quantized_state_dict, save_path / "consolidated.00.pth")
    print("quantized state dict ", quantized_state_dict)
    
    # Update and save config
    params.update(quantization_params)
    with open(save_path / "params.json", "w") as f:
        json.dump(params, f, indent=4)
        
    print(f"Quantized model saved to {save_dir} (Size reduction: ~{4 if quantization_type == 'int8' else 8}x)")


if __name__ == "__main__":
    quantize_model(
        checkpoints_dir="Llama-2-7b",
        tokenizer_dir="tokenizer.model",
        save_dir="Llama-2-7b-quantized",
        quantization_type="int8",
        device="cpu"
    )
