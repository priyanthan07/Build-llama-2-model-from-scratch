from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
import os
from llama2_model import ModelArgs, Transformer

class Llama:
    
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args
    
    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device:str, quantized: bool = False):
        prev_time = time.time()
        if load_model:
            checkpoint_path = sorted(Path(checkpoints_dir).glob('*.pth'))[0]
            print(f'Loading chackpoint from {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=device,mmap=True)
            print(f'Checkpoint loaded in {time.time() - prev_time:.2f} sec')
            prev_time = time.time()
        
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
            
        updated_params = {}
        for k, v in params.items():
            if (k !='quant_method' and k != 'quant_type'):
                updated_params[k] = v
                  
        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **updated_params
        )
        
        tokenizer = SentencePieceProcessor()
        tokenizer.Load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        
        torch.set_default_dtype(torch.float16 if device == "cuda" else torch.bfloat16)

            
        model = Transformer(model_args).to(device)
        
        if load_model:
            if quantized:
                print("Dequantizing model weights...")
                new_state_dict = {}
                
                for name, param in tqdm(model.state_dict().items(), desc="Processing layers"):
                    # Try to find quantized parameters
                    quant_key = f"{name}.quant"
                    scale_key = f"{name}.scale"
                    zerop_key = f"{name}.zerop"
                    
                    if quant_key in checkpoint:
                        quant_tensor = checkpoint[quant_key]
                        scale = checkpoint[scale_key].item() if scale_key in checkpoint else 1.0
                        zerop = checkpoint[zerop_key].item() if zerop_key in checkpoint else 0
                        
                        if quant_tensor.is_quantized:
                            dequantized = quant_tensor.dequantize().to(param.dtype)
                        else:
                            raise ValueError(f"Unsupported quantized tensor type: {quant_tensor.dtype}")
                        
                        new_state_dict[name] = dequantized
                    elif name in checkpoint:
                        new_state_dict[name] = checkpoint[name].to(param.dtype)
                        
                    else:
                        new_state_dict[name] = param
                        print(f"Warning: Missing parameter {name} in checkpoint")
                
                model.load_state_dict(new_state_dict, strict=False)
            else:
                # Load non-quantized model
                if "rope.freqs" in checkpoint:
                    del checkpoint["rope.freqs"]
                model.load_state_dict(checkpoint, strict=True)
            
            print(f"Model loaded in {time.time() - prev_time:.2f}s")
        
        return Llama(model, tokenizer, model_args)

    def sample_top_p(self, probs, p):
        probs_sort, probs_index = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1) # cumulative sum
    
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token =  torch.gather(probs_index, -1, next_token)
        
        return next_token
    
    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float= 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.model_args.max_seq_len
            
        # Tokenize the prompts
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]  # add beginnig of sentence not end of sentence
    
        # check batch size
        batch_size = len(prompt_tokens)
        assert batch_size <= self.model_args.max_batch_size, f"Batch size {batch_size} exceeds the maximum batch size {self.model_args.max_batch_size}"
        # check maximum prompt length
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.model_args.max_seq_len, f"Maximum prompt length {max_prompt_len} exceeds the maximum sequence length {self.model_args.max_seq_len}"
        
        # how many tokens we need to generate
        total_len = min(self.model_args.max_seq_len, max_gen_len + max_prompt_len)
        
        # create the list that will  contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        
        for batch, token in enumerate(prompt_tokens):
            # populate initial tokens with the prompt tokens
            tokens[batch, :len(token)] = torch.tensor(token, dtype=torch.long, device=device)
            
        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
        
        for cur_pos in tqdm(range(1, total_len), desc="Generating tokens"):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:,-1] / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
                
            else:
                # greedy selection : select the token with the highest probability
                next_token = torch.argmax(logits[:,-1], dim=-1)
                
            next_token = next_token.reshape(-1)
            # Only replace the token if it is a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
                
            # EOS is reached only if we find the EOS token
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
    
            if all(eos_reached):
                break
            
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # cut to the EOS token, if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_index = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_index]
                
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.Decode(current_prompt_tokens))
        return(out_tokens, out_text)
    
if __name__ == "__main__":
    torch.manual_seed(0)
    allow_cuda = False
    device = "cuda" if allow_cuda and torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        torch.set_num_threads(os.cpu_count()//2)
    
    prompts = [
        "What is the sum of 10 + 20?"
    ]
    
    llama = Llama.build(
        checkpoints_dir="Llama-2-7b-quantized/",
        tokenizer_path="tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device,
        quantized=True
    )
    
    print("Quantized model loaded successfully")
    
    # inference the model
    out_tokens, out_text = (llama.text_completion(prompts=prompts, max_gen_len=64))
    assert len(out_text) == len(prompts)
    for i in range(len(out_text)):
        print(f'{out_text[i]}')
        print('-' * 50)
        
    
    