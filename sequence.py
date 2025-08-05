import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 

model_dir = ".../llama" 
input_path = ".../prompt_sequence.npy"
save_path = ".../prediction"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float32,
    trust_remote_code=True
).eval()
print("Model loaded")

full_seq = np.load(input_path)  # shape: [5, H, W]
flat_seq = full_seq.reshape(full_seq.shape[0], -1)  # [5, T]
prompt_list = flat_seq[:4]  
query_T = flat_seq[4]     
input_ids = np.concatenate(prompt_list).astype(np.int64)
input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
print(f"Total prompt token length: {input_ids.shape[1]}")

max_new = query_T.shape[0]  
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
generated = outputs[0, -max_new:].cpu().numpy()

H = W = int(np.sqrt(max_new))
gen_map = generated.reshape(H, W)
np.save(save_path, gen_map)
print(f"Prediction saved to: {save_path}")
