import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # Load models

# ========== 配置路径 ==========
model_dir = "/home/pengy1/RME/llama"  # 这里包含 config.json + pytorch_model.bin
input_path = "/home/pengy1/RME/data0/llama_input/prompt_sequence.npy"
save_path = "/home/pengy1/RME/data0/prediction/B25_Ant1_f1_S47_pred_tokens.npy"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# ========== 加载模型 ==========
print("Loading InternLM model (HuggingFace-style)...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float32,
    trust_remote_code=True
).eval()
print("Model loaded successfully.")

# ========== 读取 token 输入 ==========
full_seq = np.load(input_path)  # shape: [5, H, W]
flat_seq = full_seq.reshape(full_seq.shape[0], -1)  # [5, T]
prompt_list = flat_seq[:4]  # 前4组作为 prompt（每组 T token）
query_T = flat_seq[4]       # query input tokens

input_ids = np.concatenate(prompt_list).astype(np.int64)
input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

print(f"Total prompt token length: {input_ids.shape[1]}")

# ========== 推理生成 ==========
max_new = query_T.shape[0]  # 期望输出 token 数量（=H*W）
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
generated = outputs[0, -max_new:].cpu().numpy()

# ========== 保存结果 ==========
H = W = int(np.sqrt(max_new))
gen_map = generated.reshape(H, W)
np.save(save_path, gen_map)
print(f"Prediction saved to: {save_path}")
