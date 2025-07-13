import os
import matplotlib.pyplot as plt
import numpy as np

token_freq_file = ".../RME/data1/tokens/token_frequency.txt"
output_dir = ".../RME/data1/tokens"
os.makedirs(output_dir, exist_ok=True)

frequencies = []
with open(token_freq_file, "r") as f:
    for line in f:
        if line.startswith("Token ID") or line.strip() == "":
            continue
        parts = line.strip().split()
        if len(parts) == 2:
            _, freq = parts
            frequencies.append(int(freq))

frequencies = sorted(frequencies, reverse=True)
num_tokens = len(frequencies)

cumulative = np.cumsum(frequencies)
cumulative_percent = cumulative / cumulative[-1] * 100

top_k_values = [10, 50, 100, 200, 500, 1000]
top_k_results = {k: cumulative[k-1] / cumulative[-1] * 100 for k in top_k_values if k <= num_tokens}

txt_output_path = os.path.join(output_dir, "token_topk_summary.txt")
with open(txt_output_path, "w") as f:
    f.write(f"Total unique tokens: {num_tokens}\n\n")
    for k in top_k_results:
        f.write(f"Top {k} tokens cover: {top_k_results[k]:.2f}% of total token usage\n")

log_hist_output = os.path.join(output_dir, "token_frequency_log_hist.pdf")
plt.figure(figsize=(12, 4))
plt.bar(range(num_tokens), frequencies)
plt.yscale('log')
plt.xlabel("Sorted Token Index")
plt.ylabel("Frequency (log scale)")
plt.title("Token Usage Frequency (Log Scale)")
plt.tight_layout()
plt.savefig(log_hist_output)
plt.close()

cdf_output = os.path.join(output_dir, "token_usage_cdf.pdf")
plt.figure(figsize=(12, 4))
plt.plot(range(num_tokens), cumulative_percent)
plt.xlabel("Sorted Token Index")
plt.ylabel("Cumulative Percentage (%)")
plt.title("Cumulative Token Usage Distribution (CDF)")
plt.grid(True)
plt.tight_layout()
plt.savefig(cdf_output)
plt.close()

print("Analysis completed.")
print(f"Top-k summary saved to: {txt_output_path}")
print(f"Log-scale histogram saved to: {log_hist_output}")
print(f"CDF curve saved to: {cdf_output}")
