import os
import re
import matplotlib.pyplot as plt

folder = "mmlu_results/masks_mask128"

subjects = []
accuracies = []
model_name_key = None
mask_length_val = None

for filename in os.listdir(folder):
    if not filename.endswith(".txt"):
        continue

    path = os.path.join(folder, filename)
    with open(path, "r") as f:
        content = f.read()

    # Extract field name (subject)
    m_field = re.search(r"field:\s*(\S+)", content)
    if not m_field:
        continue
    subject = m_field.group(1).strip()

    # Extract final accuracy
    m_acc = re.search(r"Accuracy:\s*([\d.]+)%", content)
    if not m_acc:
        continue
    acc = float(m_acc.group(1))

    # Extract model name once
    if model_name_key is None:
        m_model = re.search(r"model_name_or_path:\s*(\S+)", content)
        if m_model:
            model_path = m_model.group(1).strip()
            model_name_key = os.path.basename(os.path.dirname(model_path))

    # Extract mask_length once
    if mask_length_val is None:
        m_mask = re.search(r"mask_length:\s*(\d+)", content)
        if m_mask:
            mask_length_val = int(m_mask.group(1))

    subjects.append(subject)
    accuracies.append(acc)

if not subjects:
    raise ValueError("No valid result files found in the folder.")

# Sort by accuracy
sorted_pairs = sorted(zip(subjects, accuracies), key=lambda x: x[1], reverse=True)
subjects, accuracies = zip(*sorted_pairs)

# Plot histogram
plt.figure(figsize=(14, 6))
bars = plt.bar(range(len(subjects)), accuracies, color="skyblue", edgecolor="black")
plt.xticks(range(len(subjects)), subjects, rotation=75, ha="right", fontsize=9)
plt.ylabel("Accuracy (%)", fontsize=12)

title = f"MMLU Subject-wise Performance — {model_name_key} (mask={mask_length_val})"
plt.title(title, fontsize=14, pad=10)
plt.ylim(0, 100)

# Add value labels
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, acc + 1, f"{acc:.1f}",
             ha='center', va='bottom', fontsize=8)

plt.tight_layout()

# ✅ Save path includes model name + mask length
save_name = f"summary_hist_{model_name_key}_mask{mask_length_val}.png"
save_path = os.path.join(folder, save_name)
plt.savefig(save_path, dpi=150)
plt.close()

print(f"✅ Histogram saved to: {save_path}")
