import json
import csv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import traceback
import time
import os

# ---------------------------------------------------------
# 1. Load Qwen 7B model
# ---------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

def load_model():
    print("\n🧠 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("✔ Model loaded successfully.\n")
    return tokenizer, model

tokenizer, model = load_model()


# ---------------------------------------------------------
# 2. Generate LONG explanation
# ---------------------------------------------------------
def generate_long_explanation(question, choices, answer_letter):
    choices_text = "\n".join([f"{label}. {text}" for label, text in choices])

    prompt = f"""
You are a helpful AI tutor. Explain briefly in 2–4 sentences why the correct answer is {answer_letter}.

Question:
{question}

Choices:
{choices_text}

Correct answer: {answer_letter}

Explanation:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.3,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Explanation:" in decoded:
        decoded = decoded.split("Explanation:", 1)[1].strip()

    return decoded


# ---------------------------------------------------------
# 3. Short Explanation (LLM-regenerated)
# ---------------------------------------------------------
def generate_short_explanation(long_exp):
    prompt = f"""
Rewrite the following explanation into ONE short, clear sentence (10–15 words).
Rules:
- Must begin with "Because"
- Must NOT mention the question or choices
- Must be self-contained
- Do NOT add anything after the sentence

Explanation:
{long_exp}

Short sentence:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.2,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Short sentence:" in decoded:
        decoded = decoded.split("Short sentence:", 1)[1].strip()

    short = decoded.split(".")[0].strip() + "."

    if not short.lower().startswith("because"):
        short = "Because " + short.lstrip()

    return short


# ---------------------------------------------------------
# 4. Load dataset
# ---------------------------------------------------------
print("Loading CommonsenseQA...\n")
csqa = load_dataset("commonsense_qa")
train_ds = csqa["train"]   # 9741 samples
TOTAL = len(train_ds)


# ---------------------------------------------------------
# 5. Detect how many samples already generated (resume point)
# ---------------------------------------------------------
jsonl_path = "csqa_full.jsonl"
csv_path = "csqa_full.csv"

if os.path.exists(jsonl_path):
    with open(jsonl_path, "r") as f:
        existing_lines = [line for line in f.readlines() if line.strip()]
    START_INDEX = len(existing_lines)
else:
    START_INDEX = 0

print(f"➡ Detected {START_INDEX} existing samples in {jsonl_path} (will start from index {START_INDEX})")

# 这里假设你之前的 1~299 条已经写进 jsonl，那 START_INDEX = 299
# 新一轮会从 train_ds[299]（第 300 个样本）继续


# ---------------------------------------------------------
# 6. Open files in append mode
# ---------------------------------------------------------
output_file = open(jsonl_path, "a", encoding="utf-8")
csv_file = open(csv_path, "a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

# 如果是全新文件，需要写 header，这里只在文件不存在时写
if START_INDEX == 0 and os.path.getsize(csv_path) == 0:
    csv_writer.writerow(["question", "choices", "answer", "short_explanation"])

error_log = open("error_log.txt", "a", encoding="utf-8")

BATCH_SIZE = 50
flush_interval = 100


# ---------------------------------------------------------
# 7. Full generation loop (resume from START_INDEX)
# ---------------------------------------------------------
for idx in tqdm(range(START_INDEX, TOTAL),
                desc="Generating explanations",
                initial=START_INDEX,
                total=TOTAL):
    try:
        ex = train_ds[idx]
        q = ex["question"]
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        ans = ex["answerKey"]
        choices = list(zip(labels, texts))

        # Stage 1
        long_exp = generate_long_explanation(q, choices, ans)

        # Stage 2
        short_exp = generate_short_explanation(long_exp)

        record = {
            "question": q,
            "choices": texts,
            "answer": ans,
            "short_explanation": short_exp
        }

        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        csv_writer.writerow([
            q,
            json.dumps(texts, ensure_ascii=False),
            ans,
            short_exp
        ])

        # Batch notification
        if (idx + 1) % BATCH_SIZE == 0:
            print(f"\n[Batch {idx // BATCH_SIZE}] ✔ Processed {idx+1}/{TOTAL}")
            print(f" Last short explanation: {short_exp}")

        # Flush periodically
        if (idx + 1) % flush_interval == 0:
            output_file.flush()
            csv_file.flush()

    except RuntimeError as e:
        # 非 GPU 情况下这里基本不会触发，不过逻辑保留
        if "out of memory" in str(e).lower():
            print("\n⚠ OOM Detected! You may need to reduce batch size or restart.")
            # 如果你以后改成 GPU，可以在这里加 torch.cuda.empty_cache()
            error_log.write(f"OOM at index {idx}:\n{repr(e)}\n\n")
            continue

        error_log.write(f"Error at index {idx}:\n{traceback.format_exc()}\n\n")
        print(f"\n[Error at {idx}] logged to error_log.txt")
        continue


output_file.close()
csv_file.close()
error_log.close()

print("\n🎉 DONE (resume run)! csqa_full.jsonl and csqa_full.csv have been updated.")
