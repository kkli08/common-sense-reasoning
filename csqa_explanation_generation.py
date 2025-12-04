import json
import csv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import traceback
import time
import sys

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

output_file = open("csqa_full.jsonl", "w")
csv_file = open("csqa_full.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["question", "choices", "answer", "short_explanation"])

error_log = open("error_log.txt", "w")


# ---------------------------------------------------------
# 5. Full generation loop with progress, batching, OOM recovery
# ---------------------------------------------------------
BATCH_SIZE = 50
flush_interval = 100

for idx in tqdm(range(TOTAL), desc="Generating explanations"):
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
        if "out of memory" in str(e).lower():
            print("\n⚠ OOM Detected! Clearing cache and reloading model...")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            time.sleep(2)
            tokenizer, model = load_model()
            continue

        # Other errors log only
        error_log.write(f"Error at index {idx}:\n{traceback.format_exc()}\n\n")
        print(f"\n[Error at {idx}] logged to error_log.txt")
        continue


output_file.close()
csv_file.close()
error_log.close()

print("\n🎉 DONE! Generated csqa_full.jsonl and csqa_full.csv successfully!")
