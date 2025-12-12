# FLAN-T5 LoRA Fine-tuning on CommonsenseQA (cumming branch)

![Python](https://img.shields.io/badge/Python-3.12.3-blue)
![HuggingFace](https://img.shields.io/badge/Transformers-4.57.3-yellow)
![LoRA](https://img.shields.io/badge/PEFT-LoRA-green)


This branch contains my individual contribution to the project.  
I fine-tuned **FLAN-T5-Base** and **FLAN-T5-Large** on a CommonsenseQA-style dataset using LoRA, and provide:

- LoRA checkpoints for both models (Base & Large)
- A training / evaluation notebook (`FLAN-T5.ipynb`)
- A small inference script that compares both models on the same question

The environment and dataset follow the settings of the main repository  
(see the root-level `README.md` and `requirements.txt`).

---

## Directory Structure

The files in this branch are organized as follows:

```text
cumming/
├─ flan_t5_base_csqa_lora_v2/
│  └─ checkpoint-1461/
│     ├─ adapter_model.safetensors   # LoRA weights for FLAN-T5-Base
│     └─ adapter_config.json         # LoRA configuration
│
├─ flan_t5_large_csqa_lora_v1/
│  └─ checkpoint-974/
│     ├─ adapter_model.safetensors   # LoRA weights for FLAN-T5-Large
│     └─ adapter_config.json         # LoRA configuration
│
├─ FLAN-T5.ipynb                     # Notebook: training + evaluation + examples
└─ README.md                         # This file
```
The checkpoints only contain LoRA adapter weights.  
The base models (`google/flan-t5-base` and `google/flan-t5-large`) and tokenizers are downloaded automatically from Hugging Face.

---
## How to Run This Code
### 1. Clone the repo and switch branch

```bash
git clone https://github.com/kkli08/common-sense-reasoning.git
cd common-sense-reasoning
git checkout cumming
```

### 2. Install dependencies

You may reuse the group environment or create a new one:
pip install -r requirements.txt
pip install peft
Verify installation:
```bash
python -c "import torch, transformers, peft; print('OK')"
```
---

## Notebook (`FLAN-T5.ipynb`)

The notebook includes:

1. Loading the CommonsenseQA-style dataset (with choices + answer letters)
2. Building the T5 input format (`question + choices + explain your answer`)
3. LoRA fine-tuning for both FLAN-T5-Base and FLAN-T5-Large
4. Saving training logs and loss curves
5. Validation accuracy on answer letters
6. Sample outputs for qualitative analysis

Open the notebook and run all cells from top to bottom to reproduce the experiments.

---

## Inference Script (used in the report)

The last cell of `FLAN-T5.ipynb` contains a small inference script that loads the two LoRA-tuned models and compares their outputs on the same CommonsenseQA-style question.  
For convenience, here is the standalone version:
<details>
<summary>Click to view full runnable inference script</summary>

```python
import torch
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ===========================================================
# 1. Base models and LoRA checkpoints (paths are relative to this repo)
# ===========================================================
CONFIGS = {
    "flan_t5_base": {
        "base_model_name": "google/flan-t5-base",
        "lora_path": "cumming/flan_t5_base_csqa_lora_v2/checkpoint-1461",
    },
    "flan_t5_large": {
        "base_model_name": "google/flan-t5-large",
        "lora_path": "cumming/flan_t5_large_csqa_lora_v1/checkpoint-974",
    },
}

# ===========================================================
# 2. Utilities: build input text and parse answer letters
# ===========================================================
LETTERS = ["A", "B", "C", "D", "E", "F"]

def build_input(question, choices):
    choice_str = "; ".join(f"{LETTERS[i]}: {choices[i]}" for i in range(len(choices)))
    return (
        f"question: {question}\n"
        f"choices: {choice_str}\n"
        f"explain your answer:"
    )

def extract_answer_letter(text):
    t = text.lower()
    # e.g. "the answer is C", "answer: C", "answer C"
    m = re.search(r"(?:the\s+)?answer(?:\s+is|[: ]+)\s*([a-f])", t)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"answer[: ]*\s*([a-f])", t)
    if m2:
        return m2.group(1).upper()
    return None

# ===========================================================
# 3. Model loading and single-model run
# ===========================================================
def load_model(base_model_name, lora_path):
    print(f"\nLoading base model: {base_model_name}")
    base = T5ForConditionalGeneration.from_pretrained(base_model_name).to(device)
    print(f"Loading LoRA from: {lora_path}")
    model = PeftModel.from_pretrained(base, lora_path).to(device)
    tok = T5Tokenizer.from_pretrained(base_model_name)
    model.eval()
    return model, tok

def run_one_model(model_name, model, tokenizer, question, choices):
    input_text = build_input(question, choices)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=96,
            num_beams=4,
            do_sample=False,
        )

    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    letter = extract_answer_letter(decoded)

    print("\n" + "=" * 80)
    print(f"Model: {model_name}")
    print("- Input:")
    print(input_text)
    print("\n- Output:")
    print(decoded)
    print(f"\n- Parsed answer letter: {letter}")
    print("=" * 80)


# ===========================================================
# 4. Run both models on the same question
# ===========================================================
if __name__ == "__main__":
    # You can change this part to test your own question and choices.
    question = (
        "Why would a person place a metal spoon in the neck of an opened "
        "champagne bottle before putting it back in the refrigerator?"
    )
    choices = [
        "To measure the temperature of the drink",
        "To slow down the loss of carbonation",
        "To prevent the bottle from freezing",
        "To make the champagne taste sweeter",
        "To stop the bottle from tipping over",
    ]

    # Load both models
    models = {}
    for name, cfg in CONFIGS.items():
        model, tok = load_model(cfg["base_model_name"], cfg["lora_path"])
        models[name] = (model, tok)

    # Run them one by one
    for name, (model, tok) in models.items():
        run_one_model(name, model, tok, question, choices)
```

</details>

To run inference:

1. Install the environment  
2. Load both models using the paths above  
3. Set your own `question` and `choices`  
4. Run the two models and compare their outputs

---

## Minimum Requirements to Reproduce My Results

You will need:

- This branch (`cumming`)
- Hugging Face models:
  - `google/flan-t5-base`
  - `google/flan-t5-large`
- Python packages:
  - `torch`
  - `transformers`
  - `peft`

The LoRA weights and inference examples are fully included in this directory.

