# CSQA Explanation Generation Fine-tuning (T5 + LoRA)

This project aims to train a T5 based model to perform *common-sense multiple-choice reasoning* **and generate human-like explanations**.  
We use the **Common Sense QA** dataset, which provides:

- A question
- Multiple-choice answer options
- The correct answer
- A human-written rationale (explanation)

## Project Structure
```bash
.
├── csqa_explanation_generation.py
│   # Main script for generating high-quality explanations for CSQA
│   # using a large language model (Here we use Qwen2.5 7B)

├── training_t5_CoSE.ipynb
│   # Jupyter notebook for training and fine-tuning T5 models
│   # (T5-Small baseline and T5-Large + LoRA)
│   # Includes:
│   # - data preprocessing & cleaning
│   # - tokenization
│   # - LoRA configuration
│   # - training & evaluation
│   # - baseline vs fine-tuned comparison

├── data/
│   ├── csqa_full.jsonl
│   │   # Cleaned CommonsenseQA-style dataset with
│   │   # question, choices, answer, and generated explanations
│   │   # (JSONL format, one example per line)
│   │
│   └── csqa_full.csv
│       # CSV version of the same dataset for analysis,
│       # visualization, or third-party tools

├── t5_csqa_lora/
│   # Output directory for LoRA fine-tuning checkpoints
│   ├── checkpoint-*/                # Intermediate training checkpoints
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── trainer_state.json
│   │
│   └── trainer_state.json            # Final trainer metadata

├── t5_csqa_lora_merged/
│   # Final merged T5 model (base T5 + LoRA weights)
│   # Fully loadable for inference
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json

└── README.md
    # Project overview, methodology, and results

```
## Setup Instruction
Follow the steps below to set up the environment and run the project.
### 1️⃣ Clone the repository

```bash
git clone https://github.com/kkli08/common-sense-reasoning.git
cd <your-repo-name>
```

---

### 2️⃣ Create and activate a virtual environment

We recommend using a Python virtual environment to avoid dependency conflicts.

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```


### 3️⃣ Install dependencies

All required packages are listed in `requirements.txt`.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ✅ **Note (macOS / Apple Silicon):**
> Training is supported via Metal Performance Shaders (MPS).
> No CUDA or `bitsandbytes` is required.

### 4️⃣ Verify installation (optional)

You can quickly verify that PyTorch and Transformers are installed correctly:

```bash
python - <<EOF
import torch
import transformers
print("Torch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("MPS available:", torch.backends.mps.is_available())
EOF
```

## Usage Instruction
### Dataset preparation

This project includes a data generation pipeline that creates **high-quality short reasoning explanations** for the CommonsenseQA dataset using a locally deployed large language model. [Why not CoSE?](https://github.com/kkli08/common-sense-reasoning/wiki)

The generation process uses a **two-stage reasoning approach**:
1. Generate a correct long explanation for each question
2. Compress it into a clean, single-sentence short explanation

Only the final **short explanations** are kept and saved for downstream model training.

Run the following command:

```bash
python3 csqa_explanation_generation.py
```
This will:
- Load the CommonsenseQA training set 
- Generate explanations incrementally with progress tracking 
- Automatically resume from partially generated files 
- Save results to:
  - `data/csqa_full.jsonl`
- `data/csqa_full.csv`

Each generated sample follows this format:
```json
{
    "question": "...",
    "choices": ["...", "...", "...", "...", "..."],
    "answer": "B",
    "short_explanation": "Because ..."
}
```

The dataset is expected to be located in:

```text
data/
├── csqa_full.jsonl
└── csqa_full.csv
```


### Model training

To train and evaluate the model:

```text
training_t5_CoSE.ipynb
```

The notebook covers:

* Data cleaning and formatting
* Tokenization for T5
* LoRA fine-tuning
* Baseline vs fine-tuned evaluation
* Model merging and inference

Run the notebook sequentially to reproduce results.

### Inference with the trained model
After training, the merged model will be saved under:

```text
t5_csqa_lora_merged/
```
You can load it directly using Hugging Face Transformers for inference.

## Sample output
TBD
## CSQA Fine-Tuning Evaluation

### **T5-Small (No Fine-Tuning) vs T5-Large + LoRA (Fine-Tuned)**

We compare both models on the same 8 baseline CSQA questions.

| Model                          | Accuracy | Correct / Total |
| ------------------------------ | -------- | --------------- |
| **T5-Small (Baseline)**        | **0%**   | 0 / 8           |
| **T5-Large + LoRA Fine-Tuned** | **100%** | 8 / 8           |

Fine-tuning transformed model performance **from completely failing → perfect accuracy**.

### **Side-by-Side Comparison of Model Outputs**

Below, each question shows outputs from **both models**, allowing direct comparison.

**Question 1**

**Q:** Sammy wanted to go to where the people were. Where might he go?


**Choices:** race track, populated areas, the desert, apartment, roadblock


**Gold:** **B**

| Model               | Output                                                                                                    |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| **T5-Small**        | **answer: E.** Because Sammy wanted to go to where the people were. <br>❌ Predicted: **E**                |
| **T5-Large + LoRA** | **answer: B.** Because "populated areas" best fits Sammy's desire to visit populated areas. <br>✔ Correct |

**Question 2**

**Q:** Where do you store fresh vegetables?


**Choices:** garage, refrigerator, bookshelf, bathroom, attic


**Gold:** **B**

| Model               | Output                                                                                          |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| **T5-Small**        | **answer: E.** Because refrigerators are used to store fresh vegetables. <br>❌ Predicted: **E** |
| **T5-Large + LoRA** | **answer: B.** Because refrigerators are ideal for storing fresh vegetables. <br>✔ Correct      |

**Question 3**

**Q:** If you heat water to 100°C, what will happen?


**Choices:** freeze, boil, rust, glow, evaporate


**Gold:** **B**

| Model               | Output                                                                                  |
| ------------------- | --------------------------------------------------------------------------------------- |
| **T5-Small**        | **answer: E.** Because heat water to 100°C will freeze and rust. <br>❌ Predicted: **E** |
| **T5-Large + LoRA** | **answer: B.** Because boiling occurs when water reaches 100°C. <br>✔ Correct           |

**Question 4**

**Q:** What do people usually use to dry their hands after washing?


**Choices:** towel, hammer, blanket, pillow, shoe


**Gold:** **A**

| Model               | Output                                                                                    |
| ------------------- | ----------------------------------------------------------------------------------------- |
| **T5-Small**        | **answer: E.** Because a towel is used to dry hands after washing. <br>❌ Predicted: **E** |
| **T5-Large + LoRA** | **answer: A.** Because a towel is commonly used to dry hands after washing. <br>✔ Correct |

**Question 5**

**Q:** Where would you typically find books to read?


**Choices:** library, swimming pool, factory, garage, freeway


**Gold:** **A**

| Model               | Output                                                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **T5-Small**        | **answer: D.** Because libraries are the most popular places to read books. <br>❌ Predicted: **D**                     |
| **T5-Large + LoRA** | **answer: A.** Because libraries offer a wide variety of books for reading, making them the best choice. <br>✔ Correct |

**Question 6**

**Q:** If someone wants to relax and reduce stress, what might they do?


**Choices:** meditate, argue, shout, work more, run into danger


**Gold:** **A**

| Model               | Output                                                                                                |
| ------------------- | ----------------------------------------------------------------------------------------------------- |
| **T5-Small**        | **answer: D.** Because "work more" is the best way to relax and reduce stress. <br>❌ Predicted: **D** |
| **T5-Large + LoRA** | **answer: A.** Because meditation is the most effective way to relax and reduce stress. <br>✔ Correct |

**Question 7**

**Q:** What tool is commonly used to tighten screws?


**Choices:** screwdriver, spoon, pencil, comb, fork


**Gold:** **A**

| Model               | Output                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| **T5-Small**        | **answer: D.** Because a fork is commonly used to tighten screws. <br>❌ Predicted: **D**          |
| **T5-Large + LoRA** | **answer: A.** Because a screwdriver is the most common tool for tightening screws. <br>✔ Correct |

**Question 8**

**Q:** Where would you likely find many wild animals living together?


**Choices:** forest, kitchen, bathroom, rooftop, office


**Gold:** **A**

| Model               | Output                                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **T5-Small**        | **answer: D.** Because "office" is the most popular place for wild animals living together. <br>❌ Predicted: **D** |
| **T5-Large + LoRA** | **answer: A.** Because forests are ideal habitats for wild animals to live together. <br>✔ Correct                 |

