# CSQA Explanation Generation Fine-tuning (T5 + LoRA)

This project aims to train a T5 based model to perform *common-sense multiple-choice reasoning* **and generate human-like explanations**.  
We use the **CoS-E (Common Sense Explanations)** dataset, which provides:

- A question
- Multiple-choice answer options
- The correct answer
- A human-written rationale (explanation)


# 🧠 CSQA Fine-Tuning Results (T5-Small Baseline vs T5-Large + LoRA)

This section summarizes the evaluation of our Commonsense Question Answering (CSQA) model before and after fine-tuning.
We compare:

* **T5-Small baseline (no fine-tuning)**
* **T5-Large + LoRA fine-tuned on cleaned CSQA dataset**

A set of 8 baseline questions was used to evaluate both models.

---

# 📊 Accuracy Comparison

| Model                          | Accuracy | Correct / Total |
| ------------------------------ | -------- | --------------- |
| **T5-Small (no fine-tuning)**  | **0%**   | 0 / 8           |
| **T5-Large + LoRA Fine-Tuned** | **100%** | 8 / 8           |

Fine-tuning yields a **dramatic improvement** from *0% → 100%* accuracy on the baseline evaluation set.

---

# 📘 Baseline Questions & Predictions

Below are the full model outputs, preserved exactly as generated.

---

## ## 🔹 T5-Small Baseline (No Fine-Tuning)

**Accuracy: 0 / 8 (0%)**

---

### **Question 1**

**Q:** Sammy wanted to go to where the people were. Where might he go?
**Choices:** race track, populated areas, the desert, apartment, roadblock

**Model Output:**

> answer: E. Because Sammy wanted to go to where the people were.

**Predicted:** E
**Gold:** B
❌ **Incorrect**

---

### **Question 2**

**Q:** Where do you store fresh vegetables?
**Choices:** garage, refrigerator, bookshelf, bathroom, attic

**Model Output:**

> answer: E. Because refrigerators are used to store fresh vegetables.

**Predicted:** E
**Gold:** B
❌ **Incorrect**

---

### **Question 3**

**Q:** If you heat water to 100 degrees Celsius, what will happen?
**Choices:** it will freeze, it will boil, it will rust, it will glow, it will evaporate

**Model Output:**

> answer: E. Because heat water to 100 degrees Celsius will freeze and rust.

**Predicted:** E
**Gold:** B
❌ **Incorrect**

---

### **Question 4**

**Q:** What do people usually use to dry their hands after washing?
**Choices:** towel, hammer, blanket, pillow, shoe

**Model Output:**

> answer: E. Because a towel is used to dry hands after washing.

**Predicted:** E
**Gold:** A
❌ **Incorrect**

---

### **Question 5**

**Q:** Where would you typically find books to read?
**Choices:** library, swimming pool, factory, garage, freeway

**Model Output:**

> answer: D. Because libraries are the most popular places to read books.

**Predicted:** D
**Gold:** A
❌ **Incorrect**

---

### **Question 6**

**Q:** If someone wants to relax and reduce stress, what might they do?
**Choices:** meditate, argue, shout, work more, run into danger

**Model Output:**

> answer: D. Because "work more" is the best way to relax and reduce stress.

**Predicted:** D
**Gold:** A
❌ **Incorrect**

---

### **Question 7**

**Q:** What tool is commonly used to tighten screws?
**Choices:** screwdriver, spoon, pencil, comb, fork

**Model Output:**

> answer: D. Because a fork is commonly used to tighten screws.

**Predicted:** D
**Gold:** A
❌ **Incorrect**

---

### **Question 8**

**Q:** Where would you likely find many wild animals living together?
**Choices:** forest, kitchen, bathroom, rooftop, office

**Model Output:**

> answer: D. Because "office" is the most popular place for wild animals living together.

**Predicted:** D
**Gold:** A
❌ **Incorrect**

---

---

# ## 🔹 T5-Large + LoRA Fine-Tuned Model

**Accuracy: 8 / 8 (100%)**

---

### **Question 1**

**Model Output:**

> answer: B. Because "populated areas" best fits Sammy's desire to visit populated areas.

**Predicted:** B
**Correct ✔**

---

### **Question 2**

**Model Output:**

> answer: B. Because refrigerators are ideal for storing fresh vegetables.

**Predicted:** B
**Correct ✔**

---

### **Question 3**

**Model Output:**

> answer: B. Because boiling occurs when water reaches 100 degrees Celsius.

**Predicted:** B
**Correct ✔**

---

### **Question 4**

**Model Output:**

> answer: A. Because a towel is commonly used to dry hands after washing.

**Predicted:** A
**Correct ✔**

---

### **Question 5**

**Model Output:**

> answer: A. Because libraries offer a wide variety of books for reading, making them the best choice.

**Predicted:** A
**Correct ✔**

---

### **Question 6**

**Model Output:**

> answer: A. Because meditation is the most effective way to relax and reduce stress.

**Predicted:** A
**Correct ✔**

---

### **Question 7**

**Model Output:**

> answer: A. Because a screwdriver is the most common tool for tightening screws.

**Predicted:** A
**Correct ✔**

---

### **Question 8**

**Model Output:**

> answer: A. Because forests are ideal habitats for wild animals to live together.

**Predicted:** A
**Correct ✔**