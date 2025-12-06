# CSQA Explanation Generation Fine-tuning (T5 + LoRA)

This project aims to train a T5 based model to perform *common-sense multiple-choice reasoning* **and generate human-like explanations**.  
We use the **CoS-E (Common Sense Explanations)** dataset, which provides:

- A question
- Multiple-choice answer options
- The correct answer
- A human-written rationale (explanation)


# 🧠 CSQA Fine-Tuning Evaluation

## **T5-Small (No Fine-Tuning) vs T5-Large + LoRA (Fine-Tuned)**

We compare both models on the same 8 baseline CSQA questions.

---

# 📊 **Accuracy Summary**

| Model                          | Accuracy | Correct / Total |
| ------------------------------ | -------- | --------------- |
| **T5-Small (Baseline)**        | **0%**   | 0 / 8           |
| **T5-Large + LoRA Fine-Tuned** | **100%** | 8 / 8           |

Fine-tuning transformed model performance **from completely failing → perfect accuracy**.

---

# 📘 **Side-by-Side Comparison of Model Outputs**

Below, each question shows outputs from **both models**, allowing direct comparison.

---

## ### **Question 1**

**Q:** Sammy wanted to go to where the people were. Where might he go?
**Choices:** race track, populated areas, the desert, apartment, roadblock
**Gold:** **B**

| Model               | Output                                                                                                    |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| **T5-Small**        | **answer: E.** Because Sammy wanted to go to where the people were. <br>❌ Predicted: **E**                |
| **T5-Large + LoRA** | **answer: B.** Because "populated areas" best fits Sammy's desire to visit populated areas. <br>✔ Correct |

---

## ### **Question 2**

**Q:** Where do you store fresh vegetables?
**Choices:** garage, refrigerator, bookshelf, bathroom, attic
**Gold:** **B**

| Model               | Output                                                                                          |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| **T5-Small**        | **answer: E.** Because refrigerators are used to store fresh vegetables. <br>❌ Predicted: **E** |
| **T5-Large + LoRA** | **answer: B.** Because refrigerators are ideal for storing fresh vegetables. <br>✔ Correct      |

---

## ### **Question 3**

**Q:** If you heat water to 100°C, what will happen?
**Choices:** freeze, boil, rust, glow, evaporate
**Gold:** **B**

| Model               | Output                                                                                  |
| ------------------- | --------------------------------------------------------------------------------------- |
| **T5-Small**        | **answer: E.** Because heat water to 100°C will freeze and rust. <br>❌ Predicted: **E** |
| **T5-Large + LoRA** | **answer: B.** Because boiling occurs when water reaches 100°C. <br>✔ Correct           |

---

## ### **Question 4**

**Q:** What do people usually use to dry their hands after washing?
**Choices:** towel, hammer, blanket, pillow, shoe
**Gold:** **A**

| Model               | Output                                                                                    |
| ------------------- | ----------------------------------------------------------------------------------------- |
| **T5-Small**        | **answer: E.** Because a towel is used to dry hands after washing. <br>❌ Predicted: **E** |
| **T5-Large + LoRA** | **answer: A.** Because a towel is commonly used to dry hands after washing. <br>✔ Correct |

---

## ### **Question 5**

**Q:** Where would you typically find books to read?
**Choices:** library, swimming pool, factory, garage, freeway
**Gold:** **A**

| Model               | Output                                                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **T5-Small**        | **answer: D.** Because libraries are the most popular places to read books. <br>❌ Predicted: **D**                     |
| **T5-Large + LoRA** | **answer: A.** Because libraries offer a wide variety of books for reading, making them the best choice. <br>✔ Correct |

---

## ### **Question 6**

**Q:** If someone wants to relax and reduce stress, what might they do?
**Choices:** meditate, argue, shout, work more, run into danger
**Gold:** **A**

| Model               | Output                                                                                                |
| ------------------- | ----------------------------------------------------------------------------------------------------- |
| **T5-Small**        | **answer: D.** Because "work more" is the best way to relax and reduce stress. <br>❌ Predicted: **D** |
| **T5-Large + LoRA** | **answer: A.** Because meditation is the most effective way to relax and reduce stress. <br>✔ Correct |

---

## ### **Question 7**

**Q:** What tool is commonly used to tighten screws?
**Choices:** screwdriver, spoon, pencil, comb, fork
**Gold:** **A**

| Model               | Output                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| **T5-Small**        | **answer: D.** Because a fork is commonly used to tighten screws. <br>❌ Predicted: **D**          |
| **T5-Large + LoRA** | **answer: A.** Because a screwdriver is the most common tool for tightening screws. <br>✔ Correct |

---

## ### **Question 8**

**Q:** Where would you likely find many wild animals living together?
**Choices:** forest, kitchen, bathroom, rooftop, office
**Gold:** **A**

| Model               | Output                                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **T5-Small**        | **answer: D.** Because "office" is the most popular place for wild animals living together. <br>❌ Predicted: **D** |
| **T5-Large + LoRA** | **answer: A.** Because forests are ideal habitats for wild animals to live together. <br>✔ Correct                 |

---
