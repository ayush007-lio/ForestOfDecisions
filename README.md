# 🌳 TreeScope-ML  
### Exploring 7 Variants of Decision Tree Algorithms

## 📌 Overview
This project demonstrates how a single Decision Tree algorithm can behave differently under various configurations.

Instead of using multiple algorithms, we experiment with **7 distinct decision tree variations** on the same dataset to analyze performance, overfitting, and generalization.

---

## 🎯 Objective
- Implement **7 types of Decision Tree models**
- Compare their performance on a real-world dataset
- Understand how parameters affect predictions

---

## 🌲 Models Implemented

1. Gini Index Tree  
2. Entropy (Information Gain) Tree  
3. Depth-Limited Tree  
4. Min Samples Split Tree  
5. Min Samples Leaf Tree  
6. Cost Complexity Pruned Tree  
7. Random Split Tree  

---

## 📊 Dataset
- Heart Disease Dataset  
- Real-world medical dataset for classification  

---

## ⚙️ Tech Stack
- Python  
- Scikit-learn  
- Pandas  
- Matplotlib  

---

## 📈 Output
- Accuracy comparison of all 7 models  
- Decision Tree visualization  
- Results saved as CSV  

---

## 🧠 Key Insight
Small changes in parameters can significantly affect model performance, making model tuning an essential part of machine learning.

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
cd src
python main.py
