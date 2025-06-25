# AI Tools Assignment: Mastering the AI Toolkit 

## Overview

This project demonstrates practical and theoretical mastery of leading AI tools and frameworks through a series of hands-on tasks and critical analysis. The assignment is structured into three main parts:

1. **Theoretical Understanding**: Short answer and comparative analysis of AI tools.
2. **Practical Implementation**: Real-world tasks using Scikit-learn, TensorFlow, and spaCy.
3. **Ethics & Optimization**: Bias identification, mitigation, and debugging.

---

## Group Members

- Member 1: Levis Rabah
- Member 2: Gloriah	Ruitii
- Member 3: Olaitan	Popoola
- Member 4: Timilehin	Onatunde
- Member 5: Hassan Ahmad	Tijani

## Table of Contents

- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Part 1: Theoretical Understanding](#part-1-theoretical-understanding)
- [Part 2: Practical Implementation](#part-2-practical-implementation)
  - [Task 1: Classical ML with Scikit-learn](#task-1-classical-ml-with-scikit-learn)
  - [Task 2: Deep Learning with TensorFlow](#task-2-deep-learning-with-tensorflow)
  - [Task 3: NLP with spaCy](#task-3-nlp-with-spacy)
- [Part 3: Ethics & Optimization](#part-3-ethics--optimization)
- [References](#references)

---

## Project Structure

```
.
├── bias_mitigation.py
├── debug_tensorflow_script.py
├── iris_classifier.py
├── mnist_cnn.py
├── nlp_spacy_analysis.py
├── requirements.txt
└── README.md
```

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/levisrabah/AI-Tools-Assignment.git
   cd AI-Tools-Assignment
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy English model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## Part 1: Theoretical Understanding

### 1. Short Answer Questions

**Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?**

- **TensorFlow** uses static computation graphs (define-then-run), is widely used in production, and has strong support for deployment (e.g., TensorFlow Lite, TensorFlow Serving).
- **PyTorch** uses dynamic computation graphs (define-by-run), is more Pythonic, and is popular in research for its flexibility and ease of debugging.
- **Choice**: Use TensorFlow for production and deployment; use PyTorch for rapid prototyping and research.

**Q2: Describe two use cases for Jupyter Notebooks in AI development.**

1. **Interactive Data Exploration**: Visualize and preprocess data step-by-step.
2. **Experiment Tracking**: Document code, results, and visualizations in a single, shareable document.

**Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?**

- spaCy provides pre-trained models for tokenization, part-of-speech tagging, named entity recognition, and dependency parsing, enabling robust and efficient NLP pipelines beyond simple string manipulation.

### 2. Comparative Analysis

| Feature            | Scikit-learn                | TensorFlow                |
|--------------------|-----------------------------|---------------------------|
| Target Applications| Classical ML (SVM, trees)   | Deep Learning (NNs, CNNs) |
| Ease of Use        | Beginner-friendly, simple API| Steeper learning curve    |
| Community Support  | Large, mature, stable       | Large, fast-evolving      |

---

## Part 2: Practical Implementation

### Task 1: Classical ML with Scikit-learn

- **File**: `iris_classifier.py`
- **Dataset**: Iris Species
- **Steps**:
  - Data loading and preprocessing
  - Decision tree training
  - Evaluation (accuracy, precision, recall)
  - Visualization of feature importance and confusion matrix

### Task 2: Deep Learning with TensorFlow

- **File**: `mnist_cnn.py`
- **Dataset**: MNIST Handwritten Digits
- **Steps**:
  - Build and train a CNN model
  - Achieve >95% test accuracy
  - Visualize predictions and training history

### Task 3: NLP with spaCy

- **File**: `nlp_spacy_analysis.py`
- **Dataset**: Sample Amazon Product Reviews
- **Steps**:
  - Named Entity Recognition (NER) to extract brands/products
  - Sentiment analysis using TextBlob
  - Visualization of sentiment and entity distributions

---

## Part 3: Ethics & Optimization

- **Bias Mitigation**: See `bias_mitigation.py` for a checklist and strategies using tools like TensorFlow Fairness Indicators and spaCy's rule-based systems.
- **Debugging Challenge**: See `debug_tensorflow_script.py` for common TensorFlow bugs and their fixes, with explanations.

---

## How to Run

- Run each script individually:
  ```bash
  python iris_classifier.py
  python mnist_cnn.py
  python nlp_spacy_analysis.py
  python bias_mitigation.py
  python debug_tensorflow_script.py
  ```

---

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [PyTorch Documentation](https://pytorch.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [spaCy Documentation](https://spacy.io/)
- [TextBlob Documentation](https://textblob.readthedocs.io/en/dev/)

---

## Notes

- For screenshots and detailed outputs, see the accompanying report PDF.
- For the group presentation, refer to the video link shared on the Community platform.

---

# Report

## 1. Theoretical Understanding

### Q1: TensorFlow vs. PyTorch

TensorFlow is production-oriented, supports static computation graphs, and is widely adopted in industry. PyTorch is research-oriented, uses dynamic graphs, and is preferred for rapid prototyping and debugging. Choose TensorFlow for deployment and scalability; choose PyTorch for flexibility and experimentation.

### Q2: Jupyter Notebook Use Cases

- **Data Exploration**: Enables stepwise data cleaning, visualization, and feature engineering.
- **Experiment Documentation**: Combines code, results, and explanations for reproducibility and sharing.

### Q3: spaCy vs. String Operations

spaCy provides advanced NLP features (NER, POS tagging, dependency parsing) out-of-the-box, while basic string operations are limited to simple manipulations (split, replace). spaCy is faster, more accurate, and easier for complex NLP tasks.

### Comparative Analysis: Scikit-learn vs. TensorFlow

- **Target Applications**: Scikit-learn excels at classical ML (regression, classification, clustering). TensorFlow is designed for deep learning (neural networks, large-scale models).
- **Ease of Use**: Scikit-learn is easier for beginners due to its simple API. TensorFlow requires understanding of computational graphs and is more complex.
- **Community Support**: Both have large communities, but TensorFlow evolves faster due to its focus on deep learning.

---

## 2. Practical Implementation

### Task 1: Iris Classifier

- **Approach**: Loaded and cleaned the Iris dataset, trained a decision tree, and evaluated performance.
- **Results**: Achieved high accuracy, precision, and recall. Visualizations show feature importance and confusion matrix.

### Task 2: MNIST CNN

- **Approach**: Built a CNN using TensorFlow/Keras, trained on MNIST, and visualized predictions.
- **Results**: Achieved >95% test accuracy. Sample predictions and training curves included.

### Task 3: NLP with spaCy

- **Approach**: Used spaCy for NER and TextBlob for sentiment analysis on Amazon reviews.
- **Results**: Extracted brands/products and sentiment. Visualizations show sentiment distribution and entity types.

---

## 3. Ethics & Optimization

### Bias Identification & Mitigation

- **Potential Biases**: MNIST may have digit style bias; Amazon reviews may reflect brand or demographic bias.
- **Mitigation Tools**: TensorFlow Fairness Indicators can audit model fairness; spaCy's rule-based systems can help detect biased language.
- **Checklist**: See `bias_mitigation.py` for a comprehensive fairness implementation guide.

### Debugging Challenge

- **Common Issues**: Dimension mismatches, incorrect loss functions, data preprocessing errors.
- **Fixes**: See `debug_tensorflow_script.py` for corrected code and explanations.

## Ethical Reflection

Ethical AI development requires continuous monitoring for bias, transparency in data and model decisions, and inclusion of affected communities in feedback loops. Tools like TensorFlow Fairness Indicators and spaCy's explainability features are essential for responsible AI.