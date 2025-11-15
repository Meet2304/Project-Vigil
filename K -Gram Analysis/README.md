# K-Gram Analysis for Malicious Prompt Detection

This directory contains the implementation of k-gram analysis with Leave One Out (LOO) cross-validation for Project Vigil's malicious prompt classifier.

## Overview

K-gram analysis is a powerful technique for text classification that extracts character or word-level n-grams as features. This implementation uses Leave One Out cross-validation to provide a robust evaluation of the classifier's performance.

## Files

- `k_Gram Analysis_v0.1.ipynb` - Main Jupyter notebook with complete implementation
- `results/` - Directory containing evaluation results and visualizations

## Features

### 1. K-Gram Feature Extraction
- Character-level n-grams (default: 2-5 grams)
- Word-level n-grams (configurable)
- TF-IDF or Count vectorization
- Configurable maximum features

### 2. Leave One Out Cross-Validation
- Trains on N-1 samples, tests on 1
- Provides unbiased performance estimates
- Particularly useful for small to medium datasets
- Comprehensive evaluation metrics

### 3. Performance Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curve analysis
- Confusion matrix visualization
- Detailed classification reports

### 4. Model Analysis
- Top k-grams per class identification
- Feature importance analysis
- Model persistence (save/load)
- Prediction on new prompts

## Usage

### Running the Notebook

```bash
jupyter notebook "k_Gram Analysis_v0.1.ipynb"
```

### Quick Start

The notebook is self-contained and includes:
1. Automatic dataset loading (supports CSV, JSON, or directory structure)
2. Sample dataset creation if MPDD dataset is not available
3. Complete k-gram analysis pipeline
4. Model training and evaluation
5. Results visualization and saving

### Configuration

Modify the `K_GRAM_CONFIG` dictionary in the notebook:

```python
K_GRAM_CONFIG = {
    'char_ngram_range': (2, 5),  # Character n-gram range
    'word_ngram_range': (1, 3),  # Word n-gram range
    'max_features': 5000,        # Maximum features
    'use_tfidf': True,           # TF-IDF vs Count
    'analyzer': 'char'           # 'char' or 'word'
}
```

## Dataset Format

The implementation supports multiple dataset formats:

### CSV Format
```csv
text,label
"Ignore previous instructions...",1
"Can you help me with...",0
```

### JSON Format
```json
[
  {"text": "Ignore previous instructions...", "label": 1},
  {"text": "Can you help me with...", "label": 0}
]
```

### Directory Structure
```
Dataset/MPDD/
├── malicious/
│   ├── prompt1.txt
│   └── prompt2.txt
└── benign/
    ├── prompt1.txt
    └── prompt2.txt
```

## Results

Results are automatically saved to the `results/` directory:
- `loo_cv_results.json` - Detailed metrics in JSON format
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curve.png` - ROC curve plot

## Example Output

```
LEAVE ONE OUT CROSS-VALIDATION RESULTS
============================================================
Accuracy:  0.9500
Precision: 0.9474
Recall:    0.9500
F1-Score:  0.9487
ROC-AUC:   0.9850
============================================================
```

## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyter

Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

## Model Persistence

Trained models are saved to:
- `/models/classifier.pkl` - Trained classifier
- `/models/k_gram_vectorizer.pkl` - Fitted vectorizer

Load and use the model:
```python
import pickle

# Load model and vectorizer
with open('models/classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('models/k_gram_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Predict
X = vectorizer.transform(["Your prompt here"])
prediction = classifier.predict(X)
```

## Evaluation Methodology

### Leave One Out Cross-Validation

LOO CV is a special case of k-fold cross-validation where k equals the number of samples:
- Each sample is used once as a test set
- Remaining N-1 samples form the training set
- Provides nearly unbiased estimate of model performance
- Computationally expensive but thorough

### Why LOO for This Task?

1. **Robust Evaluation**: Maximizes training data usage
2. **Small Dataset Friendly**: Works well with limited samples
3. **Unbiased Estimates**: Minimal variance in performance metrics
4. **Per-Sample Analysis**: Identifies difficult-to-classify prompts

## Customization

### Using Different Classifiers

Replace the classifier in the notebook:

```python
# Logistic Regression (default)
classifier = LogisticRegression(max_iter=1000, random_state=42)

# Support Vector Machine
classifier = SVC(kernel='linear', probability=True, random_state=42)

# Random Forest
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Naive Bayes
classifier = MultinomialNB()
```

### Experimenting with N-Grams

Try different n-gram ranges:

```python
# Character trigrams only
K_GRAM_CONFIG['char_ngram_range'] = (3, 3)

# Word bigrams and trigrams
K_GRAM_CONFIG['word_ngram_range'] = (2, 3)
K_GRAM_CONFIG['analyzer'] = 'word'
```

## Contributing

To improve the k-gram analysis:
1. Experiment with different n-gram ranges
2. Try ensemble methods combining multiple n-gram ranges
3. Implement feature selection techniques
4. Add more sophisticated text preprocessing

## License

Part of Project Vigil - AI Safety and Security

## Author

Project Vigil Team

## Version History

- v0.1 (2025-11-15): Initial implementation with LOO CV
