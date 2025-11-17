# Markov Chain Implementation for Malicious Prompt Detection

This folder contains a Jupyter notebook that implements a Markov Chain-based approach to identify word sequences responsible for malicious characteristics in prompts.

## Overview

The notebook uses k-grams (bigrams and trigrams) to build separate Markov Chains for malicious and benign prompts, then uses likelihood ratios to classify new prompts and identify the most discriminative word sequences.

## Files

- `Markov_Chain_Prompt_Analysis.ipynb` - Main implementation notebook

## How to Run

### On Google Colab (Recommended)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File` → `Upload notebook`
3. Upload `Markov_Chain_Prompt_Analysis.ipynb`
4. Run all cells sequentially (Runtime → Run all)

The notebook will automatically download the dataset from GitHub.

### Locally

1. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

2. Start Jupyter:
```bash
jupyter notebook
```

3. Open `Markov_Chain_Prompt_Analysis.ipynb`
4. Run all cells sequentially

## What the Notebook Does

### 1. Data Loading & Preprocessing
- Loads the MPDD dataset (malicious and benign prompts)
- Preprocesses text (lowercase, tokenization, cleaning)
- Splits into train/test sets

### 2. K-Gram Extraction
- Extracts bigrams (2-word sequences)
- Extracts trigrams (3-word sequences)

### 3. Markov Chain Training
- Builds separate Markov Chains for malicious and benign prompts
- Calculates transition probabilities: P(word_i | word_{i-1})
- Uses Laplace smoothing for unseen transitions

### 4. Classification
- Classifies prompts using likelihood ratios
- Compares P(sequence|malicious) vs P(sequence|benign)

### 5. Evaluation
- Reports accuracy, precision, recall, F1-score
- Shows confusion matrix
- Provides detailed classification report

### 6. Analysis
- Identifies top word sequences indicating maliciousness
- Shows bigrams and trigrams most discriminative
- Explains individual predictions

## Expected Results

The Markov Chain model typically achieves:
- **Accuracy**: 60-75% (depends on dataset)
- **Fast training**: < 5 seconds
- **Fast inference**: < 1ms per prompt
- **Interpretable results**: Shows which word sequences are malicious

## Key Features

### Advantages
✓ **Simple & Interpretable**: Shows exactly which word sequences contribute to maliciousness
✓ **Fast**: Training and inference are very quick
✓ **Lightweight**: Low memory requirements
✓ **Explainable**: Provides reasoning for each prediction

### Limitations
✗ Only captures sequential dependencies (not long-range context)
✗ Assumes Markov property (future depends only on current state)
✗ May struggle with novel attack patterns

## Example Outputs

### Top Malicious Word Sequences
The notebook identifies sequences like:
- "forget → previous" (high likelihood ratio)
- "ignore → instructions" (high likelihood ratio)
- "bypass → safety" (high likelihood ratio)

### Performance Metrics
- Confusion matrix showing true/false positives/negatives
- ROC curves and classification reports
- Per-class precision and recall

## Runtime

On Google Colab (free tier):
- **Total runtime**: ~30-60 seconds
- **Training**: ~2-5 seconds
- **Evaluation**: ~5-10 seconds
- **Visualization**: ~10-20 seconds

## Future Improvements

Potential enhancements:
1. Higher-order Markov chains (4-grams, 5-grams)
2. Combine with TF-IDF features
3. Use word embeddings instead of tokens
4. Ensemble with other classifiers
5. Implement variable-length n-grams

## Troubleshooting

### Issue: "No module named X"
**Solution**: Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Issue: Dataset not found (local execution)
**Solution**: Ensure you're in the project root directory, or modify the dataset path in cell #3

### Issue: Slow execution
**Solution**: The notebook is optimized for Colab. If running locally, consider reducing the dataset size or using a faster machine.

## Citation

If you use this implementation, please cite:
```
Project Vigil - Markov Chain Implementation for Malicious Prompt Detection
https://github.com/Meet2304/Project-Vigil
```

## License

This implementation is part of the Project Vigil repository.
