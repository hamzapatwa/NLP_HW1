

# Homework 1 – Language Models

This project implements three language models—Unigram (MLE), Bigram (MLE), and Bigram with Add-One Smoothing—for evaluating a NewsCrawl corpus. It follows the processing and evaluation steps outlined in Homework 1 for CSCI366/CSCI780.

## File Structure

```
nlp_hw1_project/
├── README.md
├── nlp_hw1_solution.py
└── data/
    ├── train.txt
    └── test.txt
```

- **data/train.txt**: Training corpus (approximately 100k pre-tokenized sentences).
- **data/test.txt**: Test corpus (same domain as training).
- **nlp_hw1_solution.py**: Python script that preprocesses the data, trains the models, and computes evaluation metrics (Q1–Q7).

## Requirements

- Python 3.x

## How to Run

1. **Prepare Data**:  
   Place your `train.txt` and `test.txt` files in the `data/` directory.

2. **Run the Script**:  
   Open a terminal, navigate to the project folder, and run:
   ```bash
   python3 nlp_hw1_solution.py
   ```
   The script will output the evaluation results (including vocabulary counts, OOV percentages, log probabilities, and perplexities) as specified in the report.

## Overview

- **Pre-processing**:  
  - Lowercases and pads each sentence with `<s>` (start) and `</s>` (end).
  - Replaces singleton tokens in the training data with `<unk>`.
  - In the test data, tokens not present in the training vocabulary are mapped to `<unk>`.

- **Model Training**:  
  - **Unigram Model (MLE)**: Calculates the probability of each word.
  - **Bigram MLE Model**: Computes probabilities for adjacent word pairs.
  - **Bigram + Add-One Smoothing Model**: Applies add-one smoothing to handle unseen bigrams.

- **Evaluation**:  
  The script computes:
  - Vocabulary size and token counts (excluding `<s>`).
  - OOV token and type percentages in the test set (before mapping to `<unk>`).
  - Percentage of unseen bigrams in the test data.
  - Log base-2 probabilities and perplexities for a sample sentence.
  - Overall perplexity for the entire test corpus.

## Additional Notes

- The provided script includes detailed comments for clarity.
- Modify file paths or parameters as needed based on your setup or assignment requirements.

Happy coding!
