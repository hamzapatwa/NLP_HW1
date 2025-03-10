README.md
# Homework 1 – Bigram Language Models

This project implements language models (unigram, bigram MLE, and bigram with add-one smoothing) as part of Homework 1 for CSCI366/CSCI780.

## File Structure

nlp_hw1_project/ ├── README.md ├── nlp_hw1_solution.py └── data/ ├── train.txt └── test.txt


## Setup & Requirements

- **Python 3.x** is required.
- No external libraries are needed aside from the Python standard library.

## How to Run

1. Place your `train.txt` and `test.txt` files in the `data` directory.
2. Open a terminal and navigate to the project folder:
   ```bash
   cd path/to/nlp_hw1_project
Run the script:
python3 nlp_hw1_solution.py
The script will print out the computed answers for Q1–Q7 in the terminal.
What It Does

Pre-processing:
Lowercases and tokenizes each sentence.
Pads sentences with <s> (start) and </s> (end).
In the training data, replaces singleton tokens with <unk>.
In the test data, replaces any token not seen in training with <unk>.
Training:
Counts unigrams and bigrams to build three models:
Unigram (MLE)
Bigram (MLE)
Bigram with Add-One smoothing
Evaluation:
Computes:
Number of word types and tokens in training.
OOV percentages in the test set.
Percentages of unseen bigrams.
Log base-2 probabilities and perplexity for a sample sentence.
Overall perplexity on the test corpus.
Refer to the code comments in nlp_hw1_solution.py for more details.