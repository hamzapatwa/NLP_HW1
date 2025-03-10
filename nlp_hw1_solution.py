import sys
from collections import defaultdict, Counter
import math
import os


def preprocess_line(line):
    """
    Lowercase, strip, and pad a line with <s> and </s>.
    Assumes the data is pre-tokenized.
    """
    line = line.strip().lower()
    tokens = line.split()
    tokens = ["<s>"] + tokens + ["</s>"]
    return tokens


def find_singletons(train_lines):
    """
    Count word frequencies and return a set of words that appear exactly once.
    """
    freq = Counter()
    for line in train_lines:
        for token in line:
            freq[token] += 1
    singletons = {w for w, count in freq.items() if count == 1}
    return singletons


def replace_singletons(train_lines, singletons):
    """
    Replace singletons with <unk> in the training data (except <s> and </s>).
    """
    for i in range(len(train_lines)):
        new_tokens = []
        for token in train_lines[i]:
            if token in singletons and token not in ("<s>", "</s>"):
                new_tokens.append("<unk>")
            else:
                new_tokens.append(token)
        train_lines[i] = new_tokens


def replace_unk_test(test_lines, train_vocab):
    """
    Replace any token in test data that is not in the training vocabulary with <unk>.
    """
    for i in range(len(test_lines)):
        new_tokens = []
        for token in test_lines[i]:
            if token not in train_vocab and token not in ("<s>", "</s>"):
                new_tokens.append("<unk>")
            else:
                new_tokens.append(token)
        test_lines[i] = new_tokens


class LM:
    """
    Language Model class for Unigram and Bigram models.
    """

    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.bigram_denominator = Counter()  # Counts for w_{i-1} in bigrams
        self.total_unigrams = 0  # Total tokens count (excluding <s> as per instructions)
        self.vocab = set()  # Vocabulary (excluding <s>)

    def train(self, train_lines):
        """
        Populate counts from training data.
        """
        for line in train_lines:
            for token in line:
                self.unigram_counts[token] += 1
            for i in range(len(line) - 1):
                w1, w2 = line[i], line[i + 1]
                self.bigram_counts[(w1, w2)] += 1
                self.bigram_denominator[w1] += 1

        # Total tokens for probability calculations (skip <s>)
        for token, count in self.unigram_counts.items():
            if token != "<s>":
                self.total_unigrams += count

        # Build vocab excluding <s> but including all other tokens seen.
        for token in self.unigram_counts:
            if token != "<s>":
                self.vocab.add(token)

    def unigram_prob(self, token):
        """
        MLE Unigram probability.
        """
        if self.total_unigrams == 0:
            return 0
        return self.unigram_counts[token] / float(self.total_unigrams)

    def bigram_prob_mle(self, w1, w2):
        """
        MLE Bigram probability.
        """
        if self.bigram_denominator[w1] == 0:
            return 0.0
        return self.bigram_counts[(w1, w2)] / float(self.bigram_denominator[w1])

    def bigram_prob_addone(self, w1, w2):
        """
        Bigram probability with add-one smoothing.
        """
        V = len(self.vocab)  # Vocabulary size
        return (self.bigram_counts[(w1, w2)] + 1) / float(self.bigram_denominator[w1] + V)

    def sent_logprob_unigram(self, sent):
        """
        Compute log base-2 probability of a sentence under the unigram model.
        Skips the <s> token in the product.
        """
        logp = 0.0
        for token in sent:
            if token == "<s>":
                continue
            p = self.unigram_prob(token)
            if p == 0.0:
                return float('-inf')
            logp += math.log2(p)
        return logp

    def sent_logprob_bigram_mle(self, sent):
        """
        Compute log base-2 probability of a sentence under the bigram MLE model.
        """
        logp = 0.0
        for i in range(len(sent) - 1):
            p = self.bigram_prob_mle(sent[i], sent[i + 1])
            if p == 0.0:
                return float('-inf')
            logp += math.log2(p)
        return logp

    def sent_logprob_bigram_addone(self, sent):
        """
        Compute log base-2 probability of a sentence under the bigram add-one model.
        """
        logp = 0.0
        for i in range(len(sent) - 1):
            p = self.bigram_prob_addone(sent[i], sent[i + 1])
            logp += math.log2(p)
        return logp

    def sent_perplexity(self, sent, model='unigram'):
        """
        Compute perplexity of a single sentence under the selected model.
        N is defined as the number of tokens excluding <s>.
        """
        N = max(len(sent) - 1, 1)  # Exclude <s>
        if model == 'unigram':
            lp = self.sent_logprob_unigram(sent)
        elif model == 'bigram':
            lp = self.sent_logprob_bigram_mle(sent)
        elif model == 'addone':
            lp = self.sent_logprob_bigram_addone(sent)
        else:
            raise ValueError("Unknown model type.")
        if lp == float('-inf'):
            return float('inf')
        return 2 ** (-lp / N)


def compute_corpus_perplexity(lm, sentences, model='unigram'):
    """
    Compute the perplexity of an entire corpus.
    """
    total_logprob = 0.0
    token_count = 0
    for sent in sentences:
        if model == 'unigram':
            lp = lm.sent_logprob_unigram(sent)
        elif model == 'bigram':
            lp = lm.sent_logprob_bigram_mle(sent)
        elif model == 'addone':
            lp = lm.sent_logprob_bigram_addone(sent)
        else:
            raise ValueError("Unknown model type.")
        # Count tokens (excluding <s>)
        token_count += (len(sent) - 1)
        total_logprob += lp
    if token_count == 0:
        return float('inf')
    return 2 ** (-total_logprob / token_count)


def main():
    # Set paths for train and test data.
    train_path = os.path.join("data", "train.txt")
    test_path = os.path.join("data", "test.txt")

    # 1. Preprocess Training Data
    train_lines_raw = []
    with open(train_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                tokens = preprocess_line(line)
                train_lines_raw.append(tokens)

    # Q1 & Q2: Before replacing singletons, we will later count types and tokens.
    # Replace singletons with <unk>
    singletons = find_singletons(train_lines_raw)
    replace_singletons(train_lines_raw, singletons)

    # 2. Train Language Model on Training Data
    lm = LM()
    lm.train(train_lines_raw)

    # Q1: Number of word types (unique tokens) in training (excluding <s>)
    num_word_types = len(lm.vocab)
    print("Q1: Number of word types (excluding <s>):", num_word_types)

    # Q2: Number of word tokens in training (excluding <s>)
    print("Q2: Number of word tokens (excluding <s>):", lm.total_unigrams)

    # 3. Preprocess Test Data (before mapping unknowns for Q3)
    test_lines_raw = []
    with open(test_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                tokens = preprocess_line(line)
                test_lines_raw.append(tokens)

    # Q3: OOV percentages before mapping to <unk>
    # Build training vocabulary from raw training data (before replacement, but excluding <s>)
    train_vocab_raw = set()
    for line in train_lines_raw:
        for token in line:
            if token != "<s>":
                train_vocab_raw.add(token)
    total_test_tokens = 0
    oov_tokens = 0
    test_types = set()
    oov_types = set()
    for sent in test_lines_raw:
        for token in sent:
            if token == "<s>":
                continue
            total_test_tokens += 1
            test_types.add(token)
            if token not in train_vocab_raw:
                oov_tokens += 1
                oov_types.add(token)
    oov_token_percentage = (oov_tokens / total_test_tokens) * 100 if total_test_tokens > 0 else 0
    oov_type_percentage = (len(oov_types) / len(test_types)) * 100 if test_types else 0
    print("Q3: OOV token percentage (before mapping): {:.2f}%".format(oov_token_percentage))
    print("Q3: OOV type percentage (before mapping): {:.2f}%".format(oov_type_percentage))

    # 4. Replace unknowns in Test Data based on training vocabulary (for bigram experiments)
    # Use training vocab from lm.vocab, but add <s> because it's used in test sentences.
    train_vocab = lm.vocab.union({"<s>"})
    replace_unk_test(test_lines_raw, train_vocab)

    # Q4: Compute unseen bigrams in test (after mapping to <unk>)
    train_bigram_types = set(lm.bigram_counts.keys())
    test_bigram_types = set()
    total_test_bigram_tokens = 0
    unseen_bigram_tokens = 0
    for sent in test_lines_raw:
        for i in range(len(sent) - 1):
            bigram = (sent[i], sent[i + 1])
            test_bigram_types.add(bigram)
            total_test_bigram_tokens += 1
            if bigram not in train_bigram_types:
                unseen_bigram_tokens += 1
    percent_unseen_bigram_types = (len({bg for bg in test_bigram_types if bg not in train_bigram_types}) / len(
        test_bigram_types)) * 100
    percent_unseen_bigram_tokens = (unseen_bigram_tokens / total_test_bigram_tokens) * 100
    print("Q4: Percentage of unseen bigram types in test: {:.2f}%".format(percent_unseen_bigram_types))
    print("Q4: Percentage of unseen bigram tokens in test: {:.2f}%".format(percent_unseen_bigram_tokens))

    # 5. Evaluate a sample sentence
    sample_sentence = "I look forward to hearing your reply ."
    sample_tokens = preprocess_line(sample_sentence)
    # Replace unknown tokens in the sample sentence
    sample_tokens = [token if token in train_vocab or token in ("<s>", "</s>") else "<unk>" for token in sample_tokens]

    unigram_logp = lm.sent_logprob_unigram(sample_tokens)
    bigram_mle_logp = lm.sent_logprob_bigram_mle(sample_tokens)
    bigram_addone_logp = lm.sent_logprob_bigram_addone(sample_tokens)
    print("\nQ5: Sample Sentence Log Probabilities (base 2):")
    print("Unigram:", unigram_logp)
    print("Bigram MLE:", bigram_mle_logp, "(if -∞ then at least one bigram is unseen)")
    print("Bigram + Add-One:", bigram_addone_logp)

    # 6. Compute Perplexity for the sample sentence
    pp_unigram = lm.sent_perplexity(sample_tokens, model='unigram')
    pp_bigram_mle = lm.sent_perplexity(sample_tokens, model='bigram')
    pp_bigram_addone = lm.sent_perplexity(sample_tokens, model='addone')
    print("\nQ6: Sample Sentence Perplexities:")
    print("Unigram:", pp_unigram)
    print("Bigram MLE:", pp_bigram_mle)
    print("Bigram + Add-One:", pp_bigram_addone)

    # 7. Compute Perplexity on the Entire Test Corpus
    corpus_pp_unigram = compute_corpus_perplexity(lm, test_lines_raw, model='unigram')
    corpus_pp_bigram_mle = compute_corpus_perplexity(lm, test_lines_raw, model='bigram')
    corpus_pp_bigram_addone = compute_corpus_perplexity(lm, test_lines_raw, model='addone')
    print("\nQ7: Test Corpus Perplexities:")
    print("Unigram:", corpus_pp_unigram)
    print("Bigram MLE:", corpus_pp_bigram_mle, "(if ∞ then some bigrams were unseen)")
    print("Bigram + Add-One:", corpus_pp_bigram_addone)

    # Discussion for Q7 (printed to console)
    print("\nDiscussion:")
    print("The Bigram MLE model often yields infinite perplexity if any test bigram is unseen,")
    print("whereas the Add-One smoothed model always gives a finite (though sometimes higher) perplexity.")
    print("The Unigram model typically results in higher perplexity because it ignores word order.")


if __name__ == "__main__":
    main()
