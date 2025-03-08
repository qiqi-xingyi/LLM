# --*-- conding:utf-8 --*--
# @Time : 3/7/25 5:24 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : Word2Vec.py

import numpy as np
import matplotlib.pyplot as plt
import re


def tokenize_corpus(text):
    """
    Convert text to lowercase, remove punctuation, and split on whitespace.
    """
    text = text.lower()
    # Remove punctuation (retain letters and whitespace only)
    text = re.sub(r'[^a-z\s]', '', text)
    # Split on whitespace
    tokens = text.split()
    return tokens


def build_vocab(tokens):
    """
    Build mapping from words to indices (word2idx) and indices to words (idx2word).
    """
    vocab = sorted(set(tokens))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    return word2idx, idx2word


def generate_training_pairs(tokens, word2idx, window_size=2):
    """
    Generate (center_word_idx, context_word_idx) pairs under the Skip-Gram model.
    """
    pairs = []
    for i, token in enumerate(tokens):
        center_idx = word2idx[token]
        # Consider words within [i - window_size, i + window_size]
        for w in range(-window_size, window_size + 1):
            if w == 0:
                continue
            context_pos = i + w
            if context_pos < 0 or context_pos >= len(tokens):
                continue
            context_word = tokens[context_pos]
            context_idx = word2idx[context_word]
            pairs.append((center_idx, context_idx))
    return pairs


class Word2Vec:
    def __init__(self, vocab_size, embedding_dim=50, learning_rate=0.01):
        """
        Maintain two embedding matrices:
        U for center-word embeddings (vocab_size x embedding_dim),
        V for context-word embeddings (vocab_size x embedding_dim).
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate

        self.U = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.V = np.random.randn(vocab_size, embedding_dim) * 0.01

    def softmax(self, z):
        """
        Compute a numerically stable softmax of vector z.
        """
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

    def forward_backward(self, center_idx, context_idx):
        """
        Forward + Backward pass for a single (center_idx, context_idx) pair.
        Returns the cross-entropy loss for this pair.
        """
        U_c = self.U[center_idx]  # center word embedding, shape: (embedding_dim,)

        # Calculate scores for all words: z = U_c dot V[w], for w in [0..vocab_size-1]
        z = self.V.dot(U_c)  # shape: (vocab_size,)
        # Softmax to get predicted distribution
        y_hat = self.softmax(z)

        # True one-hot label
        y = np.zeros(self.vocab_size)
        y[context_idx] = 1.0

        # Cross-entropy loss
        loss = -np.log(y_hat[context_idx] + 1e-15)

        # Backpropagation
        dz = y_hat.copy()  # shape: (vocab_size,)
        dz[context_idx] -= 1.0  # dz = (y_hat - y)

        # Grad w.r.t. V: outer product of dz and U_c
        dV = np.outer(dz, U_c)  # shape: (vocab_size, embedding_dim)
        # Grad w.r.t. U_c
        dU_c = dz.dot(self.V)  # shape: (embedding_dim,)

        # Update parameters
        self.V -= self.lr * dV
        self.U[center_idx] -= self.lr * dU_c

        return loss

    def train(self, training_pairs, epochs=5):
        """
        Train with SGD for a given number of epochs.
        """
        losses = []
        for epoch in range(epochs):
            np.random.shuffle(training_pairs)
            total_loss = 0.0

            for (center_idx, context_idx) in training_pairs:
                loss = self.forward_backward(center_idx, context_idx)
                total_loss += loss

            avg_loss = total_loss / len(training_pairs)
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss = {avg_loss:.4f}")
        return losses

    def get_word_vector(self, word_idx):
        """
        Return the final embedding of a word by combining U[word_idx] and V[word_idx].
        You can also choose to use just U or just V if desired.
        """
        return (self.U[word_idx] + self.V[word_idx]) / 2.0


if __name__ == "__main__":

    # A larger text excerpt from Lewis Carroll's "Alice in Wonderland".

    corpus_text = """
Alice was beginning to get very tired of sitting by her sister on the bank 
and of having nothing to do once or twice she had peeped into the book her sister was reading 
but it had no pictures or conversations in it and what is the use of a book thought Alice without pictures or conversation

So she was considering in her own mind as well as she could for the hot day made her feel very sleepy and stupid 
whether the pleasure of making a daisychain would be worth the trouble of getting up and picking the daisies 
when suddenly a White Rabbit with pink eyes ran close by her

There was nothing so very remarkable in that nor did Alice think it so very much out of the way to hear the Rabbit say to itself 
Oh dear Oh dear I shall be late when she thought it over afterwards it occurred to her that she ought to have wondered at this 
but at the time it all seemed quite natural
"""

    # Tokenize
    tokens = tokenize_corpus(corpus_text)

    # Build vocab and index mappings
    word2idx, idx2word = build_vocab(tokens)
    vocab_size = len(word2idx)
    print("Vocabulary Size:", vocab_size)

    # Create Skip-Gram training pairs
    window_size = 2
    training_pairs = generate_training_pairs(tokens, word2idx, window_size=window_size)
    print("Number of (center, context) pairs:", len(training_pairs))

    # Initialize and train the model
    embedding_dim = 50
    learning_rate = 0.01
    epochs = 10

    w2v = Word2Vec(vocab_size, embedding_dim=embedding_dim, learning_rate=learning_rate)
    losses = w2v.train(training_pairs, epochs=epochs)

    # Plot the training loss
    plt.plot(range(1, len(losses) + 1), losses)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


    # Simple word similarity check
    def cosine_similarity(vec1, vec2):
        return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-15)


    test_words = ["alice", "rabbit", "book", "sister", "bank", "eyes"]
    print("\nWord Similarities:")
    for w1 in test_words:
        for w2 in test_words:
            if w1 == w2:
                continue
            sim = cosine_similarity(w2v.get_word_vector(word2idx[w1]),
                                    w2v.get_word_vector(word2idx[w2]))
            print(f"Similarity({w1}, {w2}) = {sim:.4f}")
