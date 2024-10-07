import re
import random
import streamlit as st
from collections import defaultdict

# Function to preprocess text (tokenization)
def preprocess_text(text):
    # Remove punctuations and non-Urdu characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize text by spaces
    tokens = text.split()
    return tokens

# Function to create n-grams from tokens
def build_ngram_model(tokens, n):
    ngrams = defaultdict(lambda: defaultdict(int))
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n - 1])
        next_word = tokens[i + n - 1]
        ngrams[gram][next_word] += 1
    return ngrams

# Function to calculate probabilities for the next word
def calculate_probabilities(ngrams):
    probabilities = defaultdict(dict)
    for gram, next_words in ngrams.items():
        total_count = float(sum(next_words.values()))
        for word, count in next_words.items():
            probabilities[gram][word] = count / total_count
    return probabilities

# Function to predict the next word based on previous words
def predict_next_word(ngram_model, prev_words, n):
    prev_words = tuple(prev_words[-(n-1):])  # Get the last n-1 words
    if prev_words in ngram_model:
        next_word_probs = ngram_model[prev_words]
        # Choose the next word based on the highest probability
        next_word = max(next_word_probs, key=next_word_probs.get)
        return next_word
    return None

# Main Streamlit application
def main():
    st.title("n-Gram Language Model for Urdu Text Prediction")

    # User input for corpus text
    user_text = st.text_area("Enter your text corpus (Urdu):", height=300)

    if user_text:
        # Preprocess the corpus text
        tokens = preprocess_text(user_text)

        # User defines the value of n
        n = st.slider("Select the value of n for the n-gram model", min_value=2, max_value=1000, value=2)

        # Build the n-gram model with user-defined n
        ngram_model = build_ngram_model(tokens, n)

        # Calculate probabilities for the next word
        probabilities = calculate_probabilities(ngram_model)

        # User inputs the context for prediction
        context_input = st.text_input(f"Enter the last {n-1} words for context (space-separated):")

        if context_input:
            context = context_input.split()
            predicted_word = predict_next_word(probabilities, context, n)

            # Display prediction result
            if predicted_word:
                st.write(f"The predicted next word after '{' '.join(context)}' is: **{predicted_word}**")
            else:
                st.write("No prediction available for the given context.")

if __name__ == "__main__":
    main()
