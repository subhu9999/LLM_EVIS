import nltk
from nltk.tokenize import word_tokenize
import os
import streamlit as st

#nltk.download('punkt')

# @st.cache_resource
# def download_nltk_punkt():
#     nltk.download('punkt')

# Only download 'punkt' once
# download_nltk_punkt()

nltk.download('punkt')

# List of common question words
question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'do', 'does', 'can', 'could', 'would',
                  'will', 'should'}

# Additional phrases that indicate a question
additional_indicators = ['please provide', 'can you explain', 'how does', 'could you tell', 'would you mind',
                         'could you', 'tell me', 'what is', 'which product', 'are you', 'do you']


def is_question(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence.lower())

    # Check if sentence starts with a question word or ends with a question mark
    if tokens[0] in question_words or sentence.strip().endswith('?'):
        return True

    # Check for additional question indicators
    for phrase in additional_indicators:
        if phrase in sentence.lower():
            return True

    return False