import nltk
import numpy as np
import random
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a simple dataset
dataset = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "How are you", "Is anyone there?", "Good day"],
            "responses": ["Hello!", "Hi there!", "Greetings!", "How can I assist you today?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye", "Have a nice day"],
            "responses": ["Goodbye!", "See you later!", "Have a great day!", "Take care!"]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "That's helpful", "Thanks a lot"],
            "responses": ["You're welcome!", "Happy to help!", "Any time!", "My pleasure!"]
        },
        {
            "tag": "age",
            "patterns": ["How old are you?", "What's your age?", "Tell me your age"],
            "responses": ["I am a chatbot, so I don't have an age.", "I was created recently!", "I exist beyond time!"]
        },
        {
            "tag": "name",
            "patterns": ["What's your name?", "Who are you?", "Identify yourself"],
            "responses": ["I'm ChatBot!", "You can call me ChatBot.", "I'm your friendly chatbot."]
        }
    ]
}

def preprocess_sentence(sentence):
    # Convert to lowercase
    sentence = sentence.lower()
    # Remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    words = nltk.word_tokenize(sentence)
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Extract all patterns and corresponding tags
all_patterns = []
tags = []

for intent in dataset['intents']:
    for pattern in intent['patterns']:
        processed = preprocess_sentence(pattern)
        all_patterns.append(processed)
        tags.append(intent['tag'])

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the patterns
tfidf_matrix = vectorizer.fit_transform(all_patterns)

def get_response(user_input):
    # Preprocess the input
    processed_input = preprocess_sentence(user_input)
    # Vectorize the input
    input_vector = vectorizer.transform([processed_input])
    # Compute cosine similarity
    similarities = cosine_similarity(input_vector, tfidf_matrix)
    # Get the index of the highest similarity
    max_sim_index = np.argmax(similarities)
    # Get the similarity score
    max_sim = similarities[0][max_sim_index]
    
    # Define a similarity threshold
    threshold = 0.2
    
    if max_sim < threshold:
        return "I'm sorry, I don't understand that."
    else:
        # Get the tag corresponding to the highest similarity
        tag = tags[max_sim_index]
        # Fetch the appropriate response
        for intent in dataset['intents']:
            if intent['tag'] == tag:
                return np.random.choice(intent['responses'])

def chatbot():
    print("ChatBot: Hi! I'm ChatBot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("ChatBot: Goodbye! Have a great day.")
            break
        response = get_response(user_input)
        print(f"ChatBot: {response}")

if __name__ == "__main__":
    chatbot()
