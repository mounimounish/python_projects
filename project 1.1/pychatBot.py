#NumPy and Natural Language Processing
import numpy as np
import re
import random
from collections import Counter

class SimpleChatbot:
    def __init__(self):
        # Training data - simple question-answer pairs
        self.training_data = {
            'greetings': {
                'patterns': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
                'responses': ['Hello! How can I help you?', 'Hi there!', 'Hey! What can I do for you?', 'Greetings!']
            },
            'farewell': {
                'patterns': ['bye', 'goodbye', 'see you', 'farewell', 'exit', 'quit'],
                'responses': ['Goodbye!', 'See you later!', 'Have a great day!', 'Farewell!']
            },
            'name': {
                'patterns': ['what is your name', 'who are you', 'your name'],
                'responses': ['I am a simple chatbot', 'My name is ChatBot', 'I am your AI assistant']
            },
            'help': {
                'patterns': ['help', 'what can you do', 'assist me'],
                'responses': ['I can chat with you and answer simple questions', 'I am here to help you with basic conversations']
            },
            'weather': {
                'patterns': ['weather', 'how is the weather', 'is it sunny', 'is it raining'],
                'responses': ['I cannot check real weather, but I hope it is nice!', 'Weather looks good from here!']
            },
            'hobbies': {
                'patterns': ['what do you like to do', 'your hobbies', 'do you have any hobbies', 'what are your interests'],
                'responses': ['I enjoy processing data and learning new things!', 'My main hobby is helping people like you.', 'I like to analyze text and generate responses.', 'As a program, I enjoy executing code.']
            },
            'feelings': {
                'patterns': ['how are you', 'how are you feeling', 'are you okay', 'how is your day'],
                'responses': ['I am a program, so I do not have feelings, but I am operating perfectly!', 'I am functioning as expected.', 'I am doing great, thank you for asking!']
            },
            'origin': {
                'patterns': ['where are you from', 'where were you created', 'who made you', 'who is your creator'],
                'responses': ['I was created by a programmer.', 'I live on the internet!', 'I am a digital entity, so I do not have a physical home.']
            },
            'food': {
                'patterns': ['what is your favorite food', 'do you eat', 'favorite drink'],
                'responses': ['I do not eat or drink, but I can process information about all kinds of delicious foods!', 'I run on code, not calories!', 'My favorite food is a well-structured dataset.']
            },
            'time': {
                'patterns': ['what time is it', 'what is the time', 'current time'],
                'responses': ['I cannot tell the current time accurately, as my purpose is to chat with you, not to be a clock.', 'Time is a concept I do not track in real-time.']
            },

            'joke': {
                'patterns': ['tell me a joke', 'say something funny', 'do you know any jokes'],
                'responses': ['Why do not scientists trust atoms? Because they make up everything!', 'What do you call a fake noodle? An impasta!', 'I told my wife she was drawing her eyebrows too high. She looked surprised.']
            },
            'fact': {
                'patterns': ['tell me a fact', 'give me a fact', 'did you know', 'random fact'],
                'responses': ['The world\'s oldest piece of chewing gum is over 9,000 years old.', 'A single cloud can weigh more than 1 million pounds.', 'Banging your head against a wall for one hour burns 150 calories.']
            },
            'default': {
                'patterns': [],
                'responses': ['I am not sure how to respond to that', 'Could you rephrase that?', 'Interesting! Tell me more.']
            } # pyright: ignore[reportInvalidTypeForm]
        }
        
        # Convert patterns to numpy arrays for processing
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare training data using NumPy for processing"""
        self.pattern_vectors = {}
        
        for intent, data in self.training_data.items():
            if intent != 'default':
                # Create simple word vectors using numpy
                patterns = data['patterns']
                self.pattern_vectors[intent] = np.array([self.text_to_vector(pattern) for pattern in patterns])
    
    def text_to_vector(self, text):
        """Simple text to vector conversion using character frequency"""
        # Convert to lowercase and get character frequencies
        text = text.lower()
        char_freq = Counter(text)
        
        # Create a simple 26-dimensional vector for a-z characters
        vector = np.zeros(26)
        for char, freq in char_freq.items():
            if 'a' <= char <= 'z':
                vector[ord(char) - ord('a')] = freq
        
        return vector
    
    def preprocess_input(self, text):
        """Preprocess user input"""
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    
    def calculate_similarity(self, input_vector, pattern_vectors):
        """Calculate cosine similarity between input and patterns"""
        similarities = []
        for pattern_vector in pattern_vectors:
            # Calculate cosine similarity using NumPy
            dot_product = np.dot(input_vector, pattern_vector)
            norm_input = np.linalg.norm(input_vector)
            norm_pattern = np.linalg.norm(pattern_vector)
            
            if norm_input == 0 or norm_pattern == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_input * norm_pattern)
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def get_intent(self, user_input):
        """Determine the intent of user input"""
        processed_input = self.preprocess_input(user_input)
        input_vector = self.text_to_vector(processed_input)
        
        best_intent = 'default'
        best_similarity = 0
        
        # Check for keyword matches first (simple NLP)
        for intent, data in self.training_data.items():
            if intent != 'default':
                for pattern in data['patterns']:
                    if pattern in processed_input:
                        return intent
        
        # If no keyword match, use vector similarity
        for intent, pattern_vectors in self.pattern_vectors.items():
            similarities = self.calculate_similarity(input_vector, pattern_vectors)
            max_similarity = np.max(similarities)
            
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_intent = intent
        
        # Threshold for similarity
        if best_similarity < 0.3:
            best_intent = 'default'
        
        return best_intent
    
    def get_response(self, user_input):
        """Generate response based on user input"""
        intent = self.get_intent(user_input)
        responses = self.training_data[intent]['responses']
        return random.choice(responses)
    
    def chat(self):
        """Main chat loop"""
        print("Simple Chatbot: Hello! I'm a simple chatbot. Type 'quit' to exit.")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Chatbot: Goodbye! Have a great day!")
                break
            
            response = self.get_response(user_input)
            print(f"Chatbot: {response}")

# Create and train the chatbot
def main():
    print("=== Simple Python Chatbot Project ===")
    print("This chatbot uses NumPy for data processing and simple NLP techniques")
    print()
    
    # Create chatbot instance
    chatbot = SimpleChatbot()
    
    # Demonstrate the training data
    print("Training Data Summary:")
    for intent, data in chatbot.training_data.items():
        if intent != 'default':
            print(f"- {intent}: {len(data['patterns'])} patterns, {len(data['responses'])} responses")
    print()
    
    # Start chatting
    chatbot.chat()

if __name__ == "__main__":
    main()