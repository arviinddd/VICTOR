# VICTOR- An University Chatbot

# Project Overview

"Victor" is an innovative chatbot application designed to assist prospective students in navigating the vast landscape of graduate programs offered at the University at Buffalo. By harnessing the power of natural language processing (NLP) and reinforcement learning, Victor provides instant, accurate, and contextually relevant responses to user queries, greatly enhancing the user experience and streamlining the information retrieval process.

# Key Features
1. Natural Language Processing:

       Spacy: Used for advanced text processing to understand and interpret user queries.

       Bert: Utilized for intent recognition to accurately determine the user's intent based on their input.

3. Reinforcement Learning:

        Q-Learning: Implements a Q-learning algorithm to train the chatbot for better interaction and response over time.

4. Program Details and FAQs:

        Integrates a database of program details and frequently asked questions (FAQs) to provide precise answers to user queries.

5. Frontend Interface:

        React: Provides a user-friendly interface for users to interact with the chatbot.

# Technologies Used

Flask: A micro web framework for Python used to build the server-side application.

Flask-CORS: A Flask extension for handling Cross-Origin Resource Sharing (CORS), making it possible for the frontend to communicate with the backend.

Gymnasium: A toolkit for developing and comparing reinforcement learning algorithms.

Spacy: A library for advanced natural language processing in Python.

Transformers: A library for state-of-the-art natural language processing, used here for intent recognition.

React: A JavaScript library for building user interfaces.

Q-Learning: A reinforcement learning algorithm used for training the chatbot to interact with users effectively.

Python: The primary programming language used for developing the backend.


# How It Works

User Interaction: Users interact with the chatbot through a chat interface. They can ask questions about graduate programs, application procedures, deadlines, fees, and more.

Intent Recognition: The chatbot processes the user's input using NLP techniques to determine the intent behind the query.

Information Retrieval: Based on the recognized intent, the chatbot searches the program details and FAQs to find the most relevant information.

Response Generation: The chatbot generates a response and sends it back to the user through the chat interface.

Reinforcement Learning: The chatbot uses reinforcement learning to improve its responses over time by learning from interactions.

# File Structure
app.py: Main Flask application file.

ChatWindow.js: React component for the chat window.

ChatWindow.css: CSS file for styling the chat window.

files/program_details.json: JSON file containing program details.

files/Combined_FAQs.json: JSON file containing FAQs.

# Usage
Navigate to http://localhost:3000 to interact with the chatbot.
Type a message in the input box and press send.
The chatbot will respond with information related to your query.
Customization
Program Details: Update the program_details.json file with the latest program information.
FAQs: Update the Combined_FAQs.json file with the latest frequently asked questions and answers.
Training the Q-Learning Agent: Modify the train_agent function in app.py to adjust training parameters and improve the chatbot's performance.
Future Enhancements
Integrate more advanced natural language understanding models for better intent recognition.
Implement more sophisticated reward mechanisms in the Q-learning agent.
Expand the database to include more detailed information about programs and services.

# Acknowledgments
The University at Buffalo for providing the program details and FAQs.

The developers and maintainers of Flask, Gymnasium, Spacy, Transformers, React, and other open-source projects used in this application.
