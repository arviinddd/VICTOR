# VICTOR- An University Chatbot
# Project Overview
This project is a chatbot application named "Victor," designed to assist users in finding information about graduate programs at the University at Buffalo. The chatbot leverages natural language processing and reinforcement learning to provide accurate and helpful responses to user queries.

# Technologies Used

Flask: A micro web framework for Python used to build the server-side application.

Flask-CORS: A Flask extension for handling Cross-Origin Resource Sharing (CORS), making it possible for the frontend to communicate with the backend.

Gymnasium: A toolkit for developing and comparing reinforcement learning algorithms.

Spacy: A library for advanced natural language processing in Python.

Transformers: A library for state-of-the-art natural language processing, used here for intent recognition.

React: A JavaScript library for building user interfaces.

Q-Learning: A reinforcement learning algorithm used for training the chatbot to interact with users effectively.

Python: The primary programming language used for developing the backend.
  
# Features

Natural Language Processing: Utilizes Spacy for processing and understanding user queries.

Intent Recognition: Uses a simple keyword-based approach to determine the intent of user queries.

Program Details and FAQs: Searches and matches user queries against a database of program details and frequently asked questions.

Reinforcement Learning: Implements a Q-learning agent to improve the chatbot's responses over time.

Frontend Interface: A React-based user interface allowing users to interact with the chatbot.

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
