# app.py
# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import spacy
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import random
from collections import defaultdict

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Define ChatEnvironment
class ChatEnvironment(gym.Env):
    def __init__(self):
        super(ChatEnvironment, self).__init__()
        self.action_space = spaces.Discrete(3)  # Number of actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.state = None
        self.done = False

    def reset(self, **kwargs):
        self.state = np.array([0.0], dtype=np.float32)  # Initial state
        self.done = False
        return self.state, {}  # Return observation and additional info

    def step(self, action):
        reward = 1 if action == 0 else -1  # Simplified reward logic
        self.state = np.array([0.0], dtype=np.float32)  # Update state
        self.done = True  # Simplified, ends after one step
        return self.state, reward, self.done, {}

    def render(self, mode='human'):
        pass

env = ChatEnvironment()  # Create the Gymnasium environment

# Load Spacy model for text processing
nlp = spacy.load('en_core_web_sm')

# Load JSON data
with open("files/program_details.json", 'r') as file:
    program_details = json.load(file)
with open("files/Combined_FAQs.json", 'r') as file:
    faqs = json.load(file)

# Define CustomLangChain
class CustomLangChain:
    def __init__(self, programs, faqs):
        self.program_details = programs
        self.faqs = faqs

    def check_faqs(self, query):
        query_lower = query.lower()
        best_match = None
        highest_similarity = 0.0

        for category, questions in self.faqs.items():
            for question, answer in questions.items():
                query_tokens = set(query_lower.split())
                question_tokens = set(question.lower().split())
                common_tokens = query_tokens.intersection(question_tokens)
                similarity = float(len(common_tokens)) / len(query_tokens.union(question_tokens))

                if similarity > highest_similarity and similarity > 0.5:  # Adjust the threshold as needed
                    highest_similarity = similarity
                    best_match = answer

        return best_match

    def find_program(self, query):
        query_doc = nlp(query.lower())
        query_tokens = set([token.lemma_ for token in query_doc if not token.is_stop and not token.is_punct])
        best_match = None
        max_similarity = 0
        for program in self.program_details:
            program_name = program['Program Name'].lower()
            program_doc = nlp(program_name)
            program_tokens = set([token.lemma_ for token in program_doc if not token.is_stop and not token.is_punct])
            similarity = len(query_tokens.intersection(program_tokens)) / len(query_tokens.union(program_tokens))
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = program
        return best_match

    def generate_response(self, intent, input_text):
        faq_response = self.check_faqs(input_text)
        if faq_response:
            return faq_response

        program = self.find_program(input_text)
        if program:
            response_templates = {
                0: f"The tuition fees for {program['Program Name']} are not explicitly listed. For more details, please visit {program['URL']}.",
                1: f"To apply for {program['Program Name']}, please check the detailed application steps at {program['URL']}.",
                2: f"The application deadline for {program['Program Name']} is {program['Deadline']}.",
                3: f"{program['Program Name']} is offered by the {program['College Name']}.",
                4: f"The instruction method for {program['Program Name']} is {program['Instruction Method']}.",
                5: f"{program['Program Name']} can be pursued on a {program['Full/Part Time Options']} basis.",
                6: f"{program['Program Name']} requires {program['Credits Required']} credits to complete.",
                7: f"The estimated time to complete {program['Program Name']} is {program['Time to Degree']}.",
                8: f"The application fee for {program['Program Name']} is {program['Application Fee']}.",
            }
            return response_templates.get(intent, "I'm sorry, I couldn't find the program you're asking about.")
        return "I'm sorry, I couldn't find the program you're asking about."

# Define intent recognition
def get_intent(input_text):
    lower_text = input_text.lower()
    if "fee" in lower_text or "cost" in lower_text:
        return 0
    elif "apply" in lower_text:
        return 1
    elif "deadline" in lower_text:
        return 2
    elif "college" in lower_text:
        return 3
    elif "method" in lower_text:
        return 4
    elif "time" in lower_text:
        return 5
    elif "credit" in lower_text:
        return 6
    elif "degree time" in lower_text:
        return 7
    elif "application fee" in lower_text:
        return 8
    return -1  # No intent recognized

# Initialize dialogue manager
dialogue_manager = CustomLangChain(program_details, faqs)

def respond_to_user(input_text):
    intent = get_intent(input_text)
    response = dialogue_manager.generate_response(intent, input_text)
    return response

# Define Q-learning agent
class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return self.action_space.sample()  # Explore action space
        else:
            return np.argmax(self.q_table[state])  # Exploit learned values

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        self.exploration_rate *= self.exploration_decay

agent = QLearningAgent(env.action_space)

def train_agent(episodes=10000):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(tuple(state))
            next_state, reward, done, _ = env.step(action)
            agent.learn(tuple(state), action, reward, tuple(next_state))
            state = next_state
    agent.q_table = dict(agent.q_table)  # Convert defaultdict to dict for saving

def save_agent(filepath="chatbot_model.npy"):
    np.save(filepath, agent.q_table)

def load_agent(filepath="chatbot_model.npy"):
    q_table = np.load(filepath, allow_pickle=True).item()
    agent.q_table = defaultdict(lambda: np.zeros(env.action_space.n), q_table)

train_agent()
save_agent()

def chatbot_conversation():
    load_agent()
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.choose_action(tuple(state))
        state, reward, done, _ = env.step(action)
    return "Chat session ended"  # Simplified, replace with actual rendering logic

def handle_chat_session(user_input):
    response = respond_to_user(user_input)
    print(f"Bot: {response}")
    return chatbot_conversation()

@app.route('/send', methods=['POST'])
def handle_message():
    user_input = request.json['message']
    response = respond_to_user(user_input)
    return jsonify({'reply': response})

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Run the server on port 5001
