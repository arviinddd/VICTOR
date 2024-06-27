import React, { useState } from 'react';
import './ChatWindow.css';  // Import CSS for styling

function ChatWindow() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');

    const sendMessage = async () => {
        const response = await fetch('/send', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: input }),
        });
        const data = await response.json();
        setMessages([...messages, { text: input, sender: 'user' }, { text: data.reply, sender: 'bot' }]);
        setInput('');
    };

    return (
        <div className="chat-container">
            <h1>Chat with Victor <img src="/victor.png" alt="bull" className="bull-icon" /></h1>
            <div className="chat-box">
                {messages.map((msg, index) => (
                    <p key={index} className={msg.sender === 'user' ? 'user-message' : 'bot-message'}>
                        <strong>{msg.sender === 'user' ? 'You: ' : 'Victor: '}</strong>{msg.text}
                    </p>
                ))}
            </div>
            <div className="input-group">
                <input
                    type="text"
                    placeholder="Type a message"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                />
                <button onClick={sendMessage}>
                    <i className="fas fa-paper-plane"></i>
                </button>
            </div>
        </div>
    );
}

export default ChatWindow;
