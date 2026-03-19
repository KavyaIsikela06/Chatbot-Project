from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Chatbot is running!"

@app.route("/chat")
def chat():
    user_message = request.args.get("msg")
    response = "Hello! You said: " + str(user_message)
    return jsonify({"response": response})

app.run(host="0.0.0.0", port=8000)
