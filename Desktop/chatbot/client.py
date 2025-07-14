import requests

API_URL = "http://localhost:8000/chat"

print("Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"): 
        break
    response = requests.post(API_URL, json={"text": user_input})
    if response.ok:
        print("Bot:", response.json().get("reply"))
    else:
        print("Error:", response.text) 