import requests

def reply(utterance: str) -> str:
    prompt = f"""Please respond one to three sentences in a caring, but not too emotional way to the user. The user's utterance is: {utterance}"""
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "gemma3:latest",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()["message"]["content"].strip().lower()
        return result
    else:
        raise RuntimeError(f"Request failed: {response.status_code} {response.text}")

if __name__ == "__main__":
    user_utterance = "I just lost my job. My life is falling apart."
    response = reply(user_utterance)
    print("Response from baseline:", response)