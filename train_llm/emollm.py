import requests

emo_list = ["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

def get_emo(utterance: str) -> int:
    prompt = f"""Classify the user's utterance. Return 1 for anger, 2 for disgust, 3 for fear, 4 for happiness, 5 for sadness, 6 for surprise, 0 for unknown. Only return the number. The user's utterance is: {utterance}"""
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
        return int(result)
    else:
        raise RuntimeError(f"Request failed: {response.status_code} {response.text}")

def get_fact(utterance: str) -> str:
    prompt = f"""Please summarize the fact you can get in the user's utterance. Return one to two sentences. The user's utterance is: {utterance}"""
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

def process_emo(emo: str, fact: str) -> str:
    prompt = f"""Given the user's emotion {emo} and the fact {fact}, please deduce step by step the user's current emotion. Do not return the thinking, only return the final emotion. """
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

def generate_response(emo: str, fact: str) -> str:
    prompt = f"""The user's current emotion is {emo} and {fact}. Please respond one to three sentences in a caring manner. """
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

def reply(user_utterance):
    emo = get_emo(user_utterance)
    # print("User emotion:", emo_list[emo])
    fact = get_fact(user_utterance)
    # print("User fact:", fact)
    emo_p = process_emo(emo_list[emo], fact)
    # print("Processed emotion:", emo_p)
    response = generate_response(emo_p, fact)
    # print("Response:", response)
    return response

if __name__ == "__main__":
    user_utterance = "I just lost my job. My life is falling apart."
    reply(user_utterance)