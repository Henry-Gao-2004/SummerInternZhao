def emo_classifier(text:str) -> str:
    # step 4
    prompt = f"""Please classify the emotion of the text into one of the emotions []. The text is: {text} """
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

def summarize(text: str) -> str:
    # step 5
    prompt = f"""Please summarize the text into one to two short sentences: {text} """
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