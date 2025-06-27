import json
import requests
from tqdm import tqdm
import torch
from llm_prompting import generate_output, load_model

def merge_replies_with_same_author(data):
    def merge_replies(parent, replies):
        i = 0
        while i < len(replies):
            reply = replies[i]
            if reply["signatures"] == parent["signatures"]:
                # Move text of reply into parent
                parent["text"] += "\n" + reply["text"]
                # Move replies of reply into parent's replies
                parent["replies"].extend(reply["replies"])
                # Remove the merged reply
                replies.pop(i)
            else:
                # Recursively process the replies of the current reply
                merge_replies(reply, reply["replies"])
                i += 1

    for key, value in data.items():
        for reply in value["replies"]:
            merge_replies(reply, reply["replies"])

def label_score_changes_and_attitudes(data):
    def analyze_text(text):
        prompt = f"""Given the following text, determine whether it refers to increasing, decreasing, or keeping the score the same.
    
Text: "{text}"

Respond only with one of the following: "increasing", "decreasing", or "same"."""
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        tokenizer, model = load_model("google/gemma-3-4b-it", device)
        return generate_output(prompt, tokenizer, model, device, 1)

    def analyze_attitude(text):
        url = "http://localhost:11434/api/generate"
        prompt = f"What is the sentiment of the following text? Respond with 'positive', 'negative', or 'neutral'. Text: \"{text}\""

        payload = {
            "model": "gemma3:latest",
            "prompt": prompt,
            "stream": False  # Set to True if you want streamed response
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result['response'].strip()
        else:
            print("Error:", response.status_code, response.text)
            return None
    
    def label(reply):
        reply["score_change"] = analyze_text(reply["text"])
        reply["attitude"] = analyze_attitude(reply["text"])
        for reply2 in reply["replies"]:
            label(reply2)
    
    for key, value in tqdm(data.items()):
        for reply in value["replies"]:
            for reply2 in reply["replies"]:
                for reply3 in reply2["replies"]:
                    label(reply3)
                    

def count_replies(replies):
    count = 0
    for reply in replies:
            for reply2 in reply["replies"]:
                count += 1
                count += count_replies(reply2["replies"])
    return count

conversations = json.load(open("iclr_conversations_v1.json", "r"))
merge_replies_with_same_author(conversations)
# total_replies = sum(count_replies(value["replies"]) for value in conversations.values())
# print(f"Total number of reviewer replies: {total_replies}")
label_score_changes_and_attitudes(conversations)
json.dump(conversations, open("iclr_conversations_v2.json", "w"), indent=4)