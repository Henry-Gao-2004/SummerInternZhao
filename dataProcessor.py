import json



# Example usage
if __name__ == "__main__":
    soulchat_data = json.load(open("datasets/soulchat_original.json", "r",encoding="utf-8"))
    result_soulchat_data = []
    for conversation in soulchat_data:
        topic = conversation["topic"]
        messages = conversation["messages"]
        for i in range(1,len(messages),2):
            data_point = {}
            data_point["topic"] = topic
            data_point["target_text"] = messages[i]["content"]
            data_point["prev_text"] = messages[i-1]["content"]
            if i+1 < len(messages):
                data_point["next_text"] = messages[i+1]["content"]
            result_soulchat_data.append(data_point)
    with open("datasets/soulchat_processed.json", "w", encoding="utf-8") as f:
        json.dump(result_soulchat_data, f, indent=4)