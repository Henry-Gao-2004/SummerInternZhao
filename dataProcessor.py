import json

from tqdm import tqdm
import os

def process_soulchat():
    soulchat_data = json.load(open("datasets/soulchat_original.json", "r",encoding="utf-8"))
    result_soulchat = []
    for conversation in tqdm(soulchat_data):
        topic = conversation["topic"]
        messages = conversation["messages"]
        for i in range(1,len(messages),2):
            data_point = {}
            data_point["topic"] = topic
            data_point["target_txt"] = messages[i]["content"]
            data_point["prev_txt"] = messages[i-1]["content"]
            if i+1 < len(messages):
                data_point["next_text"] = messages[i+1]["content"]
            result_soulchat.append(data_point)

    with open("datasets/soulchat_processed.json", "w", encoding="utf-8") as f:
        json.dump(result_soulchat, f, ensure_ascii=False, indent=4)

# act: 1: inform，2: question, 3: directive, 4: commissive
# emotion: 0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise
def process_daily_dialog(type: str="all"):
    if type == "all":
        process_daily_dialog("train")
        process_daily_dialog("test")
        process_daily_dialog("validation")
        return
    
    act_list = open("datasets/DailyDialog/"+type+"/dialogues_act_"+type+".txt", "r", encoding="utf-8").readlines()
    emo_list = open("datasets/DailyDialog/"+type+"/dialogues_emotion_"+type+".txt", "r", encoding="utf-8").readlines()
    txt_list = open("datasets/DailyDialog/"+type+"/dialogues_"+type+".txt", "r", encoding="utf-8").readlines()
    result_daily_dialog = []
    
    for i in range(len(act_list)):
        txt = txt_list[i].strip().split("__eou__ ")
        act = act_list[i].strip().split()
        emo = emo_list[i].strip().split()
        
        for j in range(1,len(txt)-1):
            data_point = {}
            data_point["target_txt"] = txt[j]
            data_point["target_act"] = int(act[j])
            data_point["target_emotion"] = int(emo[j])
            data_point["prev_txt"] = txt[j-1]
            data_point["prev_emo"] = int(emo[j-1])
            data_point["prev_act"] = int(act[j-1])
            data_point["next_txt"] = txt[j+1]
            data_point["next_emo"] = int(emo[j+1])
            data_point["next_act"] = int(act[j+1])
            result_daily_dialog.append(data_point)

    with open("datasets/daily_dialog_"+type+"_processed.json", "w", encoding="utf-8") as f:
        json.dump(result_daily_dialog, f, ensure_ascii=False, indent=4)


def process_meddialog(type: str="all"):
    if type == "all":
        process_meddialog("train")
        process_meddialog("test")
        process_meddialog("dev")
        return
    
    data = json.load(open("datasets/meddialog/english-"+type+".json", "r", encoding="utf-8"))
    result_meddialog = []
    for conversation in tqdm(data):
        situation = conversation["description"]
        utterances = conversation["utterances"]
        for idx in range(1,len(utterances),2):
            data_point = {}
            data_point["situation"] = situation
            data_point["target_text"] = utterances[idx][8:] # remove the "patient: " prefix
            data_point["prev_text"] = utterances[idx-1][9:] # remove the "doctor: " prefix
            if idx+1 < len(utterances):
                data_point["next_text"] = utterances[idx+1][9:] # remove the "patient: " prefix
            result_meddialog.append(data_point)

    with open("datasets/meddialog_"+type+"_processed.json", "w", encoding="utf-8") as f:
        json.dump(result_meddialog, f, ensure_ascii=False, indent=4)


def process_multiwoz(type: str="all"):
    if type == "all":
        process_multiwoz("train")
        process_multiwoz("test")
        process_multiwoz("dev")
        return
    
    result_multiwoz = []
    for f in os.listdir("datasets/multiwoz/"+type):
        if f.endswith(".json"):
            print(f)
            data = json.load(open("datasets/multiwoz/"+type+"/"+f, "r", encoding="utf-8"))
            for conversation in tqdm(data):
                conversation = conversation["turns"]
                for idx in range(1,len(conversation),2):
                    data_point = {}
                    data_point["target_text"] = conversation[idx]["utterance"]
                    data_point["prev_text"] = conversation[idx-1]["utterance"]
                    if idx+1 < len(conversation):
                        data_point["next_text"] = conversation[idx+1]["utterance"]
                    result_multiwoz.append(data_point)

    with open("datasets/multiwoz_"+type+"_processed.json", "w", encoding="utf-8") as f:
        json.dump(result_multiwoz, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    process_soulchat()
    process_daily_dialog()
    process_meddialog()
    process_multiwoz()