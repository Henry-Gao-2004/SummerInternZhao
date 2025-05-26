import json
import requests
import openreview
from tqdm import tqdm

def add_reply(conversation, id, quote, time, text,signatures,writers):
    if conversation["id"] == quote:
        if "signature" in conversation and conversation["signatures"] == signatures:
            print("two parts identified")
            conversation["text"] = conversation["text"]+"\n"+text
        else:
            conversation["replies"].append({
                "id": id,
                "quote": quote,
                "time": time,
                "text": text,
                "signatures": signatures,
                "writers": writers,
                "replies": []
            })
        return
    for reply in conversation["replies"]:
        add_reply(reply, id, quote, time, text,signatures,writers)

def main():
    client = openreview.Client(
        baseurl='https://api.openreview.net',
        username='henry.gao2@emory.edu',
        password='Gao20040701@'
    )

    with open("iclr_invitations.txt", "r") as file:
        invitations = [line.strip() for line in file.readlines()]
    print(len(invitations))
    if len(invitations) == 0:
        for invitation in client.get_all_invitations(regex=r"ICLR.*"):
            invitations.append(invitation.id)
        invitations.extend(["ICLR.cc/2013/conference/-/submission",
                   "ICLR.cc/2013/-/submission/review",
                   "ICLR.cc/2013/-/submission/reply",
                   "ICLR.cc/2014/conference/-/submission",
                   "ICLR.cc/2014/-/submission/conference/reply",
                   "ICLR.cc/2014/-/submission/conference/review",
                   "ICLR.cc/2014/workshop/-/submission",
                   "ICLR.cc/2014/-/submission/workshop/review",
                   "ICLR.cc/2014/-/submission/workshop/reply",
                   ])
    conversations = json.load(open("iclr_conversations.json", "r"))
    
    
    index = 0
    for invitation in tqdm(invitations):
        notes = client.get_all_notes(invitation=invitation, details='replies')
        for note in notes:
            for reply in note.details["replies"]:
                id = reply["id"]
                forum = reply["forum"]
                quote = reply["replyto"]
                time = reply["tcdate"]
                signatures = reply["signatures"]
                writers = reply["writers"]
                reply_string = "\n".join(f"{key}: {value}" for key, value in reply["content"].items())
                if forum not in conversations:
                    conversations[forum] = {
                        "id": forum,
                        "replies": []
                    }
                add_reply(conversations[forum], id, quote, time, reply_string, signatures, writers)
        index += 1
        if index %500 ==0:
            # print(f"\nSaving {index} invitations")
            with open("iclr_invitations.txt", "w") as invitations_file:
                for idx in range(index,len(invitations)):
                    invitation = invitations[idx]
                    invitations_file.write(invitation + "\n")
            with open("iclr_conversations.json", "w") as f:
                json.dump(conversations, f, indent=4)
    
    with open("iclr_invitations.txt", "w") as invitations_file:
        for idx in range(index, len(invitations)):
            invitation = invitations[idx]
            invitations_file.write(invitation + "\n")
    with open("iclr_conversations.json", "w") as f:
        json.dump(conversations, f, indent=4)


if __name__ == "__main__":
    main()