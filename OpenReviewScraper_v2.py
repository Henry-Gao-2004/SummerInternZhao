import json
import requests
import openreview

def add_reply(conversation, id, quote, time, text):
    if conversation["id"] == quote:
        conversation["replies"].append({
            "id": id,
            "quote": quote,
            "time": time,
            "text": text,
            "replies": []
        })
    for reply in conversation["replies"]:
        add_reply(reply, id, quote, time, text)

def main():
    client = openreview.api.OpenReviewClient(
        baseurl='https://api2.openreview.net',
        username='henry.gao2@emory.edu',
        password='Gao20040701@'
    )
    print(client.get_all_notes(writer="~Razvan_Pascanu1"))

    conversations = {}
    iclr_years = ["2013", "2014","2016","2017","2018","2019","2020","2021","2022","2023","2024","2025"]
    for year in iclr_years:
        print(f"Scraping year {year}...")
        notes = client.get_all_notes(invitation='ICLR.cc/'+year+'/Conference/-/Submission',details = 'replies')
        if len(notes) == 0:
            print(f"No notes found for year {year}")
            notes = client.get_all_notes(invitation='ICLR.cc/'+year+'/-/submission/conference/review',details = 'replies')
        for note in notes:
            for reply in note.details["replies"]:
                id = reply["id"]
                forum = reply["forum"]
                quote = reply["replyto"]
                time = reply["cdate"]
                reply_string = "\n".join(f"{key}: {value["value"]}" for key, value in reply["content"].items())
                if forum not in conversations:
                    conversations[forum] = {
                        "id": forum,
                        "replies": []
                    }
                add_reply(conversations[forum], id, quote, time, reply_string)
    with open("conversations.json", "w") as f:
        json.dump(conversations, f, indent=4)
    print(conversations["014CgNPAGy"])




if __name__ == "__main__":
    main()