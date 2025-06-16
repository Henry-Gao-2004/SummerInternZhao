from baseline import reply as baseline_reply
from emollm import reply as emo_reply

if __name__ == "__main__":
    user_utterance = "I just lost my job. My life is falling apart."
    print("User utterance:", user_utterance)
    print("Baseline:", baseline_reply(user_utterance))
    print("EmoLLM:", emo_reply(user_utterance))