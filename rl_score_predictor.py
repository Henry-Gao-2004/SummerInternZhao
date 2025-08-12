# train_sentence_context_regression.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW

import json
# ===== Example dataset =====
class SentenceContextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, context, score = self.data[idx]
        sent_enc = self.tokenizer(
            sentence, padding='max_length', truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        ctx_enc = self.tokenizer(
            context, padding='max_length', truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "sent_input_ids": sent_enc["input_ids"].squeeze(0),
            "sent_attention_mask": sent_enc["attention_mask"].squeeze(0),
            "ctx_input_ids": ctx_enc["input_ids"].squeeze(0),
            "ctx_attention_mask": ctx_enc["attention_mask"].squeeze(0),
            "score": torch.tensor(score, dtype=torch.float)
        }

# ===== Model =====
class SentenceContextRegressor(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_size=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, sent_input_ids, sent_attention_mask, ctx_input_ids, ctx_attention_mask):
        sent_out = self.bert(sent_input_ids, attention_mask=sent_attention_mask).pooler_output
        ctx_out = self.bert(ctx_input_ids, attention_mask=ctx_attention_mask).pooler_output
        combined = torch.cat([sent_out, ctx_out], dim=1)
        return self.regressor(combined).squeeze(-1)

# ===== Example Training =====
def train():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    datasets = ["uss_rating_processed.csv"]
    train_data = []
    for dataset in datasets:
        with open(dataset, "r") as f:
            data = json.load(f)
            for data_point in data:
                sentence = data_point["target_text"]
                context = data_point["history"]
                score = data_point["score"]
                train_data.append((sentence, context, score))

    dataset = SentenceContextDataset(train_data, tokenizer)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = SentenceContextRegressor()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(
                batch["sent_input_ids"].to(device),
                batch["sent_attention_mask"].to(device),
                batch["ctx_input_ids"].to(device),
                batch["ctx_attention_mask"].to(device)
            )
            loss = criterion(outputs, batch["score"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "sentence_context_regressor.pt")
    print("Model saved to sentence_context_regressor.pt")

if __name__ == "__main__":
    train()
