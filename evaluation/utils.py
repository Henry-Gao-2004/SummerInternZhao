from collections import Counter
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

nltk.download('punkt')

def compute_metrics(predictions, references):
    assert len(predictions) == len(references), "Mismatched prediction and reference counts."

    # Initialize
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction()

    # Accumulators
    rouge1, rouge2, rougel = 0, 0, 0
    bleu_scores = [0, 0, 0, 0]
    all_pred_tokens = []
    all_ref_tokens = []

    for pred, ref in zip(predictions, references):
        pred_tokens = nltk.word_tokenize(pred.lower())
        ref_tokens = nltk.word_tokenize(ref.lower())

        scores = rouge.score(ref, pred)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougel += scores['rougeL'].fmeasure

        for i in range(4):
            bleu_scores[i] += sentence_bleu([ref_tokens], pred_tokens, weights=tuple([1/(i+1)]*(i+1) + [0]*(3-i)), smoothing_function=smoothing.method1)

        # For distinct
        all_pred_tokens.extend(pred_tokens)
        all_ref_tokens.extend(ref_tokens)

    # Distinct
    def distinct_n(tokens, n):
        ngrams = list(zip(*[tokens[i:] for i in range(n)]))
        return len(set(ngrams)) / (len(ngrams) + 1e-8)

    distinct_1 = distinct_n(all_pred_tokens, 1)
    distinct_2 = distinct_n(all_pred_tokens, 2)

    # BERTScore
    (P, R, F), hashname = score(predictions, references, lang="en", return_hash=True)
    avg_bertscore = F.mean().item()

    # Return all metrics
    num = len(predictions)
    return {
        'ROUGE-1': rouge1 / num,
        'ROUGE-2': rouge2 / num,
        'ROUGE-L': rougel / num,
        'BLEU-1': bleu_scores[0] / num,
        'BLEU-2': bleu_scores[1] / num,
        'BLEU-3': bleu_scores[2] / num,
        'BLEU-4': bleu_scores[3] / num,
        'Distinct-1': distinct_1,
        'Distinct-2': distinct_2,
        'BERTScore-F1': avg_bertscore
    }


def compare_responses(response1, response2, reference, model_name="microsoft/deberta-v3-large"):
    """
    Compare two responses against a reference using a transformer model and return the better one.
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Tokenize inputs
    inputs1 = tokenizer(response1, reference, return_tensors="pt", truncation=True, padding=True)
    inputs2 = tokenizer(response2, reference, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    with torch.no_grad():
        score1 = model(**inputs1).logits.squeeze().item()
        score2 = model(**inputs2).logits.squeeze().item()

    # Return the better response
    return response1 if score1 > score2 else response2

if __name__  == "__main__":
    # Example usage
    predictions = ["The cat sat on the mat.", "Dogs are great companions."]
    references = ["A cat is sitting on the mat.", "Dogs make wonderful friends."]
    
    metrics = compute_metrics(predictions, references)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")