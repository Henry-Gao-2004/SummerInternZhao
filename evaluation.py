import json
import sys
from utils import compute_metrics

if __name__ == "__main__":
    data = json.load(open('evaluation/cold_start_7b_long_reasoning_sft_eval.valid.json'))
    ground_truth = []
    predictions = []
    for item in data:
        ground_truth.append(item['true_response'])
        predictions.append(item['reasoning_sft_model_response'])
    
    print(compute_metrics(predictions, ground_truth))

