import json
import sys
from utils import compute_metrics
import os
from tqdm import tqdm

if __name__ == "__main__":
    for f in os.listdir('evaluation'):
        print("Processing file:", f)
        data = json.load(open('evaluation/'+f, 'r'))
        ground_truth = []
        predictions = []
        for item in data:
            ground_truth.append(item['next_utterance'])
            predictions.append(item['out_model'])
        print(compute_metrics(predictions, ground_truth))

