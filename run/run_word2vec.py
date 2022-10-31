"""
Runs Word2Vec Baseline Replication
"""

from boto import config
from util import *
from data import *
from models import *
import json
import sys

if __name__ == '__main__':
    dataset = sys.argv[1]
    print('Running Word2Vec Replication:')
    
    stem = input("Enter 1 to stem data: ")
    print()

    print(f"Loading Data: {dataset} {'with' if stem else 'without'} stemming")
    d = Data(
        data_dir='data/', 
        dataset=dataset, 
        stem=stem, 
        special_tokens=True
    )
    print()
    w2v = Word2Vec_Model(d)
    
    while True:
        config_path = input('Enter 1 to load model setting from w2v_config.json, 2 to load model: ')
        if config_path == '1':
            with open('w2v_config.json', 'r') as f:
                model_config = json.load(f)
            
            print("Training Model:")
            w2v.fit(**model_config)
            break
        elif config_path == '2':
            model_path = input("Enter model path ended with .model: ")
            w2v.load_model(model_path)
            break
    
    # predict
    w2v_pred = w2v.predict()
    metric = evaluate(w2v.data.df.label, w2v_pred)
    for k, v in metric.items():
        print(f"{k}: {v}")
