"""
Runs TF-IDF Baseline Replication
"""

from util import *
from data import *
from models import *
import json
import sys

if __name__ == '__main__':
    dataset = sys.argv[1]
    print('Running TF-IDF Replication:')

    for stem in [True, False]:
        if dataset == '20news/fine' and stem:
            continue
        
        print(f"Loading Data: {dataset} {'with' if stem else 'without'} stemming")
        d = Data(
            data_dir='data/', 
            dataset=dataset, 
            stem=stem, 
            special_tokens=True
        )
        print()
        
        tfidf = Tfidf_Model(data=d)
        for agg in ['mean', 'mean_sq', 'max']:
            print(f"Stem: {stem}\nAggregate: {agg}")
            tfidf_pred = tfidf.predict(agg=agg)
            metric = evaluate(tfidf.data.df.label, tfidf_pred)
            for k, v in metric.items():
                print(f"{k}: {v}")
            print()
        print()
