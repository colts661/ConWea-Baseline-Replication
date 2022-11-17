import sys
import os
import json
import argparse

sys.path.insert(0, 'src')

from util import *
from data import Data
from models import *

def main():
    """
    Runs the project logic, given the targets
    """
    targets = sys.argv[1:]

    # test target
    if 'test' in targets:
        d = Data(
            data_dir='test/', 
            dataset='testdata', 
            stem=True, 
            special_tokens=True
        )
        tfidf = Tfidf_Model(data=d)
        for agg in ['mean', 'mean_sq', 'max']:
            print(f'Test Data, Stemmed, {agg}')
            tfidf_pred = tfidf.predict(agg=agg)
            metric = evaluate(tfidf.data.df.label, tfidf_pred)
            print(metric)
            print()

if __name__ == "__main__":
    main()