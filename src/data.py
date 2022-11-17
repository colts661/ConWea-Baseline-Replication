"""
Data Loading
"""

import pickle
import json
import os
import numpy as np
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer

class Data:
    
    def __init__(self, data_dir, dataset, stem=False, special_tokens=False):
        """
        Load and preprocess data.
        """
        self.name = dataset
        path = data_dir + dataset

        # load corpus
        self.df = pickle.load(open(os.path.join(path, "df.pkl"), "rb"))
        self.true_labels = self.df.label
        self.START, self.END = ('<START>', '<END>') if special_tokens else ('', '')
        
        # load seedwords
        self.seedwords = {
            label: lst if label in lst else lst + [label] 
            for label, lst in 
            json.load(open(os.path.join(path, "seedwords.json"), "rb")).items()
        }

        # process labels
        self.labels = list(self.seedwords.keys())
        self.pred_to_label = np.vectorize(lambda idx: self.labels[idx])

        # preprocess corpus
        self.stemmer = EnglishStemmer() if stem else None
        self.tokenizer = RegexpTokenizer('\w\w+')
        self.preprocess()

    def preprocess(self):
        if self.stemmer: # tokenize, stem both seedwords and corpus
            self.seedwords = {
                label: set(self.stemmer.stem(w) for w in lst)
                for label, lst in 
                self.seedwords.items()
            }
            self.corpus = (
                self.df['sentence']
                .str.lower()
                .apply(lambda s: [self.START] + [
                    self.stemmer.stem(w) 
                    for w in self.tokenizer.tokenize(s)
                ] + [self.END])
                .tolist()
            )
        else:
            self.corpus = (
                self.df['sentence']
                .str.lower()
                .apply(lambda s: [self.START] + [
                    w for w in self.tokenizer.tokenize(s)
                ] + [self.END])
                .tolist()
            )
        
    def __repr__(self):
        return f"{self.name.replace('/', '_')} dataset"
 