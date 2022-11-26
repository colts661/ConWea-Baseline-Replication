"""
Data Loading
"""

import pickle
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer

class Data:
    
    def __init__(self, data_dir, dataset, stem=False, special_tokens=True):
        """
        Load and preprocess data.
        """
        if dataset == '20news/fine' and stem:
            raise RuntimeError(f'Stemming {dataset} not supported')
        
        self.name = dataset
        self.path = os.path.join(data_dir, dataset)
        self.START, self.END = ('<START>', '<END>') if special_tokens else ('', '')
        self.stem = stem
        self.stemmer = EnglishStemmer().stem if stem else lambda word: word
        self.tokenizer = RegexpTokenizer('\w\w+')
        self.corpus_path = os.path.join(self.path, f"corpus_{'stem' if stem else ''}.pkl")

        # load corpus
        if os.path.exists(self.corpus_path):
            self.corpus = pickle.load(open(self.corpus_path, 'rb'))
            self.true_labels = pickle.load(open(os.path.join(self.path, 'labels.pkl'), 'rb'))
        else:
            self.preprocess()
        
        # load seedwords
        self.seedwords = {
            label: 
                set(self.stemmer(w) for w in lst) if label in lst 
                else set(self.stemmer(w) for w in lst + [label])
            for label, lst in 
            json.load(open(os.path.join(self.path, "seedwords.json"), "rb")).items()
        }

        # process labels
        self.labels = list(self.seedwords.keys())
        self.pred_to_label = np.vectorize(lambda idx: self.labels[idx])
    
    def preprocess(self):
        """
        Preprocess corpus and store result into file
        """
        print(f'Preprocessing Corpus {self.name}')
        df = pickle.load(open(os.path.join(self.path, "df.pkl"), "rb"))

        # tokenize, stem both seedwords and corpu
        self.corpus = (
            df['sentence']
            .str.lower()
            .apply(lambda s: [
                self.stemmer(w) 
                for w in self.tokenizer.tokenize(f'{self.START} {s} {self.END}'.strip())
            ])
            .tolist()
        )
        self.true_labels = df['label'].to_numpy()

        with open(self.corpus_path, 'wb') as f:
            pickle.dump(self.corpus, f)
        
        with open(os.path.join(self.path, 'labels.pkl'), 'wb') as f:
            pickle.dump(self.true_labels, f)
        print()
    
    def plot_distribution(self):
        df = pickle.load(open(os.path.join(self.path, "df.pkl"), "rb"))

        # visualize lengths
        df_len = df.assign(length=df.sentence.apply(lambda s: len(s)))
        bins = np.linspace(df_len.length.min(), df_len.length.max(), 100)

        # overlay histograms
        for name, group_df in df_len.groupby('label'):
            plt.hist(group_df.length, bins, alpha=0.5, label=name)

        plt.legend(loc='upper right')
        plt.title(f"Distribution of Document Lengths: {name.replace('/', '_')}")
        plt.show()

    def __repr__(self):
        return f"{self.name.replace('/', '_')} dataset"
 