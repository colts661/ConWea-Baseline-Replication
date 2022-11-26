"""
Collection of Models
"""

from data import *
from util import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from tqdm import tqdm

class Tfidf_Model:
    """
    Baseline model that utlizes TF-IDF to find document-label relevance.
    """
    
    def __init__(self, data: Data):
        self.data = data
        self.corpus = data.corpus
        self.seedwords = data.seedwords
        self.labels = data.labels
        self.num_seeds = {label: len(lst) for label, lst in self.data.seedwords.items()}
    
    def fit(self):
        """
        No fitting/training needed for TF-IDF.
        """
        pass

    def predict(self, agg='mean'):
        """
        Find TF-IDF value and predict class label.
        """
        try:
            # check agg input
            if agg not in ['mean', 'mean_sq', 'max']:
                print('Warning: Unrecognized aggregate, defaulted to mean.')
                agg = 'mean'

            # define tfidf sklearn object
            tfidf_vectorizer = TfidfVectorizer(vocabulary=[
                w for lst in self.seedwords.values() for w in lst
            ])

            # find tfidf values
            tfidf_values = tfidf_vectorizer.fit_transform(
                [' '.join(doc) for doc in self.corpus]
            ).toarray()

            # compute relevance for each label
            split_idx = np.cumsum(list(self.num_seeds.values())[:-1])
            doc_relevance = np.column_stack([
                self.aggregate(agg, axis=1)(label_values)
                for label_values in np.split(tfidf_values, split_idx, axis=1)
            ])

            # make predictions
            return self.data.pred_to_label(doc_relevance.argmax(axis=1))
        except Exception as e:
            print(e)
            

    def aggregate(self, mode, axis):
        """
        Aggregate scalars for revelance value.
        """
        func = {
            'mean': lambda arr: arr.mean(axis=axis),
            'mean_sq': lambda arr: (arr ** 2).mean(axis=axis),
            'max': lambda arr: arr.max(axis=axis)
        }
        return func[mode]

    @staticmethod
    def run(d: Data, **config) -> dict:
        """
        Perform a TF-IDF Run.
        """
        print('Running TF-IDF Model')
        result = dict(config)
        result.update({'stem': d.stem, 'data': d.name})

        if check_config(config, 'agg'):
            aggs = ['mean', 'mean_sq', 'max']
        else:
            aggs = config['agg']
        
        tfidf = Tfidf_Model(data=d)
        for agg in aggs:
            print(f'{d.name}, {"Stemmed" if d.stem else "Not Stemmed"}, {agg}')
            tfidf_pred = tfidf.predict(agg=agg)
            metric = evaluate(tfidf.data.true_labels, tfidf_pred)
            print(metric)
            print()
            result.update({agg: metric})

        return result



class Word2Vec_Model:
    """
    Baseline model that utilizes the Word2Vec algorithm to learn
    vector representations for words.
    """

    def __init__(self, data: Data):
        self.data = data
        self.corpus = data.corpus
        self.seedwords = data.seedwords
        self.labels = data.labels
    
    def fit(self, **config):
        """
        Train the Word2Vec model using config in model.
        """
        self.model = Word2Vec(self.corpus, **config)
    
    def load_model(self, path):
        self.model = Word2Vec.load(path)
    
    def predict(self):
        self.check_fitted()

        # find representations
        self.doc_rep = np.empty((len(self.corpus), self.model.vector_size))
        self.label_rep = np.empty((len(self.labels), self.model.vector_size))

        for i, doc in tqdm(enumerate(self.corpus), "Finding Document Representations"):
            self.doc_rep[i] = np.vstack([
                self.model.wv[w] 
                for w in doc 
                if w in self.model.wv
            ]).mean(axis=0)
    
        for i, seeds in tqdm(enumerate(self.seedwords.values()), "Finding Label Representations"):
            self.label_rep[i] = np.vstack([
                self.model.wv[s] 
                for s in seeds 
                if s in self.model.wv
            ]).mean(axis=0)
        
        # find relevance
        self.relevance = np.empty((len(self.corpus), len(self.labels)))

        for i, doc in tqdm(enumerate(self.corpus), "Finding Similarity"):
            for j, label in enumerate(self.seedwords):
                self.relevance[i][j] = cosine_similarity(self.doc_rep[i], self.label_rep[j])
        
        # predict
        return self.data.pred_to_label(self.relevance.argmax(axis=1))
    
    def check_fitted(self):
        if not hasattr(self, "model"):
            raise NotFittedError('Please fit or load the model first.')

    def find_similar(self, word):
        return (
            self.model.wv.most_similar(positive=word)
            if word in self.model.wv
            else None
        )

    def display_vector_pca(self, **reps):
        """
        Plot the 2-dimensional vector representation of the given content.
        The method takes in any number of keyword arguments, with each argument
        either a word or a vector representation as value and their name as key.
        """
        all_reps = {
            k: w if isinstance(w, np.ndarray)
            else self.model.wv[w]
            for k, w in reps.items()
            if isinstance(w, np.ndarray) or w in self.model.wv
        }

        if not all_reps:
            return

        rep_2d = PCA(n_components=2).fit_transform(np.vstack(list(all_reps.values())))
        labels = list(all_reps.keys())

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))        
        ax.scatter(rep_2d[:, 0], rep_2d[:, 1], edgecolors='k', c='r')
        for word, (x, y) in zip(labels, rep_2d):
            ax.text(x + 0.05, y + 0.05, word)
        ax.set_title('Document and Labels Representation in 2D')

        return fig

    @staticmethod
    def run(d: Data, **config) -> dict:
        """
        Perform a Word2Vec Run.
        """
        print('Running Word2Vec Model')
        result = dict(config)
        result.update({'stem': d.stem, 'data': d.name})

        if check_config(config, 'model_params'):
            model_config = dict()
        else:
            model_config = config['model_params']
        
        w2v = Word2Vec_Model(d)

        # load or train model
        if not check_config(config, 'model_inpath'):
            w2v.load_model(config['model_inpath'])
        else:
            w2v.fit(**model_config)
        
        # evaluate
        w2v_pred = w2v.predict()
        metric = evaluate(w2v.data.true_labels, w2v_pred)
        for k, v in metric.items():
            print(f"{k}: {v}")
        result.update(metric)
        print()

        # output model
        if not check_config(config, 'model_outpath'):
            w2v.model.save(config['model_outpath'])

        return result, w2v.model
