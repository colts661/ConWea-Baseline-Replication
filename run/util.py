"""
Utility Functions for Replication
"""

from data import *
import numpy as np
from sklearn.metrics import f1_score
from scipy.spatial import distance
import matplotlib.pyplot as plt

### Metrics
def accuracy(label, pred):
    return (label == pred).mean()

def macro_f1(label, pred):
    return f1_score(label, pred, average='macro')

def micro_f1(label, pred):
    return f1_score(label, pred, average='micro')

def evaluate(label, pred, methods=['micro_f1', 'macro_f1']):
    func = {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1
    }
    return {m: func[m](label, pred) for m in methods}

### Similarity
def cosine_similarity(u, v):
    return 1 - distance.cosine(u, v)

### General Plotting
def plot_distribution(data: Data):
    # visualize lengths
    df_len = data.df.assign(length=data.df.sentence.apply(lambda s: len(s)))
    bins = np.linspace(df_len.length.min(), df_len.length.max(), 100)

    # overlay histograms
    for name, group_df in df_len.groupby('label'):
        plt.hist(group_df.length, bins, alpha=0.5, label=name)

    plt.legend(loc='upper right')
    plt.title(f"Distribution of Document Lengths: {data.name.replace('/', '_')}")
