"""
Utility Functions for Replication
"""

from data import *
from sklearn.metrics import f1_score
from scipy.spatial import distance
import json
from datetime import datetime

### config files
def load_config(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return dict()
        
def check_config(config, key):
    return (key not in config) or (not config[key])

def write_result(path, **config):
    t = datetime.now().strftime("%D %H:%M:%S")
    if os.path.exists(path):
        with open(path, 'r') as f:
            content = json.load(f)
            content[t] = config

        with open(path, 'w') as f:
            json.dump(content, f)
    else:
        with open(path, 'w') as f:
            json.dump({t: config}, f)

def flatten_dict(d):
    result = dict()
    for key, value in d.items():
        if isinstance(value, dict):
            for sub_key, sub_values in flatten_dict(value).items():
                result[f'{key}_{sub_key}'] = sub_values
        else:
            result[key] = value
    return result

def combine_dict(d, one_d):
    for key, value in one_d.items():
        if key not in d:
            d[key] = [value]
        else:
            d[key].append(value)

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
