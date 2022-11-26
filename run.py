import sys
from itertools import product
import argparse

sys.path.insert(0, 'src')

from util import *
from data import Data
from models import *


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run script using configuration defined in `config/`"
    )
    parser.add_argument(
        "target", choices=['test', 'experiment', 'exp', 'hyperparameter', 'ht'],
        type=str, default='experiment',
        help="run target. Default experiment; if test is selected, ignore all other flags."
    )
    parser.add_argument("-d", "--data", type=str, help="data path", default='nyt/coarse')
    parser.add_argument(
        "-m", "--model", type=str, nargs='+', choices=['tfidf', 'w2v'], 
        help="models to run", action='extend', default=['tfidf', 'w2v']
    )
    parser.add_argument("-s", "--stem", action='store_true', help="only used in experiments")
    
    parser.add_argument("-o", "--output", action='store_true', help="result filepath, only used in experiments")
    parser.add_argument(
        "-p", "--plot", action='store_true', 
        help="visualize document length distribution, only used in experiments"
    )
    return parser.parse_args()


def test() -> None:
    """
    Run test target
    """
    print('Running Test Data Target:')
    d = Data(
        data_dir='test/', 
        dataset='testdata', 
        stem=False, 
        special_tokens=True
    )
    
    Tfidf_Model.run(d)
    config = load_config('config/test_w2v_config.json')
    Word2Vec_Model.run(d, **config)


def experiment(dataset: str, stem, models: list, viz=False) -> None:
    print('Running Experiment Target:')
    d = Data(
        data_dir='data/', 
        dataset=dataset, 
        stem=stem, 
        special_tokens=True
    )

    if viz:
        d.plot_distribution()

    # run TF-IDF Model
    if 'tfidf' in models:
        config = load_config('config/exp_tfidf_config.json')
        tfidf_result = Tfidf_Model.run(d, **config)
        if args.output:
            write_result('results/tfidf_runs.json', **tfidf_result)
    
    # run Word2Vec Model
    if 'w2v' in models:
        config = load_config('config/exp_w2v_config.json')
        w2v_result, _ = Word2Vec_Model.run(d, **config)
        if args.output:
            write_result('results/w2v_runs.json', **w2v_result)


def tune(dataset: str, models: list) -> None:
    print('Running Hyparameter Target:')

    # run TF-IDF Model
    if 'tfidf' in models:
        config = load_config('config/ht_tfidf_config.json')
        tfidf_result = dict()

        keys, values = zip(*config.items())
        for bundle in product(*values):
            # load
            one_config = dict(zip(keys, bundle))
            d = Data(
                data_dir='data/', 
                dataset=dataset, 
                stem=one_config['stem'],
                special_tokens=True
            )
            one_tfidf_result = flatten_dict(Tfidf_Model.run(d, **one_config))
            combine_dict(tfidf_result, one_tfidf_result)
        
        write_result('results/tfidf_ht.json', **tfidf_result)

    # run Word2Vec Model
    if 'w2v' in models:
        config = load_config('config/ht_w2v_config.json')
        w2v_result = dict()
        best_model, best_macro, best_config = None, 0, None

        keys, values = zip(*config.items())
        for bundle in product(*values):
            # load
            one_config = {'model_params': dict(zip(keys, bundle))}
            d = Data(
                data_dir='data/', 
                dataset=dataset, 
                stem=False if dataset == '20news/fine' else True,
                special_tokens=True
            )
            one_w2v_result, model = Word2Vec_Model.run(d, **one_config)
            one_w2v_result = flatten_dict(one_w2v_result)
            combine_dict(w2v_result, one_w2v_result)
            
            # check best
            if one_w2v_result['macro_f1'] > best_macro:
                best_model, best_macro, best_config = (
                    model, one_w2v_result['macro_f1'], one_config['model_params']
                )
        
        best_str = f'best_{dataset.replace("/", "_")}'
        best_model.save(f'models/new_ht/{best_str}.model')
        with open(f'models/new_ht/{best_str}_config.json', 'w') as f:
            json.dump(best_config, f)
        write_result('results/w2v_ht.json', **w2v_result)
            

if __name__ == "__main__":
    # parse command-line arguments
    args = parse()

    # test target
    if args.target == 'test':
        test()
    
    else:
        # load data
        models = list(set(args.model))

        # experiment target
        if args.target in ['experiment', 'exp']:
            experiment(args.data, args.stem, models, args.plot)

        # hyperparameter tuning target
        elif args.target in ['hyperparameter', 'ht']:
            tune(args.data, models)
            