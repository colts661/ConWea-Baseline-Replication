# ConWea Baseline Replication
In this project, we aim to replicate two baseline results from the [ConWea](#conwea) paper: IR-TF-IDF and [Word2Vec](#word2vec).

### Load Environment
- Option 1: Run the docker container: `docker run -p 8888:8888 -e DOCKER_STACKS_JUPYTER_CMD=notebook yaw006/conwea-rep`;
- Option 2: Build your own Docker Image using `docker build -t <image-fullname> .`;
- Option 3: Install all required packages in `requirements.txt`.

### Data
- Each dataset should contain following files, and placed in `data/`:
  - **DataFrame pickle file**
    - Example: ```data/nyt/coarse/df.pkl```
      - This dataset should contain two columns named ```sentence```, ```label```
      - ```sentence``` contains text and ```label``` contains its corresponding label.
      - Must be named as ```df.pkl```
  - **Seed Words Json file**
    - Example: ```data/nyt/coarse/seedwords.json```
      - This json file contains seed words list for each label.
      - Must be named as ```seedwords.json```
- Accepted dataset names: `nyt/coarse, nyt/fine, 20news/coarse, 20news/fine`

### Commands
The two models have separate scripts, please enter one of the accepted dataset names above. Both runnable scripts are in the `run` directory.
- IR-TF-IDF:
    1. Run `python run/run_tfidf.py <dataset>`;
    2. Follow the instructions.
- Word2Vec:
    1. Add your model configurations in `w2v_config.json`, the sample corresponds to the best `nyt/coarse` setting. Refer to the accepted parameters on [Gensim Documentation](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec);
    2. Run `python run/run_word2vec.py <dataset>`;
    3. Follow the instructions.
  
  **Note**: For DSMLP graders, the default runnable script in `submission.json` is `run/run_tfidf.py`, paired with the `nyt/coarse` dataset. Please manually change to `python run/run_word2vec.py <dataset>` to run the Word2Vec script.

---
### Citations
#### ConWea
```
@inproceedings{mekala-shang-2020-contextualized,
    title = "Contextualized Weak Supervision for Text Classification",
    author = "Mekala, Dheeraj  and
      Shang, Jingbo",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.30",
    pages = "323--333",
    abstract = "Weakly supervised text classification based on a few user-provided seed words has recently attracted much attention from researchers. Existing methods mainly generate pseudo-labels in a context-free manner (e.g., string matching), therefore, the ambiguous, context-dependent nature of human language has been long overlooked. In this paper, we propose a novel framework ConWea, providing contextualized weak supervision for text classification. Specifically, we leverage contextualized representations of word occurrences and seed word information to automatically differentiate multiple interpretations of the same word, and thus create a contextualized corpus. This contextualized corpus is further utilized to train the classifier and expand seed words in an iterative manner. This process not only adds new contextualized, highly label-indicative keywords but also disambiguates initial seed words, making our weak supervision fully contextualized. Extensive experiments and case studies on real-world datasets demonstrate the necessity and significant advantages of using contextualized weak supervision, especially when the class labels are fine-grained.",
}
```

#### Word2Vec
```
    @article{word2vec,
    title={Efficient estimation of word representations in vector space},
    author={Mikolov, Tomas and Chen, Kai and Corrado, Greg and Dean, Jeffrey},
    journal={arXiv preprint arXiv:1301.3781},
    year={2013}
    }
```