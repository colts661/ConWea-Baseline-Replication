# ConWea Baseline Replication
In this project, we aim to replicate two baseline results from the [ConWea](#conwea) paper: IR-TF-IDF and [Word2Vec](#word2vec).



### Environment
- **DSMLP**: Since the data for this project is large, please run DSMLP launch script using a larger RAM. The suggested command is `launch.sh -i yaw006/conwea-rep:submit -m 16`. Please **DO NOT** use the default, otherwise Python processes might be killed halfway.
- Other options:
  - Option 1: Run the docker container: `docker run yaw006/conwea-rep:submit`;
  - Option 2: Install all required packages in `requirements.txt`.

### Data
#### Data Information
- Two datasets used in the report can be found on [Google Drive](https://drive.google.com/drive/folders/1AOnhV4g0U7GIDTek4ghDQ6EiwgQDXiW1?usp=sharing): `nyt` and `20news`.
- Each dataset contains both the coarse and fine-grained versions, so the data `-d` tag currently supports `nyt/coarse, nyt/fine, 20news/coarse, 20news/fine`.

#### Get Data
- **DSMLP**/Linux: Run the commands below for the desired data:
  - NYT:
  ```
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1V21UpNElA3hARO0QUEN4aNBiGaio5bJ7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1V21UpNElA3hARO0QUEN4aNBiGaio5bJ7" -O 'data/raw/nyt.zip' && rm -rf /tmp/cookies.txt
  mkdir data/processed/nyt
  mkdir data/processed/nyt/coarse
  mkdir data/processed/nyt/fine
  cd data/raw
  unzip -o nyt.zip
  rm nyt.zip
  cd ../..
  ```
  - 20News:
  ```
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1V21UpNElA3hARO0QUEN4aNBiGaio5bJ7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1IqPdhBh_wi79p1DhM_QNnBcnQUk0DXml" -O 'data/raw/20news.zip' && rm -rf /tmp/cookies.txt
  mkdir data/processed/20news
  mkdir data/processed/20news/coarse
  mkdir data/processed/20news/fine
  cd data/raw
  unzip -o 20news.zip
  rm 20news.zip
  cd ../..
  ```
- Under Linux command line, for other Google Drive zips, 
  - Follow the `wget` [tutorial](https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99)
    - Find the Large File section (highlighted code section towards the end)
    - Paste the `<FILEID>` from the `zip` file sharing link found on Google Drive
    - Change the `<FILENAME>` to your data title
  - Run `cd <dir>` to change directory into the data directory
  - Run `unzip -o <zip name>` to unzip the data
  - Run `rm <zip name>` to avoid storing too many objects in the container
  - Run `cd <root>` to change directory back to your working directory
  - Run `mkdir <data>` to create the processed data directory
- Under non-command line, go to the Google Drive link, download the zip directly, place the files according to the requirements in the **Data Format** section, and manually created the directory needed for processed files. See the **File Outline** section for example.

#### Data Format
- Raw Data: Each dataset should contain following files, and placed in `data/raw/`:
  - **DataFrame pickle file**
    - Example: ```data/raw/nyt/coarse/df.pkl```
      - This dataset should contain two columns named ```sentence```, ```label```
      - ```sentence``` contains text and ```label``` contains its corresponding label.
      - Must be named as ```df.pkl```
  - **Seed Words JSON file**
    - Example: ```data/raw/nyt/coarse/seedwords.json```
      - This json file contains seed words list for each label.
      - Must be named as ```seedwords.json```
- Processed Data: 
  - The corpus will be processed after the first run, and processed files will be placed in `data/processed`.
  - The processed file will be directly loaded for subsequent runs.

### Commands
The main script is located in the root directory. It supports 3 targets:
- `test`: Run the test data. All other flags are ignored.
- `experiment` (or `exp`) [default]: Perform one run using configuration from `config/exp_<model>_config.json`.
- `hyperparameter` (or `ht`): Perform hyperparameter search using parameters from `config/ht_<model>_config.json`.

The two models supported are:
- `tfidf`: Run the TF-IDF model
- `w2v`: Run the Word2Vec model

The full command is:

```
python run.py [-h] test [-d DATA] [-m MODEL [...]] [-s] [-o] [-p]

optional arguments:

  -h, --help             show help message and exit
  -d, --data DATA        data path, default is `nyt/coarse`
  -m, --model MODEL      model. Supports `tfidf` and/or `w2v`. Need at least 1, default is both
  -s, --stem             whether to stem the corpus. Only used in `experiment` target
  -o, --output           whether to write result. Only used in `experiment` target
  -p, --plot             visualize document length distribution. Only used in `experiment` target
```

### Code File Outline
```
ConWea-Baseline-Replication/
├── run.py                               <- main script
├── config/                              <- model configuration JSON files
│   ├── exp_tfidf_config.json
│   ├── ht_tfidf_config.json
│   ├── test_w2v_config.json
│   └── ...
├── data/                                <- all data files
│   ├── raw                              <- raw files (after download)
│   │   ├── nyt
│   │   |   ├── coarse                   <- dataset: nyt/coarse
│   |   │   |   ├── df.pkl               <- required DataFrame pickle file
│   |   │   |   └── seedwords.json       <- required seedwords JSON file
│   |   │   └── fine
│   |   └── ...
│   └── processed                        <- processed files (after preprocessing)
|       ├── nyt
|       |   ├── coarse
|       │   |   ├── corpus_stem.pkl      <- preprocessed corpus (generated by code)
|       │   |   └── labels.pkl           <- preprocessed labels (generated by code)
|       │   └── fine
|       └── ...
├── models/                          <- saved models
│   ├── new_ht                       <- newly saved models from tuning
│   ├── others                       <- newly saved models from experiments
│   └── tuned                        <- originally tuned models
├── plots/                           <- visualizations
├── results/                         <- code run results
│   ├── tfidf_ht.json
│   ├── tfidf_runs.json
|   └── ...
├── src/                             <- source code library
│   ├── data.py
│   ├── models.py
│   └── util.py
└── test/                            <- test target data
    └── testdata/
        ├── df.pkl
        ├── seedwords.json
        └── ...
```

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