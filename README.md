## Neural machine translator

This is a neural machine translation system which is capable of working with English and German languages. You can get the additional information about the data used for this project in `data/LICENSE.md`.

### How to run a project
1. Navigate to the project folder.
2. Set up a virtual environment. See `requirements.txt` for the versions.
3. Activate a virtual environment.
4. Run `python main.py` command.

**N.B.** 
- It is not necessary to train any models - you can find available models' weights in `models/english-to-german/en_de` (English-to-German translator) and `models/german-to-english/de_en` (German-to-English translator).
- If you want to change some parameters and train your own model, then see file-level docstring of `train/en_ge_train.py` and `train/ge_en_train.py`.

### Model's architecture
The model used for the translation is based on the *transformer* architecture proposed in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.

#### Transformer model:
The Transformer model introduces an architecture solely based on attention mechanisms and forgoes the need for recurrence and convolutions. The key components of the Transformer model are:

- **Multi-Head Attention**: This mechanism allows the model to focus on different positions simultaneously, which results in better handling of long-range dependencies.

- **Position-Wise Feed-Forward Networks**: These are fully connected feed-forward networks applied to each position separately and identically.

- **Positional Encoding**: Since the model doesn't have any recurrence or convolution, positional encodings are added to provide the model with some information about the relative or absolute position of the tokens in the sequence.

- **Residual Connections and Layer Normalization**: These are used around each sub-layer, followed by layer normalization.

### Data description
See `data/deu.txt` file for the data used in training. `data/LICENSE.md` contains the information about the source and the license. In short, the *transformer* model was trained using the short phrases and single words passed to the **encoder** with their translations passed to **decoder**.

### Project structure
- `assets/` contains the icons and screenshots used in the project.
- `data/` contains the data `deu.txt` and the license `LICENSE.md` files.
- `models/` contains the weights of 2 models: English-to-German (`english-to-german/en_de/`) & German-to-English (`german-to-english/de_en/`) translators.
- `train/` contains `vocabularies_de_en` and `vocabularies_en_de` with the vocabularies for the models. It also includes `data_preprocess.py` file, which can load the data and initialize vocabularies, `en_ge_train.py` & `ge_en_train.py`that can be started as modules (see their file-level docstrings) in order to train the models.
- `main.py` that you can run to start the project.
- `requirements.txt` contains external python modules used in the project and their versions.