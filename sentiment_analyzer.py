from keras import layers
import keras
import os
import re
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from keras.models import Sequential
from keras import backend as K
from adam_w import AdamW

GLOVE_DIR = 'glove.6B'
EMBEDDING_DIM = 300

MAX_LEN = 20
TEXT_DATA_DIR = '.'
DELIMITERS = r"[\w']+|[.,!?;$]"
VALIDATION_SPLIT = 0.48

# LABELS = {'neutral': 0, 'bull': 1, 'bear': 2}
# PRED_2_LABEL = {0: 'neutral', 1: 'bull', 2: 'bear'}
LABELS = {'bull': 0, 'bear': 1}
PRED_2_LABEL = {
    0: 'bull', 1: 'bear',
    (1, 0): 'bull', (0, 1): 'bear'
}

def create_y(df):
    titles = df['title'].tolist()
    titles = list([(title.lower()) for title in titles])

    # bull_words = [
    #     'call',
    #     'long', 'all in', 'moon', 'going up', 'rocket', 'buy', 'long term', 'green', 'up', 'bull', 'diamond'
    # ]
    # bear_words = ['put', 'short', 'going down', 'drop', 'bear', 'sell', 'red', 'dump', 'crash']
    bull_words = ['moon', 'rocket', 'diamond', 'up', 'all in', 'green', 'bull', '🚀']
    bear_words = ['crash', 'bear', 'red', 'down', 'dump', 'lose', 'lost', 'loss', 'risk', 'hedge', 'hedgies', 'bad', 'life', 'false', 'inflation', 'porn loss', 'loss porn']

    bull_scores = []
    bear_scores = []
    y = []
    for title in titles:
        bull = False
        bear = False
        for word in bull_words:
            if word in title:
                bull = True
        if re.findall(r'(\b\d{1,4}[c]\b)|(\b\d{1,4}[ ][c]\b)', title):
            bull = True

        for word in bear_words:
            if word in title:
                bear = True
        if re.findall(r'(\b\d{1,4}[p]\b)|(\b\d{1,4}[ ][p]\b)', title):
            bear = True

        if bull == True and bear == True:
            bull_scores.append(0)
            bear_scores.append(0)
            y.append(LABELS['neutral'])
        if bull == False and bear == False:
            bull_scores.append(0)
            bear_scores.append(0)
            y.append(LABELS['neutral'])
        if bull == True and bear == False:
            bull_scores.append(1)
            bear_scores.append(0)
            y.append(LABELS['bull'])
        if bull == False and bear == True:
            bull_scores.append(0)
            bear_scores.append(1)
            y.append(LABELS['bear'])

    new_df = df.copy(deep=True)
    new_df['y'] = y
    new_df['bull score'] = bull_scores
    new_df['bear score'] = bear_scores

    num_samples_per_class = min(sum(bull_scores), sum(bear_scores))

    bull_data = new_df[new_df['bull score'] == 1].iloc[:num_samples_per_class].copy(deep=True)
    bear_data = new_df[new_df['bear score'] == 1].iloc[:num_samples_per_class].copy(deep=True)
    neutral_data = new_df[(new_df['bull score'] == 0) & (new_df['bear score'] == 0)].iloc[:num_samples_per_class].copy(deep=True)

    new_df = pd.concat([bull_data, bear_data, neutral_data])
    new_df = new_df.sample(frac=1)

    return new_df


def get_vocab_len(corpus):
    all_words = []
    for index, row in corpus.iterrows():
        title = row['title'].split(" ")
        for word in title:
            all_words.append(word)

    return len(set(all_words))


def prepare_data(vocab_len, x, y):
    num_validation_samples = int(VALIDATION_SPLIT * len(x))
    x_train = x.iloc[:-num_validation_samples]
    x_dev = x.iloc[-num_validation_samples:]

    tokenizer = Tokenizer(num_words=vocab_len, lower=True)
    tokenizer.fit_on_texts(x_train)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_dev = tokenizer.texts_to_sequences(x_dev)
    x_train = pad_sequences(x_train, padding='post', maxlen=MAX_LEN)
    x_dev = pad_sequences(x_dev, padding='post', maxlen=MAX_LEN)

    labels = to_categorical(np.asarray(y))
    y_train = labels[:-num_validation_samples]
    y_dev = labels[-num_validation_samples:]

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    print('Shape of x_train tensor:', x_train.shape)
    print('Shape of y_train tensor:', y_train.shape)

    print('Shape of x_dev tensor:', x_dev.shape)
    print('Shape of y_dev tensor:', y_dev.shape)

    x_dev_text = x[-num_validation_samples:].tolist()

    return x_train, y_train, x_dev, y_dev, tokenizer, x_dev_text


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def build_net(word_index, optimizer=None):
    def prepare_embeds():
        embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    embedding_matrix = prepare_embeds()

    model = Sequential()
    model.add(layers.Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=False)))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))

    model.summary()

    if optimizer.lower() == 'adam_w':
        optimizer = AdamW(lr=1e-4, model=model, lr_multipliers={'lstm_1': 0.5},
              use_cosine_annealing=True, total_iterations=24)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["accuracy", f1_m, precision_m, recall_m]
    )

    return model


def train(model, epochs, optimizer_label, l_style, x_train, y_train, x_dev, y_dev, ax_metric, ax_loss):
    results = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=64,
        validation_data=(x_dev, y_dev)
    )
    print("Test-Accuracy:", np.mean(results.history['val_accuracy']))
    print("Test-F1:", np.mean(results.history['val_f1_m']))

    if ax_metric is not None:
        ax_metric.plot(results.history['val_f1_m'], label=optimizer_label, linestyle=l_style)

    if ax_loss is not None:
        ax_loss.plot(results.history['val_loss'], label=optimizer_label, linestyle=l_style)
