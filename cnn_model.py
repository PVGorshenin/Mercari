# Based on https://www.kaggle.com/knowledgegrappler/a-simple-nn-solution-with-keras-0-48611-pl by noobhound

import numpy as np
import pandas as pd

import tensorflow as tf
import random as rn
import os
import time
from keras import backend as K
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from subprocess import check_output
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam,  Nadam
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation,
concatenate, Conv1D, Embedding, Flatten, BatchNormalization,
GlobalMaxPooling1D, GlobalAveragePooling1D


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)
print(check_output(["ls", "../input"]).decode("utf8"))

print("Loading data...")
train = pd.read_table("../input/train.tsv")
test = pd.read_table("../input/test.tsv")

test_id = test.test_id
print(train.shape,
      test.shape)

MAX_WORDS = 50000
BATCH_SIZE = 50000


def rmsle_cust(y_true, y_pred):
    return K.sqrt(K.mean(K.square(K.log(y_true + 1) - K.log(y_pred + 1)),  axis=-1))


def rmsle(y_true, y_hat):
    return(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_hat))**2)))


def handle_missing(dataset):
    ''' Filling missing values with string '''

    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return (dataset)


def get_model(X_train, max_vocabulary):

    dr_r = 0.1
    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    cat_1 = Input(shape=[1], name="cat_1")
    cat_2 = Input(shape=[1], name="cat_2")
    cat_3 = Input(shape=[1], name="cat_3")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    # Embeddings layers
    emb_name = Embedding(max_vocabulary['MAX_TEXT'], 50)(name)
    emb_item_desc = Embedding(max_vocabulary['MAX_TEXT'], 50)(item_desc)
    emb_brand_name = Embedding(max_vocabulary['MAX_BRAND'], 10)(brand_name)
    emb_cat_1 = Embedding(max_vocabulary['MAX_CAT_1'], 10)(cat_1)
    emb_cat_2 = Embedding(max_vocabulary['MAX_CAT_2'], 10)(cat_2)
    emb_cat_3 = Embedding(max_vocabulary['MAX_CAT_3'], 10)(cat_3)
    emb_item_condition = Embedding(max_vocabulary['MAX_CONDITION'], 5)(item_condition)

    # сnn layer
    cnn_layer1_3 = Conv1D(filters=20, kernel_size=3, activation='relu')(emb_item_desc)
    cnn_layer2_3 = Conv1D(filters=20, kernel_size=3, activation='relu')(emb_name)
    cnn_layer1_3 = GlobalAveragePooling1D()(cnn_layer1_3)
    cnn_layer2_3 = GlobalAveragePooling1D()(cnn_layer2_3)

    # main layer
    main_l = concatenate([
        Flatten()(emb_brand_name),
        Flatten()(emb_cat_1),
        Flatten()(emb_cat_2),
        Flatten()(emb_cat_3),
        Flatten()(emb_item_condition),
        cnn_layer1_3,
        cnn_layer2_3,
        num_vars
    ])

    main_l = Dropout(dr_r + 0.1)(Dense(512, activation='relu')(main_l))
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(0.05)(Dense(256, activation='relu')(main_l))
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(0.05)(Dense(128, activation='relu')(main_l))
    main_l = Dropout(0.)(Dense(20, activation='relu')(main_l))

    # output
    output = Dense(1, activation="linear")(main_l)

    # model
    model = Model([name, item_desc, brand_name, cat_1, cat_2, cat_3,
                  item_condition, num_vars], output)
    optimizer = Adam()
    model.compile(loss='mean_squared_logarithmic_error', optimizer=optimizer, metrics=[rmsle_cust])
    return model


def get_keras_data(dataset, max_vocabulary):
    ''' Form input data in Keras format '''

    print(dataset.name_seq.shape[0])
    X = {
        'name': pad_sequences(dataset.name_seq, maxlen=max_vocabulary['MAX_NAME_SEQ']),
        'item_desc': pad_sequences(dataset.item_seq, maxlen=max_vocabulary['MAX_ITEM_DESC_SEQ']),
        'brand_name': np.array(dataset.brand_name),
        'cat_1': np.array(dataset.cat_1),
        'cat_2': np.array(dataset.cat_2),
        'cat_3': np.array(dataset.cat_3),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping"]])
    }
    return X

train = train.loc[train.price != 0].reset_index(drop=True)   
train_shape = train.shape[0]
train = handle_missing(train)
test = handle_missing(test)

raw_text = np.hstack([train.item_description.str.lower(), train.name.str.lower()])

tok = Tokenizer(filters='!#()*,-./:;<=>?[\\]^_`{|}~\t\n', num_words=MAX_WORDS)
tok.fit_on_texts(raw_text)

train["item_seq"] = tok.texts_to_sequences(train.item_description.str.lower())
test["item_seq"] = tok.texts_to_sequences(test.item_description.str.lower())
train['name_seq'] = tok.texts_to_sequences(train.name.str.lower())
test['name_seq'] = tok.texts_to_sequences(test.name.str.lower())
print('------------------Sequence formed------------------')

category_name_train = train.category_name.map(lambda z: z.split('/'))
category_name_train = pd.DataFrame(list(category_name_train.values))

train['cat_1'] = category_name_train.iloc[:, 0]
train['cat_2'] = category_name_train.iloc[:, 1]
train['cat_3'] = category_name_train.iloc[:, 2]

print(category_name_train[:3])

category_name_test = test.category_name.map(lambda z: z.split('/'))
category_name_test = pd.DataFrame(list(category_name_test.values))

test['cat_1'] = category_name_test.iloc[:, 0]
test['cat_2'] = category_name_test.iloc[:, 1]
test['cat_3'] = category_name_test.iloc[:, 2]

le = LabelEncoder()
le.fit(np.hstack([train.brand_name, test.brand_name]))
train.brand_name = le.transform(train.brand_name)
test.brand_name = le.transform(test.brand_name)

train.cat_2.fillna('missing', inplace=True)
train.cat_3.fillna('missing', inplace=True)
test.cat_2.fillna('missing', inplace=True)
test.cat_3.fillna('missing', inplace=True)

le.fit(np.hstack([train.cat_1, test.cat_1]))
train.cat_1 = le.transform(train.cat_1)
test.cat_1 = le.transform(test.cat_1)

le.fit_transform(np.hstack([train.cat_2, test.cat_2]))
train.cat_2 = le.transform(train.cat_2)
test.cat_2 = le.transform(test.cat_2)

le.fit_transform(np.hstack([train.cat_3, test.cat_3]))
train.cat_3 = le.transform(train.cat_3)
test.cat_3 = le.transform(test.cat_3)

max_vocabulary = {
    'MAX_NAME_SEQ': 10,
    'MAX_ITEM_DESC_SEQ': 75,
    'MAX_TEXT': MAX_WORDS + 1,
    'MAX_CAT_1': np.max([train.cat_1.max(), test.cat_1.max()]) + 1,
    'MAX_CAT_2': np.max([train.cat_2.max(), test.cat_2.max()]) + 1,
    'MAX_CAT_3': np.max([train.cat_3.max(), test.cat_3.max()]) + 1,
    'MAX_BRAND': np.max([train.brand_name.max(), test.brand_name.max()]) + 1,
    'MAX_CONDITION': np.max([train.item_condition_id.max(), test.item_condition_id.max()]) + 1
}

print(max_vocabulary, '\n')
dtrain, dvalid = train_test_split(train, shuffle=False, train_size=0.999)

X_train = get_keras_data(dtrain, max_vocabulary)
X_valid = get_keras_data(dvalid, max_vocabulary)
X_test = get_keras_data(test, max_vocabulary)

model = get_model(X_train, max_vocabulary)

start_time = time.time()
epochs = 5


lr_arr = [0.0015, 0.0011, 0.001, 0.0009, 0.0008]
batch_arr = [3000, 3000, 2500, 2000, 2000]

callbacks = EarlyStopping(monitor='val_loss', min_delta=0.03, patience=10, verbose=0, mode='auto')
preds = np.zeros([X_test['brand_name'].shape[0], 3])
for i_epoch in range(epochs):
    X_train = get_keras_data(dtrain.sample(frac=0.85, random_state=i_epoch), max_vocabulary)
    model.fit(X_train, dtrain.price.sample(frac=0.85, random_state=i_epoch),
              epochs=1, batch_size=batch_arr[i_epoch],
              validation_data=(X_valid, dvalid.price),
              verbose=2, callbacks=[callbacks])
    if (i_epoch > 1):
       preds[:, i_epoch - 3] = model.predict(X_test, batch_size=BATCH_SIZE).flatten()           
print(time.time() - start_time)

pred_NN = 0.2*preds[:, 0] + 0.4*preds[:, 1] + 0.4*preds[:, 2]
pred_NN[pred_NN < 3] = 3
print(pred_NN[:10], preds[:10, :], pred_NN.shape)
