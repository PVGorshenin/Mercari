# Based on brilliant https://www.kaggle.com/rumbok/ridge-lb-0-41944 by Serg Lavrikov
# Added test prediction by batches
# Decreased n_gram to (1,2) for name and item_description vectorizers
# Made fit on train, transform on train and test separate instead fit_transform on train + test

import multiprocessing as mp
import pandas as pd
from time import time
from scipy.sparse import csr_matrix
import os
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import gc
from sklearn.base import BaseEstimator, TransformerMixin
import re
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['JOBLIB_START_METHOD'] = 'forkserver'

train = pd.read_table(os.path.join('../input/train.tsv'),
                           engine='c',
                      dtype={'item_condition_id': 'category',
                                  'shipping': 'category'}
                           )
test = pd.read_table(os.path.join('../input/test.tsv'),
                          engine='c' ,
                     dtype={'item_condition_id': 'category',
                                  'shipping': 'category'}
                    )
                    


def dameraulevenshtein(seq1, seq2):
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        twoago, oneago, thisrow = (oneago, thisrow, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


class SymSpell:
    def __init__(self, max_edit_distance=3, verbose=0):
        self.max_edit_distance = max_edit_distance
        self.verbose = verbose

        self.dictionary = {}
        self.longest_word_length = 0

    def get_deletes_list(self, w):
        """given a word, derive strings with up to max_edit_distance characters
           deleted"""

        deletes = []
        queue = [w]
        for d in range(self.max_edit_distance):
            temp_queue = []
            for word in queue:
                if len(word) > 1:
                    for c in range(len(word)):  # character index
                        word_minus_c = word[:c] + word[c + 1:]
                        if word_minus_c not in deletes:
                            deletes.append(word_minus_c)
                        if word_minus_c not in temp_queue:
                            temp_queue.append(word_minus_c)
            queue = temp_queue

        return deletes

    def create_dictionary_entry(self, w):
        '''add word and its derived deletions to dictionary'''
        new_real_word_added = False
        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = (self.dictionary[w][0], self.dictionary[w][1] + 1)
        else:
            self.dictionary[w] = ([], 1)
            self.longest_word_length = max(self.longest_word_length, len(w))

        if self.dictionary[w][1] == 1:
            new_real_word_added = True
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    # add (correct) word to delete's suggested correction list
                    self.dictionary[item][0].append(w)
                else:
                    # note frequency of word in corpus is not incremented
                    self.dictionary[item] = ([w], 0)

        return new_real_word_added

    def create_dictionary_from_arr(self, arr, token_pattern=r'[a-z]+'):
        total_word_count = 0
        unique_word_count = 0

        for line in arr:
            # separate by words by non-alphabetical characters
            words = re.findall(token_pattern, line.lower())
            for word in words:
                total_word_count += 1
                if self.create_dictionary_entry(word):
                    unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def create_dictionary(self, fname):
        total_word_count = 0
        unique_word_count = 0

        with open(fname) as file:
            for line in file:
                # separate by words by non-alphabetical characters
                words = re.findall('[a-z]+', line.lower())
                for word in words:
                    total_word_count += 1
                    if self.create_dictionary_entry(word):
                        unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def get_suggestions(self, string, silent=False):
        """return list of suggested corrections for potentially incorrectly
           spelled word"""
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                print("no items in dictionary within maximum edit distance")
            return []

        suggest_dict = {}
        min_suggest_len = float('inf')

        queue = [string]
        q_dictionary = {}  # items other than string that we've checked

        while len(queue) > 0:
            q_item = queue[0]  # pop
            queue = queue[1:]

            # early exit
            if ((self.verbose < 2) and (len(suggest_dict) > 0) and
                    ((len(string) - len(q_item)) > min_suggest_len)):
                break

            # process queue item
            if (q_item in self.dictionary) and (q_item not in suggest_dict):
                if self.dictionary[q_item][1] > 0:
                    assert len(string) >= len(q_item)
                    suggest_dict[q_item] = (self.dictionary[q_item][1],
                                            len(string) - len(q_item))
                    # early exit
                    if (self.verbose < 2) and (len(string) == len(q_item)):
                        break
                    elif (len(string) - len(q_item)) < min_suggest_len:
                        min_suggest_len = len(string) - len(q_item)

                for sc_item in self.dictionary[q_item][0]:
                    if sc_item not in suggest_dict:

                        assert len(sc_item) > len(q_item)
                        assert len(q_item) <= len(string)

                        if len(q_item) == len(string):
                            assert q_item == string
                            item_dist = len(sc_item) - len(q_item)


                        assert sc_item != string


                        item_dist = dameraulevenshtein(sc_item, string)

                        if (self.verbose < 2) and (item_dist > min_suggest_len):
                            pass
                        elif item_dist <= self.max_edit_distance:
                            assert sc_item in self.dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (self.dictionary[sc_item][1], item_dist)
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist

                        if self.verbose < 2:
                            suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

            assert len(string) >= len(q_item)

            if (self.verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len):
                pass
            elif (len(string) - len(q_item)) < self.max_edit_distance and len(q_item) > 1:
                for c in range(len(q_item)):  # character index
                    word_minus_c = q_item[:c] + q_item[c + 1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this


        if not silent and self.verbose != 0:
            print("number of possible corrections: %i" % len(suggest_dict))
            print("  edit distance for deletions: %i" % self.max_edit_distance)


        as_list = suggest_dict.items()
        outlist = sorted(as_list, key=lambda x: (x[1][1], -x[1][0]))

        if self.verbose == 0:
            return outlist[0]
        else:
            return outlist

    def best_word(self, s, silent=False):
        try:
            return self.get_suggestions(s, silent)[0]
        except:
            return None


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field, start_time=time()):
        self.field = field
        self.start_time = start_time

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        print(f'[{time()-self.start_time}] select {self.field}')
        dt = dataframe[self.field].dtype
        if is_categorical_dtype(dt):
            return dataframe[self.field].cat.codes[:, None]
        elif is_numeric_dtype(dt):
            return dataframe[self.field][:, None]
        else:
            return dataframe[self.field]


class DropColumnsByDf(BaseEstimator, TransformerMixin):
    def __init__(self, min_df=1, max_df=1.0):
        self.min_df = min_df
        self.max_df = max_df

    def fit(self, X, y=None):
        m = X.tocsc()
        self.nnz_cols = ((m != 0).sum(axis=0) >= self.min_df).A1
        if self.max_df < 1.0:
            max_df = m.shape[0] * self.max_df
            self.nnz_cols = self.nnz_cols & ((m != 0).sum(axis=0) <= max_df).A1
        return self

    def transform(self, X, y=None):
        m = X.tocsc()
        return m[:, self.nnz_cols]


def get_rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))


def split_cat(text):
    try:
        cats = text.split("/")
        return cats[0], cats[1], cats[2], cats[0] + '/' + cats[1]
    except:
        print("no category")
        return 'other', 'other', 'other', 'other/other'


def brands_filling(dataset):
    vc = dataset['brand_name'].value_counts()
    brands = vc[vc > 0].index
    brand_word = r"[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+"

    many_w_brands = brands[brands.str.contains(' ')]
    one_w_brands = brands[~brands.str.contains(' ')]

    ss2 = SymSpell(max_edit_distance=0)
    ss2.create_dictionary_from_arr(many_w_brands, token_pattern=r'.+')

    ss1 = SymSpell(max_edit_distance=0)
    ss1.create_dictionary_from_arr(one_w_brands, token_pattern=r'.+')

    two_words_re = re.compile(r"(?=(\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+))")

    def find_in_str_ss2(row):
        for doc_word in two_words_re.finditer(row):
            print(doc_word)
            suggestion = ss2.best_word(doc_word.group(1), silent=True)
            if suggestion is not None:
                return doc_word.group(1)
        return ''

    def find_in_list_ss1(list):
        for doc_word in list:
            suggestion = ss1.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    def find_in_list_ss2(list):
        for doc_word in list:
            suggestion = ss2.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    print(f"Before empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_name]

    n_desc = dataset[dataset['brand_name'] == '']['item_description'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_desc]

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in n_name]

    desc_lower = dataset[dataset['brand_name'] == '']['item_description'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in desc_lower]

    print(f"After empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    del ss1, ss2
    gc.collect()


def preprocess_regex(dataset, start_time=time()):
    karats_regex = r'(\d)([\s-]?)(karat|karats|carat|carats|kt)([^\w])'
    karats_repl = r'\1k\4'

    unit_regex = r'(\d+)[\s-]([a-z]{2})(\s)'
    unit_repl = r'\1\2\3'

    dataset['name'] = dataset['name'].str.replace(karats_regex, karats_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(karats_regex, karats_repl)
    print(f'[{time() - start_time}] Karats normalized.')

    dataset['name'] = dataset['name'].str.replace(unit_regex, unit_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(unit_regex, unit_repl)
    print(f'[{time() - start_time}] Units glued.')


def preprocess_pandas(merge, start_time=time()):
    #merge: pd.DataFrame = pd.concat([train, test])
    print(merge.shape[0])
    gc.collect()

    merge['has_category'] = (merge['category_name'].notnull()).astype('category')
    print(f'[{time() - start_time}] Has_category filled.')

    merge['category_name'] = merge['category_name'] \
        .fillna('other/other/other') \
        .str.lower() \
        .astype(str)
    merge['general_cat'], merge['subcat_1'], merge['subcat_2'], merge['gen_subcat1'] = \
        zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    print(f'[{time() - start_time}] Split categories completed.')

    merge['has_brand'] = (merge['brand_name'].notnull()).astype('category')
    print(f'[{time() - start_time}] Has_brand filled.')

    merge['gencat_cond'] = merge['general_cat'].map(str) + '_' + merge['item_condition_id'].astype(str)
    merge['subcat_1_cond'] = merge['subcat_1'].map(str) + '_' + merge['item_condition_id'].astype(str)
    merge['subcat_2_cond'] = merge['subcat_2'].map(str) + '_' + merge['item_condition_id'].astype(str)
    print(f'[{time() - start_time}] Categories and item_condition_id concancenated.')

    merge['name'] = merge['name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    merge['brand_name'] = merge['brand_name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    merge['item_description'] = merge['item_description'] \
        .fillna('') \
        .str.lower() \
        .replace(to_replace='No description yet', value='')
    print(f'[{time() - start_time}] Missing filled.')

    preprocess_regex(merge, start_time)

    brands_filling(merge)
    print(f'[{time() - start_time}] Brand name filled.')

    merge['name'] = merge['name'] + ' ' + merge['brand_name']
    print(f'[{time() - start_time}] Name concancenated.')


    return merge


def intersect_drop_columns(train: csr_matrix, valid: csr_matrix, min_df=0):
    t = train.tocsc()
    v = valid.tocsc()
    nnz_train = ((t != 0).sum(axis=0) >= min_df).A1
    nnz_valid = ((v != 0).sum(axis=0) >= min_df).A1
    nnz_cols = nnz_train & nnz_valid
    res = t[:, nnz_cols], v[:, nnz_cols]
    return res

start_time = time()
train = train.loc[train.price !=0].reset_index(drop=True)
y_train = np.log1p(train.price)
dtrain = preprocess_pandas(train, start_time)
start_time = time()
dtest = preprocess_pandas(test, start_time)

meta_params = {'name_ngram': (1, 2),
               'name_max_f': 75000,
               'name_min_df': 10,

               'category_ngram': (2, 3),
               'category_token': '.+',
               'category_min_df': 10,

               'brand_min_df': 10,

               'desc_ngram': (1, 3),
               'desc_max_f': 150000,
               'desc_max_df': 0.5,
               'desc_min_df': 10}

stopwords = frozenset(['the', 'a', 'an', 'is', 'it', 'this', ])

vectorizer_LR = FeatureUnion([
    ('name', Pipeline([
        ('select', ItemSelector('name', start_time=start_time)),
        ('transform', HashingVectorizer(
            ngram_range=(1, 2),
            n_features=2 ** 27,
            norm='l2',
            lowercase=False,
            stop_words=stopwords
        )),
        ('drop_cols', DropColumnsByDf(min_df=2))
    ])),
    ('category_name', Pipeline([
        ('select', ItemSelector('category_name', start_time=start_time)),
        ('transform', HashingVectorizer(
            ngram_range=(1, 1),
            token_pattern='.+',
            tokenizer=split_cat,
            n_features=2 ** 27,
            norm='l2',
            lowercase=False
        )),
        ('drop_cols', DropColumnsByDf(min_df=2))
    ])),
    ('brand_name', Pipeline([
        ('select', ItemSelector('brand_name', start_time=start_time)),
        ('transform', CountVectorizer(
            token_pattern='.+',
            min_df=2,
            lowercase=False
        )),
    ])),
    ('gencat_cond', Pipeline([
        ('select', ItemSelector('gencat_cond', start_time=start_time)),
        ('transform', CountVectorizer(
            token_pattern='.+',
            min_df=2,
            lowercase=False
        )),
    ])),
    ('subcat_1_cond', Pipeline([
        ('select', ItemSelector('subcat_1_cond', start_time=start_time)),
        ('transform', CountVectorizer(
            token_pattern='.+',
            min_df=2,
            lowercase=False
        )),
    ])),
    ('subcat_2_cond', Pipeline([
        ('select', ItemSelector('subcat_2_cond', start_time=start_time)),
        ('transform', CountVectorizer(
            token_pattern='.+',
            min_df=2,
            lowercase=False
        )),
    ])),
    ('has_brand', Pipeline([
        ('select', ItemSelector('has_brand', start_time=start_time)),
        ('ohe', OneHotEncoder())
    ])),
    ('shipping', Pipeline([
        ('select', ItemSelector('shipping', start_time=start_time)),
        ('ohe', OneHotEncoder())
    ])),
    ('item_condition_id', Pipeline([
        ('select', ItemSelector('item_condition_id', start_time=start_time)),
        ('ohe', OneHotEncoder())
    ])),
    ('item_description', Pipeline([
        ('select', ItemSelector('item_description', start_time=start_time)),
        ('hash', HashingVectorizer(
            ngram_range=(1, 2),
            n_features=2 ** 27,
            dtype=np.float32,
            norm='l2',
            lowercase=False,
            stop_words=stopwords
        )),
        ('drop_cols', DropColumnsByDf(min_df=2)),
    ]))
], n_jobs=1)


X_train = vectorizer_LR.fit_transform(dtrain)
print('X_train.vectorized')
X_test_1 = vectorizer_LR.transform(dtest[:int(0.5*dtest.shape[0])])
X_test_2 = vectorizer_LR.transform(dtest[int(0.5*dtest.shape[0]):])
import scipy
X_test = scipy.sparse.vstack([X_test_1, X_test_2])
print('[{time() - start_time}]Merge vectorized')
print('X_test_shape', X_test.shape)

tfidf_transformer = TfidfTransformer()
X_train = tfidf_transformer.fit_transform(X_train)
X_test = tfidf_transformer.transform(X_test)
print('[{time() - start_time}] TF/IDF completed')

#del merge
#del sparse_merge
del vectorizer_LR
del tfidf_transformer
gc.collect()


print('Made data')

model_LR_1 = Ridge(
        solver='auto',
        fit_intercept=True,
        alpha=0.5,
        max_iter=100,
        normalize=False,
        tol=0.05)
        
model_LR_1.fit(X_train, y_train)
pred_LR_1 = np.expm1(model_LR_1.predict(X_test))
pred_LR_1[pred_LR_1 <3] = 3
print('preg_LR', pred_LR_1[:10], pred_LR_1.shape)
