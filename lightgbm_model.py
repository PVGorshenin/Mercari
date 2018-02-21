import lightgbm as lgb
from time import time
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import gc
from sklear

y_train = np.log1p(train["price"])
df = pd.concat([train,test], 0)
nrow_train = train.shape[0]


df['name'] = df['name'].fillna('missing').astype(str)
df['category_name'] = df['category_name'].fillna('Other').astype(str)
df['brand_name'] = df['brand_name'].fillna('missing').astype(str)
df['shipping'] = df['shipping'].astype(str)
df['item_condition_id'] = df['item_condition_id'].astype(str)
df['item_description'] = df['item_description'].fillna('None')

train = df.iloc[:nrow_train]
test = df.iloc[nrow_train:]

default_preprocessor = CountVectorizer().build_preprocessor()
def build_preprocessor(field):
    field_idx = list(train.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])

vectorizer_LG = FeatureUnion([
    ('name', CountVectorizer(
        max_features=50000,
        preprocessor=build_preprocessor('name'))),
    ('category_name', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('category_name'))),
    ('brand_name', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('brand_name'))),
    ('shipping', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('shipping'))),
    ('item_condition_id', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('item_condition_id'))),
    ('item_description', TfidfVectorizer(
        max_features=100000,
        preprocessor=build_preprocessor('item_description'))),
])

dtrain, dvalid = train_test_split(train, shuffle = False, train_size=0.999)
X_train = vectorizer_LG.fit_transform(dtrain.values)
X_valid= vectorizer_LG.transform(dvalid.values)
X_test = vectorizer_LG.transform(test.values)
print('Made data', X_train.shape[0], X_valid.shape[0])
lgb_train = lgb.Dataset(X_train, label=y_train[:X_train.shape[0]])
lgb_val = lgb.Dataset(X_valid, label=y_train[X_train.shape[0]:])

params = {
        'learning_rate': 0.45,
        'application': 'regression',
        'max_depth': 7,
        'num_leaves': 31,        
        'metric': 'RMSE',        
        'bagging_fraction': 0.8,
        'feature_fraction': 0.9,
         'num_thread':4
    }
    
model_LGB = lgb.train(params, train_set=lgb_train, num_boost_round=1600, verbose_eval=100, valid_sets = lgb_val)   
pred_LGB = np.expm1(model_LGB.predict(X_test))
pred_LGB[pred_LGB <3] = 3
print('preg_LGB', pred_LGB[:10])
