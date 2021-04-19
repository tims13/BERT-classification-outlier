from operator import index
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import trim_string, sentence_token_nltk

des_path = 'sentence/'
data_path = des_path + 'annotation_sentence.xlsx'

data_pos_csv_path = des_path + 'data_pos.csv'
data_neg_csv_path = des_path + 'data_neg.csv'
data_irre_csv_path = des_path + 'data_irre.csv'
data_novel_csv_path = des_path + 'data_novel.csv'
data_need_csv_path = des_path + 'need_all_sentence.csv'

train_test_ratio = 0.90
train_valid_ratio = 0.80

data_paths = [
    data_pos_csv_path,
    data_neg_csv_path,
    data_irre_csv_path,
    data_novel_csv_path
]

headers = [
    ['pos'],
    ['neg'],
    ['irre'],
    ['novel']
]

data = pd.read_excel(data_path)
data = data.iloc[:, 0:4]

for i in range(4):
    data_t = data.iloc[:, i].astype(str)
    data_t = data_t[data_t != 'nan']
    data_t.reset_index(drop=True, inplace=True)
    data_t.to_csv(data_paths[i], header=headers[i], index=0)

data_pos = pd.read_csv(data_pos_csv_path)
data_pos['label'] = 0
data_pos.rename(columns={'pos': 'text'}, inplace=True)
data_pos['text'] = data_pos['text'].apply(trim_string)

data_neg = pd.read_csv(data_neg_csv_path)
data_neg['label'] = 0
data_neg.rename(columns={'neg': 'text'}, inplace=True)
data_neg['text'] = data_neg['text'].apply(trim_string)

data_irre = pd.read_csv(data_irre_csv_path)
data_irre['label'] = 0
data_irre.rename(columns={'irre': 'text'}, inplace=True)
data_irre['text'] = data_irre['text'].apply(trim_string)

data_novel = pd.read_csv(data_novel_csv_path)
data_novel['label'] = 1
data_novel.rename(columns={'novel': 'text'}, inplace=True)
data_novel['text'] = data_novel['text'].apply(trim_string)

data_need = pd.read_csv(data_need_csv_path)
data_need['label'] = 1
data_need['text'] = data_need['text'].apply(trim_string)

# Train - Test
df_pos_full_train, df_pos_test = train_test_split(data_pos, train_size = train_test_ratio, random_state=1)
df_neg_full_train, df_neg_test = train_test_split(data_neg, train_size = train_test_ratio, random_state=1)
df_irre_full_train, df_irre_test = train_test_split(data_irre, train_size = train_test_ratio, random_state=1)
# df_novel_full_train, df_novel_test = train_test_split(data_novel, train_size = train_test_ratio, random_state=1)
df_need_full_train, df_need_test = train_test_split(data_need, train_size = train_test_ratio, random_state=1)

# Train - valid
df_pos_train, df_pos_valid = train_test_split(df_pos_full_train, train_size = train_valid_ratio, random_state=1)
df_neg_train, df_neg_valid = train_test_split(df_neg_full_train, train_size = train_valid_ratio, random_state=1)
df_irre_train, df_irre_valid = train_test_split(df_irre_full_train, train_size = train_valid_ratio, random_state=1)
# df_novel_train, df_novel_valid = train_test_split(df_novel_full_train, train_size = train_valid_ratio, random_state=1)
df_need_train, df_need_valid = train_test_split(df_need_full_train, train_size = train_valid_ratio, random_state=1)

df_train = pd.concat([df_pos_train, df_neg_train, df_irre_train, df_need_train], ignore_index=True, sort=False)
df_valid = pd.concat([df_pos_valid, df_neg_valid, df_irre_valid, df_need_valid], ignore_index=True, sort=False)
df_test = pd.concat([df_pos_test, df_neg_test, df_irre_test, df_need_test, data_novel], ignore_index=True, sort=False)

print('need:', str(data_need.shape[0]))
print('non-need:', str(data_pos.shape[0] + data_neg.shape[0] + data_irre.shape[0]))
print('train-valid-test', str(df_train.shape), str(df_valid.shape), str(df_test.shape))

df_train.to_csv(des_path + 'train.csv', index=False)
df_valid.to_csv(des_path + 'valid.csv', index=False)
df_test.to_csv(des_path + 'test.csv', index=False)

print("Preprocess finished")