import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np

from utils import trim_string
from sklearn.model_selection import train_test_split

des_path = 'laptop/'
needs_file_path = des_path + 'needs_byasin.csv'
needs_preprocessed_path = des_path + 'needs.csv'
review_file_path = des_path + 'amazon_reviews.csv'

novel_file_path = des_path + 'novel_needs.xlsx'
novel_needs_csv_path = des_path + 'novel_needs.csv'

train_test_ratio = 0.90
train_valid_ratio = 0.80

needs = pd.read_csv(needs_file_path, index_col=0)

data_needs = needs.iloc[:,0].astype(str)
data_needs = data_needs[data_needs != 'nan']

for index in range(1, needs.shape[1]):
    temp = needs.iloc[:,index].astype(str)
    temp = temp[temp != 'nan']
    data_needs = pd.concat([data_needs, temp])

data_needs.reset_index(drop=True, inplace=True)
data_needs.to_csv(needs_preprocessed_path, index=0)

data_needs = pd.read_csv(needs_preprocessed_path)

data_needs["label"] = 1
data_needs['0'] = data_needs['0'].apply(trim_string)
data_needs.rename(columns={'0': 'text'}, inplace=True)


data_review = pd.read_csv(review_file_path)
data_review["label"] = 0
data_review['0'] = data_review['0'].apply(trim_string)
data_review.rename(columns={'0': 'text'}, inplace=True)

# Train - Test
df_need_full_train, df_need_test = train_test_split(data_needs, train_size = train_test_ratio, random_state=1)
df_review_full_train, df_review_test = train_test_split(data_review, train_size = train_test_ratio, random_state=1)

# Train - valid
df_need_train, df_need_valid = train_test_split(df_need_full_train, train_size = train_valid_ratio, random_state=1)
df_review_train, df_review_valid = train_test_split(df_review_full_train, train_size = train_valid_ratio, random_state=1)
print("train-valid-test:")
print("need:", df_need_train.shape, df_need_valid.shape, df_need_test.shape)
print("review:", df_review_train.shape, df_review_valid.shape, df_review_test.shape)

df_train = pd.concat([df_need_train, df_review_train], ignore_index=True, sort=False)
df_valid = pd.concat([df_need_valid, df_review_valid], ignore_index=True, sort=False)
df_test = pd.concat([df_need_test, df_review_test], ignore_index=True, sort=False)

df_train.to_csv(des_path + 'train.csv', index=False)
df_valid.to_csv(des_path + 'valid.csv', index=False)
df_test.to_csv(des_path + 'test.csv', index=False)

# preprocess novel data
novel_needs = pd.read_excel(novel_file_path, index_col=0)
novel_needs["label"] = 1
novel_needs.rename(columns={'make-up reviews': 'text'}, inplace=True)
novel_needs.to_csv(novel_needs_csv_path, header=1, index=0)

print("Finished...")
