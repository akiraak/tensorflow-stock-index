# -*- coding: utf-8 -*-
import os
import pandas as pd


class FeedCache(object):
    FILE_DIR = 'cache'

    def __init__(self, code, label):
        self.code = code
        self.file_path = os.path.join(self.FILE_DIR, '{}_{}.csv'.format(code, label))
        print self.file_path

    def is_exist(self):
        return os.path.exists(self.file_path)

    def load(self):
        return pd.read_csv(self.file_path, index_col='Date').sort_index()

    def save(self, pandas):
        if not os.path.exists(self.FILE_DIR):
            os.mkdir(self.FILE_DIR)
        pandas.to_csv(self.file_path)
