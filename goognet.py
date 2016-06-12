# -*- coding: utf-8 -*-
'''
Code based on:
https://github.com/corrieelston/datalab/blob/master/FinancialTimeSeriesTensorFlow.ipynb
'''
from __future__ import print_function

import datetime
import urllib2
import math
from os import path
import operator as op
from collections import namedtuple
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from download import YahooCom, YahooJp
from brands import brand_name


DAYS_BACK = 3
REMOVE_NIL_DATE = True  # 計算対象の日付に存在しないデータを削除する


# 銘柄情報の入れ物
class Stock(object):
    def __init__(self, downloadClass, code):
        self.downloadClass = downloadClass
        self.code = code

    @property
    def dataframe(self):
        return self.downloadClass(self.code).dataframe


# 株価指標
stocks_base = {
    'DOW': Stock(YahooCom, '^DJI'),
    'FTSE': Stock(YahooCom, '^FTSE'),
    'GDAXI': Stock(YahooCom, '^GDAXI'),
    'HSI': Stock(YahooCom, '^HSI'),
    'N225': Stock(YahooCom, '^N225'),
    'NASDAQ': Stock(YahooCom, '^IXIC'),
    'SP500': Stock(YahooCom, '^GSPC'),
    'SSEC': Stock(YahooCom, '000001.SS'),
}
stocks = stocks_base

Dataset = namedtuple(
    'Dataset',
    'training_predictors training_classes test_predictors test_classes')
Environ = namedtuple('Environ', 'sess model actual_classes training_step dataset feature_data')


def load_exchange_dataframes(target_brand):
    '''EXCHANGESに対応するCSVファイルをPandasのDataFrameとして読み込む。

    Returns:
        {EXCHANGES[n]: pd.DataFrame()}
    '''

    # 株価を読み込む
    datas = {}
    for (name, stock) in stocks.items():
        datas[name] = stock.dataframe

    # 計算対象の日付に存在しないデータを削除する
    if REMOVE_NIL_DATE:
        target_indexes = datas[target_brand].index
        for (exchange, data) in datas.items():
            for index in data.index:
                if not index in target_indexes:
                    datas[exchange] = datas[exchange].drop(index)

    return datas


def load_exchange_dataframe(exchange):
    '''exchangeに対応するCSVファイルをPandasのDataFrameとして読み込む。

    Args:
        exchange: 指標名
    Returns:
        pd.DataFrame()
    '''
    return pd.read_csv('index_{}.csv'.format(exchange)).set_index('Date').sort_index()


def get_using_data(dataframes):
    '''各指標の必要なカラムをまとめて1つのDataFrameに詰める。

    Args:
        dataframes: {key: pd.DataFrame()}
    Returns:
        pd.DataFrame()
    '''
    using_data = pd.DataFrame()
    for exchange, dataframe in dataframes.items():
        using_data['{}_OPEN'.format(exchange)] = dataframe['Open']
        using_data['{}_CLOSE'.format(exchange)] = dataframe['Close']
    using_data = using_data.fillna(method='ffill')
    return using_data


def get_log_return_data(using_data):
    '''各指標について、終値を1日前との比率の対数をとって正規化する。

    Args:
        using_data: pd.DataFrame()
    Returns:
        pd.DataFrame()
    '''

    log_return_data = pd.DataFrame()
    #for (exchange, _) in EXCHANGES_DEFINE:
    for (name, stock) in stocks.items():
        open_column = '{}_OPEN'.format(name)
        close_column = '{}_CLOSE'.format(name)
        # np.log(当日終値 / 前日終値) で前日からの変化率を算出
        # 前日よりも上がっていればプラス、下がっていればマイナスになる
        log_return_data['{}_CLOSE_RATE'.format(name)] = np.log(using_data[close_column]/using_data[close_column].shift())
        # 終値 >= 始値 なら1。それ意外は0
        log_return_data['{}_RESULT'.format(name)] = map(int, using_data[close_column] >= using_data[open_column])

    return log_return_data


def build_training_data(log_return_data, target_brand, max_days_back=DAYS_BACK, use_subset=None):
    '''学習データを作る。分類クラスは、target_brandの終値が前日に比べて上ったか下がったかの2つである。
    また全指標の終値の、当日から数えてmax_days_back日前までを含めて入力データとする。

    Args:
        log_return_data: pd.DataFrame()
        target_exchange: 学習目標とする銘柄
        max_days_back: 何日前までの終値を学習データに含めるか
        # 終値 >= 始値 なら1。それ意外は0
        use_subset (float): 短時間で動作を確認したい時用: log_return_dataのうち一部だけを学習データに含める
    Returns:
        pd.DataFrame()
    '''

    columns = ['positive', 'negative']

    # 「上がる」「下がる」の結果を
    log_return_data['positive'] = 0
    positive_indices = op.eq(log_return_data['{}_RESULT'.format(target_brand)], 1)
    log_return_data.ix[positive_indices, 'positive'] = 1
    log_return_data['negative'] = 0
    negative_indices = op.eq(log_return_data['{}_RESULT'.format(target_brand)], 0)
    log_return_data.ix[negative_indices, 'negative'] = 1

    num_categories = len(columns)

    # 各指標のカラム名を追加
    for colname, _, _ in iter_exchange_days_back(target_brand, max_days_back):
        columns.append(colname)

    '''
    columns には計算対象の positive, negative と各指標の日数分のラベルが含まれる
    例：[
        'positive',
        'negative',
        'DOW_0',
        'DOW_1',
        'DOW_2',
        'FTSE_0',
        'FTSE_1',
        'FTSE_2',
        'GDAXI_0',
        'GDAXI_1',
        'GDAXI_2',
        'HSI_0',
        'HSI_1',
        'HSI_2',
        'N225_0',
        'N225_1',
        'N225_2',
        'NASDAQ_0',
        'NASDAQ_1',
        'NASDAQ_2',
        'SP500_1',
        'SP500_2',
        'SP500_3',
        'SSEC_0',
        'SSEC_1',
        'SSEC_2'
    ]
    計算対象の SP500 だけ当日のデータを含めたらダメなので1〜3が入る
    '''

    # データ数をもとめる
    max_index = len(log_return_data)
    if use_subset is not None:
        # データを少なくしたいとき
        max_index = int(max_index * use_subset)

    # 学習データを作る
    training_test_data = pd.DataFrame(columns=columns)
    for i in range(max_days_back + 10, max_index):
        # 先頭のデータを含めるとなぜか上手くいかないので max_days_back + 10 で少し省く
        values = {}
        # 「上がる」「下がる」の答を入れる
        values['positive'] = log_return_data['positive'].ix[i]
        values['negative'] = log_return_data['negative'].ix[i]
        # 学習データを入れる
        for colname, exchange, days_back in iter_exchange_days_back(target_brand, max_days_back):
            values[colname] = log_return_data['{}_CLOSE_RATE'.format(exchange)].ix[i - days_back]
        training_test_data = training_test_data.append(values, ignore_index=True)

    return num_categories, training_test_data


def iter_exchange_days_back(target_brand, max_days_back):
    '''指標名、何日前のデータを読むか、カラム名を列挙する。
    '''
    #for exchange in EXCHANGES_LABEL:
    for (exchange, _) in stocks.items():
        # SP500 の結果を予測するのに SP500 の当日の値が含まれてはいけないので１日づらす
        #start_days_back = 1 if exchange == target_brand else 0
        start_days_back = 1 # N225 で行う場合は全て前日の指標を使うようにする
        end_days_back = start_days_back + max_days_back
        for days_back in range(start_days_back, end_days_back):
            colname = '{}_{}'.format(exchange, days_back)
            yield colname, exchange, days_back


def split_training_test_data(num_categories, training_test_data):
    '''学習データをトレーニング用とテスト用に分割する。
    '''
    # 先頭２つより後ろが学習データ
    predictors_tf = training_test_data[training_test_data.columns[num_categories:]]
    # 先頭２つが「上がる」「下がる」の答えデータ
    classes_tf = training_test_data[training_test_data.columns[:num_categories]]

    # 学習用とテスト用のデータサイズを求める
    training_set_size = int(len(training_test_data) * 0.8)
    test_set_size = len(training_test_data) - training_set_size

    # 古いデータ0.8を学習とし、新しいデータ0.2がテストとなる
    return Dataset(
        training_predictors=predictors_tf[:training_set_size],
        training_classes=classes_tf[:training_set_size],
        test_predictors=predictors_tf[training_set_size:],
        test_classes=classes_tf[training_set_size:],
    )


def tf_confusion_metrics(model, actual_classes, session, feed_dict):
    '''与えられたネットワークの評価指標を出力する。
    '''
    # 予測したカテゴリのインデックスを取得するオペレーター
    predictions = tf.argmax(model, 1)
    # 正解カテゴリのインデックスを取得するオペレーター
    # 日数 x 1
    actuals = tf.argmax(actual_classes, 1)

    # 1を詰めたactualsと同じサイズの行列を作る
    ones_like_actuals = tf.ones_like(actuals)
    # 0を詰めたactualsと同じサイズの行列を作る
    zeros_like_actuals = tf.zeros_like(actuals)
    # 1を詰めたpredictionsと同じサイズの行列を作る
    ones_like_predictions = tf.ones_like(predictions)
    # 0を詰めたpredictionsと同じサイズの行列を作る
    zeros_like_predictions = tf.zeros_like(predictions)

    # true-positives: 真陽性の数を数える
    tp_op = tf.reduce_sum(
        tf.cast(
            # 正解と予測、共に1であるか？
            tf.logical_and(
                # 正解のインデックスが1であるか？
                tf.equal(actuals, ones_like_actuals),
                # 予測のインデックスが1であるか？
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    # true-negatives: 真陰性の数を数える
    tn_op = tf.reduce_sum(
        tf.cast(
            # 正解と予測、共に0であるか？
            tf.logical_and(
                # 正解のインデックスが0であるか？
                tf.equal(actuals, zeros_like_actuals),
                # 予測のインデックスが0であるか？
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    # false-positives: 偽陽性の数を数える
    fp_op = tf.reduce_sum(
        tf.cast(
            # 正解は0、予測は1であるか？
            tf.logical_and(
                # 正解のインデックスが0であるか？
                tf.equal(actuals, zeros_like_actuals),
                # 予測のインデックスが1であるか？
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    # false-negatives: 偽陰性の数を数える
    fn_op = tf.reduce_sum(
        tf.cast(
            # 正解は1、予測は0であるか？
            tf.logical_and(
                # 正解のインデックスが1であるか？
                tf.equal(actuals, ones_like_actuals),
                # 予測のインデックスが0であるか？
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    # 実際の値を得る
    tp, tn, fp, fn = session.run(
        [tp_op, tn_op, fp_op, fn_op],
        feed_dict
    )

    # 実際に陽性カテゴリに分類される全ケースに比べ、正しく判定されたものの割合
    tpr = float(tp)/(float(tp) + float(fn))
    # 実際に陽性カテゴリに分類される全ケースに比べ、誤って陽性とされたものの割合
    fpr = float(fp)/(float(tp) + float(fn))

    # 正解度は、全ケースのうち正しく陽性・陰性を予測できたものの割合である
    accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))

    # 再現率は、上で定義したtprと同じ意味
    recall = tpr

    # 陽性と予測したケースが1つでもある場合
    if (float(tp) + float(fp)):
        # 適合率は、陽性と予測したもののうち実際に陽性であるものの割合である
        precision = float(tp)/(float(tp) + float(fp))
        # F1値に意味がある場合
        if (precision + recall) != 0:
            # F1値は、適合率と再現率の調和平均である
            f1_score = (2 * (precision * recall)) / (precision + recall)
        else:
            f1_score = 0
    # 陽性と予測したケースがまったくなかった場合
    else:
        precision = 0
        f1_score = 0

    return {
        # 精度、適合率
        'Precision': precision,
        # 再現率
        'Recall': recall,
        # F1値
        'F1 Score': f1_score,
        # 正確度
        'Accuracy': accuracy
    }


def smarter_network(dataset):
    '''隠しレイヤー入りのもうちょっと複雑な分類モデルを返す。
    '''
    sess = tf.Session()

    num_predictors = len(dataset.training_predictors.columns)
    num_classes = len(dataset.training_classes.columns)

    feature_data = tf.placeholder("float", [None, num_predictors])
    actual_classes = tf.placeholder("float", [None, num_classes])

    weights1 = tf.Variable(tf.truncated_normal([(DAYS_BACK * len(stocks)), 50], stddev=0.0001))
    biases1 = tf.Variable(tf.ones([50]))

    weights2 = tf.Variable(tf.truncated_normal([50, 25], stddev=0.0001))
    biases2 = tf.Variable(tf.ones([25]))

    weights3 = tf.Variable(tf.truncated_normal([25, 2], stddev=0.0001))
    biases3 = tf.Variable(tf.ones([2]))

    # This time we introduce a single hidden layer into our model...
    hidden_layer_1 = tf.nn.relu(tf.matmul(feature_data, weights1) + biases1)
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights2) + biases2)
    model = tf.nn.softmax(tf.matmul(hidden_layer_2, weights3) + biases3)

    cost = -tf.reduce_sum(actual_classes*tf.log(model))

    training_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

    init = tf.initialize_all_variables()
    sess.run(init)

    return Environ(
        sess=sess,
        model=model,
        actual_classes=actual_classes,
        training_step=training_step,
        dataset=dataset,
        feature_data=feature_data,
    )


def train(env, steps=30000, checkin_interval=5000):
    '''学習をsteps回おこなう。
    '''
    correct_prediction = tf.equal(
        tf.argmax(env.model, 1),
        tf.argmax(env.actual_classes, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    bestResult = None
    for i in range(1, 1 + steps):
        env.sess.run(
            env.training_step,
            feed_dict=feed_dict(env, test=False),
        )
        if i % checkin_interval == 0:
            """
            print(i, env.sess.run(
                accuracy,
                feed_dict=feed_dict(env, test=False),
            ))
            """
            result = tf_confusion_metrics(env.model, env.actual_classes, env.sess, feed_dict(env, True))
            if not bestResult:
                bestResult = result
            else:
                bast = math.fabs(bestResult['Accuracy'] - 0.5)
                now = math.fabs(result['Accuracy'] - 0.5)
                print(bast, now)
                if bast < now:
                    bestResult = result

    #bestResult = tf_confusion_metrics(env.model, env.actual_classes, env.sess, feed_dict(env, True))
    return bestResult


def feed_dict(env, test=False):
    '''学習/テストに使うデータを生成する。
    '''
    prefix = 'test' if test else 'training'
    predictors = getattr(env.dataset, '{}_predictors'.format(prefix))
    classes = getattr(env.dataset, '{}_classes'.format(prefix))
    return {
        env.feature_data: predictors.values,
        env.actual_classes: classes.values.reshape(len(classes.values), len(classes.columns))
    }


def main(args):
    #print('株価指標データをダウンロードしcsvファイルに保存')
    #fetchStockIndexes()
    #print('株価指標データを読み込む')
    all_data  = load_exchange_dataframes(args.target_brand)
    #print('終値を取得')
    using_data = get_using_data(all_data)
    #print('データを学習に使える形式に正規化')
    log_return_data = get_log_return_data(using_data)
    #print('答と学習データを作る')
    num_categories, training_test_data = build_training_data(
        log_return_data, args.target_brand,
        use_subset=args.use_subset)
    #print('学習データをトレーニング用とテスト用に分割する')
    dataset = split_training_test_data(num_categories, training_test_data)

    #print('器械学習のネットワークを作成')
    env = smarter_network(dataset)

    if args.inspect:
        import code
        print('Press Ctrl-d to proceed')
        code.interact(local=locals())

    #print('学習')
    return train(env, steps=args.steps, checkin_interval=args.checkin)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target_brand')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--checkin', type=int, default=1000)
    parser.add_argument('--use-subset', type=float, default=None)
    parser.add_argument('--inspect', default=False, action='store_true')

    args = parser.parse_args()

    stocks[args.target_brand] = Stock(YahooJp, args.target_brand)
    result = main(args)
    target_brand_name = brand_name(args.target_brand)
    print('[{}]{}: {}'.format(args.target_brand, target_brand_name, result['Accuracy']))

    with open('results.csv', 'a') as f:
        f.write('{},{},{}\n'.format(args.target_brand, target_brand_name, str(result['Accuracy'])))
