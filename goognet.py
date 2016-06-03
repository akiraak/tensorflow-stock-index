# -*- coding: utf-8 -*-
'''
Code based on:
https://github.com/corrieelston/datalab/blob/master/FinancialTimeSeriesTensorFlow.ipynb
'''
from __future__ import print_function

import datetime
import urllib2
from os import path
import operator as op
from collections import namedtuple
import numpy as np
import pandas as pd
import tensorflow as tf


DAYS_BACK = 3
REMOVE_NIL_DATE = True  # 計算対象の日付に存在しないデータを削除する
FROM_YEAR = '1991'
EXCHANGES_DEFINE = [
    #['DOW', '^DJI'],
    ['FTSE', '^FTSE'],
    ['GDAXI', '^GDAXI'],
    ['HSI', '^HSI'],
    ['N225', '^N225'],
    #['NASDAQ', '^IXIC'],
    ['SP500', '^GSPC'],
    ['SSEC', '000001.SS'],
]
EXCHANGES_LABEL = [exchange[0] for exchange in EXCHANGES_DEFINE]


Dataset = namedtuple(
    'Dataset',
    'training_predictors training_classes test_predictors test_classes')
Environ = namedtuple('Environ', 'sess model actual_classes training_step dataset feature_data')


def setupDateURL(urlBase):
    now = datetime.date.today()
    return urlBase.replace('__FROM_YEAR__', FROM_YEAR)\
            .replace('__TO_MONTH__', str(now.month - 1))\
            .replace('__TO_DAY__', str(now.day))\
            .replace('__TO_YEAR__', str(now.year))


def fetchCSV(fileName, url):
    if path.isfile(fileName):
        print('fetch CSV for local: ' + fileName)
        with open(fileName) as f:
            return f.read()
    else:
        print('fetch CSV for url: ' + url)
        csv = urllib2.urlopen(url).read()
        with open(fileName, 'w') as f:
            f.write(csv)
        return csv


def fetchYahooFinance(name, code):
    fileName = 'index_%s.csv' % name
    url = setupDateURL('http://chart.finance.yahoo.com/table.csv?s=%s&a=0&b=1&c=__FROM_YEAR__&d=__TO_MONTH__&e=__TO_DAY__&f=__TO_YEAR__&g=d&ignore=.csv' % code)
    csv = fetchCSV(fileName, url)


def fetchStockIndexes():
    '''株価指標のデータをダウンロードしファイルに保存
    '''
    for exchange in EXCHANGES_DEFINE:
        fetchYahooFinance(exchange[0], exchange[1])


def load_exchange_dataframes(target_exchange):
    '''EXCHANGESに対応するCSVファイルをPandasのDataFrameとして読み込む。

    Returns:
        {EXCHANGES[n]: pd.DataFrame()}
    '''

    datas = {exchange: load_exchange_dataframe(exchange)
            for exchange in EXCHANGES_LABEL}

    # 計算対象の日付に存在しないデータを削除する
    if REMOVE_NIL_DATE:
        target_indexes = datas[target_exchange].index
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
    for (exchange, _) in EXCHANGES_DEFINE:
        open_column = '{}_OPEN'.format(exchange)
        close_column = '{}_CLOSE'.format(exchange)
        # np.log(当日終値 / 前日終値) で前日からの変化率を算出
        # 前日よりも上がっていればプラス、下がっていればマイナスになる
        log_return_data['{}_CLOSE_RATE'.format(exchange)] = np.log(using_data[close_column]/using_data[close_column].shift())
        # 終値 >= 始値 なら1。それ意外は0
        log_return_data['{}_RESULT'.format(exchange)] = map(int, using_data[close_column] >= using_data[open_column])

    return log_return_data


def build_training_data(log_return_data, target_exchange, max_days_back=DAYS_BACK, use_subset=None):
    '''学習データを作る。分類クラスは、target_exchange指標の終値が前日に比べて上ったか下がったかの2つである。
    また全指標の終値の、当日から数えてmax_days_back日前までを含めて入力データとする。

    Args:
        log_return_data: pd.DataFrame()
        target_exchange: 学習目標とする指標名
        max_days_back: 何日前までの終値を学習データに含めるか
        # 終値 >= 始値 なら1。それ意外は0
        use_subset (float): 短時間で動作を確認したい時用: log_return_dataのうち一部だけを学習データに含める
    Returns:
        pd.DataFrame()
    '''

    columns = ['positive', 'negative']

    # 「上がる」「下がる」の結果を
    log_return_data['positive'] = 0
    positive_indices = op.eq(log_return_data['{}_RESULT'.format(target_exchange)], 1)
    log_return_data.ix[positive_indices, 'positive'] = 1
    log_return_data['negative'] = 0
    negative_indices = op.eq(log_return_data['{}_RESULT'.format(target_exchange)], 0)
    log_return_data.ix[negative_indices, 'negative'] = 1

    num_categories = len(columns)

    # 各指標のカラム名を追加
    for colname, _, _ in iter_exchange_days_back(target_exchange, max_days_back):
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
        for colname, exchange, days_back in iter_exchange_days_back(target_exchange, max_days_back):
            values[colname] = log_return_data['{}_CLOSE_RATE'.format(exchange)].ix[i - days_back]
        training_test_data = training_test_data.append(values, ignore_index=True)

    return num_categories, training_test_data


def iter_exchange_days_back(target_exchange, max_days_back):
    '''指標名、何日前のデータを読むか、カラム名を列挙する。
    '''
    for exchange in EXCHANGES_LABEL:
        # SP500 の結果を予測するのに SP500 の当日の値が含まれてはいけないので１日づらす
        start_days_back = 1 if exchange == target_exchange else 0
        #start_days_back = 1 # N225 で行う場合は全て前日の指標を使うようにする
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
    '''与えられたネットワークの正解率などを出力する。
    '''
    predictions = tf.argmax(model, 1)
    actuals = tf.argmax(actual_classes, 1)

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    tp, tn, fp, fn = session.run(
        [tp_op, tn_op, fp_op, fn_op],
        feed_dict
    )

    tpr = float(tp)/(float(tp) + float(fn))
    fpr = float(fp)/(float(tp) + float(fn))

    accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))

    recall = tpr
    if (float(tp) + float(fp)):
        precision = float(tp)/(float(tp) + float(fp))
        f1_score = (2 * (precision * recall)) / (precision + recall)
    else:
        precision = 0
        f1_score = 0

    print('Precision = ', precision)
    print('Recall = ', recall)
    print('F1 Score = ', f1_score)
    print('Accuracy = ', accuracy)


def smarter_network(dataset):
    '''隠しレイヤー入りのもうちょっと複雑な分類モデルを返す。
    '''
    sess = tf.Session()

    num_predictors = len(dataset.training_predictors.columns)
    num_classes = len(dataset.training_classes.columns)

    feature_data = tf.placeholder("float", [None, num_predictors])
    actual_classes = tf.placeholder("float", [None, num_classes])

    weights1 = tf.Variable(tf.truncated_normal([(DAYS_BACK * len(EXCHANGES_DEFINE)), 50], stddev=0.0001))
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

    for i in range(1, 1 + steps):
        env.sess.run(
            env.training_step,
            feed_dict=feed_dict(env, test=False),
        )
        if i % checkin_interval == 0:
            print(i, env.sess.run(
                accuracy,
                feed_dict=feed_dict(env, test=False),
            ))

    tf_confusion_metrics(env.model, env.actual_classes, env.sess, feed_dict(env, True))


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
    print('株価指標データをダウンロードしcsvファイルに保存')
    fetchStockIndexes()
    print('株価指標データを読み込む')
    all_data  = load_exchange_dataframes(args.target_exchange)
    print('終値を取得')
    using_data = get_using_data(all_data)
    print('データを学習に使える形式に正規化')
    log_return_data = get_log_return_data(using_data)
    print('答と学習データを作る')
    num_categories, training_test_data = build_training_data(
        log_return_data, args.target_exchange,
        use_subset=args.use_subset)
    print('学習データをトレーニング用とテスト用に分割する')
    dataset = split_training_test_data(num_categories, training_test_data)

    print('器械学習のネットワークを作成')
    env = smarter_network(dataset)

    if args.inspect:
        import code
        print('Press Ctrl-d to proceed')
        code.interact(local=locals())

    print('学習')
    train(env, steps=args.steps, checkin_interval=args.checkin)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('target_exchange', choices=EXCHANGES_LABEL)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--checkin', type=int, default=1000)
    parser.add_argument('--use-subset', type=float, default=None)
    parser.add_argument('--inspect', type=bool, default=False)

    args = parser.parse_args()

    main(args)
