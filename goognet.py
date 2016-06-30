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
from download import Download, YahooCom, YahooJp
from brands import brand_name


TEST_COUNT = 200        # テスト日数
TRAIN_MIN = 800         # 学習データの最低日数
DAYS_BACK = 3           # 過去何日分を計算に使用するか
STEPS = 10000           # 学習回数
CHECKIN_INTERVAL = 1000 # 学習の途中結果を表示する間隔
REMOVE_NIL_DATE = True  # 計算対象の日付に存在しないデータを削除する


# 銘柄情報の入れ物
class Stock(object):
    def __init__(self, downloadClass, code, start_days_back):
        self.downloadClass = downloadClass
        self.code = code
        self.download = self.downloadClass(self.code, auto_upload=False)
        self.start_days_back = start_days_back

    @property
    def dataframe(self):
        return self.download.dataframe

Dataset = namedtuple(
    'Dataset',
    'training_predictors training_classes test_predictors test_classes')
Environ = namedtuple('Environ', 'sess model actual_classes training_step dataset feature_data')


def load_exchange_dataframes(stocks, target_brand):
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


def get_log_return_data(stocks, using_data):
    '''各指標について、終値を1日前との比率の対数をとって正規化する。

    Args:
        using_data: pd.DataFrame()
    Returns:
        pd.DataFrame()
    '''

    log_return_data = pd.DataFrame()
    for (name, stock) in stocks.items():
        open_column = '{}_OPEN'.format(name)
        close_column = '{}_CLOSE'.format(name)
        # np.log(当日終値 / 前日終値) で前日からの変化率を算出
        # 前日よりも上がっていればプラス、下がっていればマイナスになる
        log_return_data['{}_CLOSE_RATE'.format(name)] = np.log(using_data[close_column]/using_data[close_column].shift())
        # 終値 >= 始値 なら1。それ意外は0
        log_return_data['{}_RESULT'.format(name)] = map(int, using_data[close_column] >= using_data[open_column])

    return log_return_data


def build_training_data(stocks, log_return_data, target_brand, max_days_back=DAYS_BACK):
    '''学習データを作る。分類クラスは、target_brandの終値が前日に比べて上ったか下がったかの2つである。
    また全指標の終値の、当日から数えてmax_days_back日前までを含めて入力データとする。

    Args:
        log_return_data: pd.DataFrame()
        target_exchange: 学習目標とする銘柄
        max_days_back: 何日前までの終値を学習データに含めるか
        # 終値 >= 始値 なら1。それ意外は0
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
    for colname, _, _ in iter_exchange_days_back(stocks, target_brand, max_days_back):
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

    # 学習データを作る
    training_test_data = pd.DataFrame(columns=columns)
    for i in range(max_days_back + 10, max_index):
        # 先頭のデータを含めるとなぜか上手くいかないので max_days_back + 10 で少し省く
        values = {}
        # 「上がる」「下がる」の答を入れる
        values['positive'] = log_return_data['positive'].ix[i]
        values['negative'] = log_return_data['negative'].ix[i]
        # 学習データを入れる
        for colname, exchange, days_back in iter_exchange_days_back(stocks, target_brand, max_days_back):
            values[colname] = log_return_data['{}_CLOSE_RATE'.format(exchange)].ix[i - days_back]
        training_test_data = training_test_data.append(values, ignore_index=True)

    # index（日付ラベル）を引き継ぐ
    training_test_data.index = log_return_data.index[max_days_back + 10: max_index]
    return num_categories, training_test_data


def iter_exchange_days_back(stocks, target_brand, max_days_back):
    '''指標名、何日前のデータを読むか、カラム名を列挙する。
    '''
    for (exchange, stock) in stocks.items():
        end_days_back = stock.start_days_back + max_days_back
        for days_back in range(stock.start_days_back, end_days_back):
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
    test_set_size = TEST_COUNT
    training_set_size = len(training_test_data) - test_set_size

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

    fn_op2 = tf.equal(predictions, ones_like_predictions)
    fn_op3 = tf.equal(predictions, zeros_like_predictions)

    # 実際の値を得る
    tp, tn, fp, fn, fn2, fn3 = session.run(
        [tp_op, tn_op, fp_op, fn_op, fn_op2, fn_op3],
        feed_dict
    )
    #print(tp, tn, fp, fn, fn2, fn3)

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


def smarter_network(stocks, dataset):
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
        sess=sess,                      # tensorflow.python.client.session.Session
        model=model,                    # tensorflow.python.framework.ops.Tensor
        actual_classes=actual_classes,  # tensorflow.python.framework.ops.Tensor
        training_step=training_step,    # tensorflow.python.framework.ops.Operation
        dataset=dataset,
        feature_data=feature_data,      # tensorflow.python.framework.ops.Tensor
    )


def train(env):
    '''学習をsteps回おこなう。
    '''
    correct_prediction = tf.equal(
        tf.argmax(env.model, 1),
        tf.argmax(env.actual_classes, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    bestResult = None
    for i in range(1, 1 + STEPS):
        env.sess.run(
            env.training_step,
            feed_dict=feed_dict(env, test=False),
        )
        if i % CHECKIN_INTERVAL == 0:
            print(i, env.sess.run(
                accuracy,
                feed_dict=feed_dict(env, test=False),
            ))
            """
            # テストデータで正解率を求める
            result = tf_confusion_metrics(env.model, env.actual_classes, env.sess, feed_dict(env, True))
            if not bestResult:
                bestResult = result
            else:
                bast = math.fabs(bestResult['Accuracy'] - 0.5)
                now = math.fabs(result['Accuracy'] - 0.5)
                print(bast, now)
                if bast < now:
                    bestResult = result
            """

    bestResult = tf_confusion_metrics(env.model, env.actual_classes, env.sess, feed_dict(env, True))
    return bestResult


def expect(env, date):
    '''指定された日付の株価を予想する
    Returns:
        Boolean: True=上がる False=下がる
    '''
    prefix = 'test'
    predictors = getattr(env.dataset, '{}_predictors'.format(prefix))
    classes = getattr(env.dataset, '{}_classes'.format(prefix))
    feed_dict = {
        env.feature_data: np.array([predictors.ix[date].values]),
        env.actual_classes: np.array([classes.ix[date].values])
    }
    predictions = tf.argmax(env.model, 1)
    actuals = tf.argmax(env.actual_classes, 1)
    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)
    op = tf.equal(predictions, ones_like_predictions)
    result = env.sess.run(
        op,
        feed_dict
    )
    return result[0]


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


def buy_charge(yen):
    # GOMクリック証券現物手数料
    if yen <= 100000:
        return 95
    elif yen <= 200000:
        return 105
    elif yen <= 500000:
        return 260
    elif yen <= 1000000:
        return 470
    elif yen <= 1500000:
        return 570
    elif yen <= 30000000:
        return 900
    else:
        return 960


# 売買シミュレーション
def gamble(stocks, target_brand, env):
    money = 10000000 # 元金
    positive_correct = 0
    negative_correct = 0
    results = []
    target_prices = stocks[target_brand]
    for date in env.dataset.test_predictors.index:
        # 予測する
        expectation = expect(env, date)
        results.append(expectation)
        if expectation:
            # 上がるので買う
            price = target_prices.download.price(date)
            if price != None:
                open_value = float(price[Download.COL_OPEN])    # 始値
                close_value = float(price[Download.COL_CLOSE])  # 終値
                value = money / open_value                      # 購入可能な株数
                buy_value = int(open_value * value)             # 購入金額
                sell_value = int(close_value * value)           # 売却金額
                # 購入
                money -= buy_value
                # 購入手数料の支払い
                money -= buy_charge(buy_value)
                # 売却
                money += sell_value
                # 売却手数料の支払い
                money -= buy_charge(sell_value)
                # 購入の予想が当たっていたか
                if close_value > open_value:
                    positive_correct += 1
        else:
            # 下がるので買わない
            price = target_prices.download.price(date)
            if price != None:
                open_value = float(price[Download.COL_CLOSE])   # 始値
                close_value = float(price[Download.COL_OPEN])   # 終値
                # 下がるという予想が当たっていたか
                if close_value < open_value:
                    negative_correct += 1


    day_count = len(results)
    # 買い予想日数
    positive_expectation = len([r for r in results if r == True])
    # 売り予想日数
    negative_expectation = day_count - positive_expectation
    # 買い予想の正解日数
    positive_expectation_rate = int(positive_correct / float(positive_expectation) * 100) if positive_expectation != 0 else 0
    # 売り予想の正解日数
    negative_expectation_rate = int(negative_correct / float(negative_expectation) * 100) if negative_expectation != 0 else 0
    print('買い回数 {} 日中 {} 日'.format(day_count, positive_expectation))
    print('買い正解率 {}/{} {}%'.format(positive_correct, positive_expectation, positive_expectation_rate))
    print('売り正解率 {}/{} {}%'.format(negative_correct, negative_expectation, negative_expectation_rate))
    print('売買シミュレーション結果 {:,d}円'.format(money))
    return money, day_count, positive_expectation, positive_correct, negative_expectation, negative_correct


def main(stocks, target_brand, result_file=None):
    target_brand_name = brand_name(target_brand)
    #print('株価指標データを読み込む')
    all_data  = load_exchange_dataframes(stocks, target_brand)
    #print('終値を取得')
    using_data = get_using_data(all_data)
    #print('データを学習に使える形式に正規化')
    log_return_data = get_log_return_data(stocks, using_data)

    #print('答と学習データを作る')
    num_categories, training_test_data = build_training_data(
        stocks, log_return_data, target_brand)
    #print('学習データをトレーニング用とテスト用に分割する')
    dataset = split_training_test_data(num_categories, training_test_data)
    if len(dataset.training_predictors) < TRAIN_MIN:
        print('[{}]{}: 学習データが少なすぎるため計算を中止'.format(target_brand, target_brand_name))
        with open('results.csv', 'a') as f:
            f.write('{},{},ERROR\n'.format(target_brand, target_brand_name))
        return

    #print('器械学習のネットワークを作成')
    env = smarter_network(stocks, dataset)

    #print('学習')
    result = train(env)

    # 購入シュミレーション
    money, day_count, positive_expectation, positive_correct, negative_expectation, negative_correct = gamble(stocks, target_brand, env)

    print('[{}]{}: {}'.format(target_brand, target_brand_name, result['Accuracy']))
    if result_file:
        with open(result_file, 'a') as f:
            f.write('{},{},{},{},{},{},{},{},{}\n'.format(target_brand, target_brand_name, str(result['Accuracy']), money, day_count, positive_expectation, positive_correct, negative_expectation, negative_correct))
    else:
        print(money, day_count, positive_expectation, positive_correct, negative_expectation, negative_correct)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target_brand')
    args = parser.parse_args()

    # 株価指標
    stocks = {
        'DOW': Stock(YahooCom, '^DJI', 1),
        'FTSE': Stock(YahooCom, '^FTSE', 1),
        'GDAXI': Stock(YahooCom, '^GDAXI', 1),
        'HSI': Stock(YahooCom, '^HSI', 1),
        'N225': Stock(YahooCom, '^N225', 1),
        'NASDAQ': Stock(YahooCom, '^IXIC', 1),
        'SP500': Stock(YahooCom, '^GSPC', 1),
        'SSEC': Stock(YahooCom, '000001.SS', 1),
        args.target_brand: Stock(YahooJp, args.target_brand, 1)
    }

    main(stocks, args.target_brand, result_file='results.csv')
