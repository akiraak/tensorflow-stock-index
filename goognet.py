# -*- coding: utf-8 -*-
'''
Code based on:
https://github.com/corrieelston/datalab/blob/master/FinancialTimeSeriesTensorFlow.ipynb
'''
from __future__ import print_function

import datetime
import urllib2
import math
import os
import operator as op
from collections import namedtuple
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from download import Download, YahooCom, YahooJp
from brands import brand_name
from layer_log import LayerLog
from brands import nikkei225_s
from feed_cache import FeedCache


TEST_COUNT = 200        # テスト日数
TRAIN_MIN = 1000        # 学習データの最低日数
TRAIN_MAX = None        # 学習データの最大日数
DAYS_BACK = 3           # 過去何日分を計算に使用するか
STEPS = 10000           # 学習回数
CHECKIN_INTERVAL = 100  # 学習の途中結果を表示する間隔
REMOVE_NIL_DATE = True  # 計算対象の日付に存在しないデータを削除する
PASS_DAYS = 10          # 除外する古いデータの日数
DROP_RATE = 0.1         # 学習時のドロップアウトの比率
UP_RATE = 0.07          # 上位何パーセントを買いと判断するか
STDDEV = 1e-4           # 学習係数
REMOVE_NEWEST_DAYS = 200 * 1    # 除外する最新のデータ日数

CLASS_LABELS = ['DOWN', 'NEUTRAL', 'UP']
CLASS_DOWN = 0
CLASS_NEUTRAL = 1
CLASS_UP = 2
CLASS_COUNT = len(CLASS_LABELS)

# 学習データに使用するパラメータのラベル
PARAM_LABELS = ['Close', 'High', 'Low', 'Volume']


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
Environ = namedtuple('Environ', 'sess model actual_classes training_step dataset feature_data keep_prob')


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
    return pd.read_csv('index_{}.csv'.format(exchange), index_col='Date').sort_index()


def get_using_data(dataframes, target_brand):
    '''各指標の必要なカラムをまとめて1つのDataFrameに詰める。

    Args:
        dataframes: {key: pd.DataFrame()}
    Returns:
        pd.DataFrame()
    '''
    using_data = pd.DataFrame()
    datas = [(target_brand, dataframes[target_brand])]
    datas.extend([(exchange, dataframe) for exchange, dataframe in dataframes.items() if exchange != target_brand])
    for exchange, dataframe in datas:
        using_data['{}_Open'.format(exchange)] = dataframe['Open']
        using_data['{}_Close'.format(exchange)] = dataframe['Close']
        using_data['{}_High'.format(exchange)] = dataframe['High']
        using_data['{}_Low'.format(exchange)] = dataframe['Low']
        using_data['{}_Volume'.format(exchange)] = dataframe['Volume']
    using_data = using_data.fillna(method='ffill')
    return using_data


def zscore(np_array):
    '''配列の標準化を行う
    '''
    a1 = np_array.replace(0, 1.).replace(np.inf, 1.).replace(np.nan, 1.)
    a2 = a1 - a1.mean()
    a3 = a2 / a2.std()
    return a3


def get_log_return_data(stocks, using_data):
    '''各指標について、終値を1日前との比率の対数をとって正規化する。

    Args:
        using_data: pd.DataFrame()
    Returns:
        pd.DataFrame()
    '''
    log_return_data = pd.DataFrame()
    for (name, stock) in stocks.items():
        open_column = '{}_Open'.format(name)
        close_column = '{}_Close'.format(name)
        high_column = '{}_High'.format(name)
        low_column = '{}_Low'.format(name)
        volume_column = '{}_Volume'.format(name)

        # 学習データの「終値／始値」を取得
        train_close_rates = (using_data[close_column]/using_data[close_column].shift()).values[:len(using_data[close_column]) - TEST_COUNT]
        # 小さい順にソートする
        train_close_rates.sort()
        # 何%以上上昇した場合に購入するかの閾値を得る
        up_index = int(len(train_close_rates) * (1. - UP_RATE))
        up_rate = train_close_rates[up_index] - 1.

        # np.log(当日終値 / 前日終値) で前日からの変化率を算出
        # 前日よりも上がっていればプラス、下がっていればマイナスになる
        log_return_data['{}_Close_RATE'.format(name)] = zscore(using_data[close_column]/using_data[close_column].shift())
        # 当日高値 / 当日始値
        log_return_data['{}_High_RATE'.format(name)] = zscore(using_data[high_column]/using_data[open_column])
        # 当日安値 / 当日始値
        log_return_data['{}_Low_RATE'.format(name)] = zscore(using_data[low_column]/using_data[open_column])
        # 当日出来高 / 前日出来高
        log_return_data['{}_Volume_RATE'.format(name)] = zscore(using_data[volume_column]/using_data[volume_column].shift())

        # 答を求める
        answers = []
        # 下がる／上がると判断する変化率
        change_rate = up_rate
        for value in (using_data[close_column] / using_data[open_column]).values:
            if value < (1 - change_rate):
                # 下がる
                answers.append(CLASS_DOWN)
            elif value > (1 + change_rate):
                # 上がる
                answers.append(CLASS_UP)
            else:
                # 変化なし
                answers.append(CLASS_NEUTRAL)
        log_return_data['{}_RESULT'.format(name)] = answers

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

    # 答を詰める
    columns = ['answer_{}'.format(label) for label in CLASS_LABELS]
    for i in range(CLASS_COUNT):
        column = columns[i]
        log_return_data[column] = 0
        indices = op.eq(log_return_data['{}_RESULT'.format(target_brand)], i)
        log_return_data.ix[indices, column] = 1

    # 各指標のカラム名を追加
    for colname, _, _ in iter_exchange_days_back(stocks, target_brand, max_days_back):
        for date_type in PARAM_LABELS:
            columns.append('{}_{}'.format(colname, date_type))

    # データ数をもとめる
    max_index = len(log_return_data)

    # 学習データを作る
    training_test_data = pd.DataFrame(columns=columns)
    for i in range(max_days_back + PASS_DAYS, max_index):
        # 先頭のデータを含めるとなぜか上手くいかないので max_days_back + PASS_DAYS で少し省く
        values = {}
        # 答を入れる
        for answer_i in range(CLASS_COUNT):
            column = columns[answer_i]
            values[column] = log_return_data[column].ix[i]
        # 学習データを入れる
        for colname, exchange, days_back in iter_exchange_days_back(stocks, target_brand, max_days_back):
            for date_type in PARAM_LABELS:
                col = '{}_{}'.format(colname, date_type)
                values[col] = log_return_data['{}_{}_RATE'.format(exchange, date_type)].ix[i - days_back]
        training_test_data = training_test_data.append(values, ignore_index=True)

    # index（日付ラベル）を引き継ぐ
    training_test_data.index = log_return_data.index[max_days_back + PASS_DAYS: max_index]
    return training_test_data


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

    # 最新のデータを除外する
    if REMOVE_NEWEST_DAYS:
        training_test_data = training_test_data[:-REMOVE_NEWEST_DAYS]

    # 学習とテストに使用するデータ数を絞る
    if TRAIN_MAX:
        training_test_data = training_test_data[:TRAIN_MAX+TEST_COUNT]

    # 先頭のいくつかより後ろが学習データ
    predictors_tf = training_test_data[training_test_data.columns[num_categories:]]
    # 先頭のいくつかが答えデータ
    classes_tf = training_test_data[training_test_data.columns[:num_categories]]

    # 学習用とテスト用のデータサイズを求める
    training_set_size = len(training_test_data) - TEST_COUNT

    return Dataset(
        training_predictors=predictors_tf[:training_set_size],
        training_classes=classes_tf[:training_set_size],
        test_predictors=predictors_tf[training_set_size:],
        test_classes=classes_tf[training_set_size:],
    )


def smarter_network(stocks, dataset, layer1, layer2):
    '''隠しレイヤー入りのもうちょっと複雑な分類モデルを返す。
    '''
    sess = tf.Session()

    num_predictors = len(dataset.training_predictors.columns)
    num_classes = len(dataset.training_classes.columns)

    feature_data = tf.placeholder("float", [None, num_predictors])
    actual_classes = tf.placeholder("float", [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    layer_counts = [layer1, layer2, CLASS_COUNT]
    weights = []
    biases = []
    model = None
    for i, count in enumerate(layer_counts):
        # 重み付けの変数定義
        if i == 0:
            weights = tf.Variable(tf.truncated_normal([num_predictors, count], stddev=STDDEV))
        else:
            weights = tf.Variable(tf.truncated_normal([layer_counts[i - 1], count], stddev=STDDEV))
        # バイアスの変数定義
        biases = tf.Variable(tf.ones([count]))

        if model == None:
            # 一番最初のレイヤー
            model = tf.nn.relu(tf.matmul(feature_data, weights) + biases)
        else:
            if (i + 1) < len(layer_counts):
                # 最後ではないレイヤー
                model = tf.nn.relu(tf.matmul(model, weights) + biases)
            else:
                # 最終レイヤーの前には dropout を入れる
                model = tf.nn.dropout(model, keep_prob)
                model = tf.nn.softmax(tf.matmul(model, weights) + biases)

    # 予測が正しいかを計算（学習に使用する）
    cost = -tf.reduce_sum(actual_classes*tf.log(model))
    training_step = tf.train.AdamOptimizer(learning_rate=STDDEV).minimize(cost)

    # 変数の初期化処理
    init = tf.initialize_all_variables()
    sess.run(init)

    return Environ(
        sess=sess,
        model=model,
        actual_classes=actual_classes,
        training_step=training_step,
        dataset=dataset,
        feature_data=feature_data,
        keep_prob=keep_prob,
    )


def train(env, target_prices):
    '''学習をsteps回おこなう。
    '''

    # 予測（model）と実際の値（actual）が一致（equal）した場合の配列を取得する
    #   結果の例: [1,1,0,1,0] 1が正解
    correct_prediction = tf.equal(
        tf.argmax(env.model, 1),
        tf.argmax(env.actual_classes, 1))
    # 結果（例：[1,1,0,1,0] 1が正解）を float にキャストして
    # 全ての平均（reduce_mean）を得る
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    max_train_accuracy = 0
    lastScore = None
    for i in range(1, 1 + STEPS):
        env.sess.run(
            env.training_step,
            feed_dict=feed_dict(env, test=False, keep_prob=(1.0 - DROP_RATE)),
        )
        if i % CHECKIN_INTERVAL == 0:
            train_accuracy = env.sess.run(
                accuracy,
                feed_dict=feed_dict(env, test=False),
            )
            money, trues, falses, actual_count, deal_logs = gamble(env, target_prices)
            true_count = trues[CLASS_UP] + falses[CLASS_UP]
            true_rate = 0.
            if true_count:
                true_rate = float(trues[CLASS_UP]) / float(true_count)

            # テストデータの開始と終了の日付を取得
            test_dates = env.dataset.test_predictors.index
            test_from_date = test_dates[0]
            test_to_date = test_dates[-1]

            print(i, '{:,d}円 {:.3f} {:.3f} {}-{}'.format(money, true_rate, train_accuracy, test_from_date, test_to_date))
            if max_train_accuracy < train_accuracy:
                max_train_accuracy = train_accuracy
            elif train_accuracy < 0.5:
                break
            lastScore = (max_train_accuracy, money, trues, falses, actual_count, deal_logs)

    return lastScore


# 売買シミュレーション
def gamble(env, target_prices):
    # 予想
    expectations = env.sess.run(
        tf.argmax(env.model, 1),
        feed_dict=feed_dict(env, test=True),
    )

    # 元金
    money = 10000 * 1000
    # 売買履歴
    deal_logs = []
    # 予想が当たった数
    trues = np.zeros(CLASS_COUNT, dtype=np.int64)
    # 予想が外れた数
    falses = np.zeros(CLASS_COUNT, dtype=np.int64)
    # 実際の結果の数
    actual_count = np.zeros(CLASS_COUNT, dtype=np.int64)
    # 実際の結果
    actual_classes = getattr(env.dataset, 'test_classes')

    # 結果の集計と売買シミュレーション
    for (i, date) in enumerate(env.dataset.test_predictors.index):
        expectation = expectations[i]
        if expectation == CLASS_UP:
            # 上がる予想なので買う
            price = target_prices.download.price(date)
            if price != None:
                 # 始値
                open_value = float(price[Download.COL_OPEN])
                # 終値
                close_value = float(price[Download.COL_CLOSE])
                # 購入可能な株数
                value = money / open_value
                # 購入金額
                buy_value = int(open_value * value)
                # 売却金額
                sell_value = int(close_value * value)
                # 購入
                money -= buy_value
                # 購入手数料の支払い
                money -= buy_charge(buy_value)
                # 売却
                money += sell_value
                # 売却手数料の支払い
                money -= buy_charge(sell_value)

        actual = np.argmax(actual_classes.ix[date].values)
        if expectation == actual:
            # 当たった
            trues[expectation] += 1
        else:
            # 外れた
            falses[expectation] += 1
        actual_count[actual] += 1
        deal_logs.append([date, CLASS_LABELS[expectation], CLASS_LABELS[actual], money])

    return money, trues, falses, actual_count, deal_logs


def feed_dict(env, test=False, keep_prob=1.):
    '''学習/テストに使うデータを生成する。
    '''
    prefix = 'test' if test else 'training'
    predictors = getattr(env.dataset, '{}_predictors'.format(prefix))
    classes = getattr(env.dataset, '{}_classes'.format(prefix))
    return {
        env.feature_data: predictors.values,
        env.actual_classes: classes.values.reshape(len(classes.values), len(classes.columns)),
        env.keep_prob: keep_prob
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


def save_deal_logs(target_brand, deal_logs):
    save_dir = 'deal_logs'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open('{}/{}.csv'.format(save_dir, target_brand), 'w') as f:
        f.write('\n'.join([','.join([str(log) for log in logs])  for logs in deal_logs]))


def main(stocks, target_brand, layer1, layer2, result_file=None):
    # 学習データのキャッシュ
    feed_cache = FeedCache(target_brand, str(REMOVE_NEWEST_DAYS))

    # 対象の銘柄名
    target_brand_name = brand_name(target_brand)

    if not feed_cache.is_exist():
        print('Make cache')
        # 株価指標データを読み込む
        all_data  = load_exchange_dataframes(stocks, target_brand)

        # 終値を取得
        using_data = get_using_data(all_data, target_brand)

        # データを学習に使える形式に正規化
        log_return_data = get_log_return_data(stocks, using_data)

        # 答と学習データを作る
        training_test_data = build_training_data(
            stocks, log_return_data, target_brand)
        feed_cache.save(training_test_data)
    else:
        print('Exist cache')
        training_test_data = feed_cache.load()

    # 学習データをトレーニング用とテスト用に分割する
    dataset = split_training_test_data(CLASS_COUNT, training_test_data)
    if len(dataset.training_predictors) < TRAIN_MIN:
        print('[{}]{}: 学習データが少なすぎるため計算を中止'.format(target_brand, target_brand_name))
        with open('results.csv', 'a') as f:
            f.write('{},{},ERROR\n'.format(target_brand, target_brand_name))

        # レイヤー検証ログに保存
        brands = nikkei225_s
        codes = [code for (code, name, _) in brands]
        layerLog = LayerLog('layer_logs', '{}_{}.csv'.format(layer1, layer2), codes)
        layerLog.add(
            target_brand,
            [-1, 0, 0, 0, 0, 0, 0, 0]
        )
        return

    print('[{}]{}'.format(target_brand, target_brand_name))

    # 器械学習のネットワークを作成
    env = smarter_network(stocks, dataset, layer1, layer2)

    # 学習
    train_accuracy, money, trues, falses, actual_count, deal_logs = train(env, stocks[target_brand])

    print('-- テスト --')
    # 各クラスの正解率
    rates = np.zeros(CLASS_COUNT, dtype=np.int64)
    # 各クラスの正解数
    counts = np.zeros(CLASS_COUNT, dtype=np.int64)
    for i in range(CLASS_COUNT):
        counts[i] = trues[i] + falses[i]
        if counts[i]:
            # 各クラスの正解率（予想数 / 正解数）
            rates[i] = int(float(trues[i]) / float(counts[i]) * 100)
    print('下げ正解率    : {}% 予想{}回'.format(rates[CLASS_DOWN], counts[CLASS_DOWN]))
    print('変化なし正解率: {}% 予想{}回'.format(rates[CLASS_NEUTRAL], counts[CLASS_NEUTRAL]))
    print('上げ正解率    : {}% 予想{}回'.format(rates[CLASS_UP], counts[CLASS_UP]))

    print('-- 売買シミュレーション --')
    print('売買シミュレーション結果 {:,d}円'.format(money))

    # 結果をファイル保存
    if result_file:
        with open(result_file, 'a') as f:
            f.write('{},{},{},{},{},{},{},{},{}\n'.format(
                target_brand,
                target_brand_name,
                money,
                trues[CLASS_DOWN], falses[CLASS_DOWN],
                trues[CLASS_NEUTRAL], falses[CLASS_NEUTRAL],
                trues[CLASS_UP], falses[CLASS_UP]))
    else:
        print(
            target_brand,
            target_brand_name,
            money,
            trues[CLASS_DOWN], falses[CLASS_DOWN],
            trues[CLASS_NEUTRAL], falses[CLASS_NEUTRAL],
            trues[CLASS_UP], falses[CLASS_UP])

    # 売買履歴をファイルに保存
    save_deal_logs(target_brand, deal_logs)

    # レイヤー検証ログに保存
    brands = nikkei225_s
    codes = [code for (code, name, _) in brands]
    layerLog = LayerLog('layer_logs', '{}_{}.csv'.format(layer1, layer2), codes)
    layerLog.add(
        target_brand,
        [
            train_accuracy,
            money,
            trues[CLASS_DOWN], falses[CLASS_DOWN],
            trues[CLASS_NEUTRAL], falses[CLASS_NEUTRAL],
            trues[CLASS_UP], falses[CLASS_UP]
        ]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target_brand')
    parser.add_argument('--layer1', type=int, default=512)
    parser.add_argument('--layer2', type=int, default=512)
    args = parser.parse_args()

    stocks = {
        # 株価指標
        'DOW': Stock(YahooCom, '^DJI', 1),
        'FTSE': Stock(YahooCom, '^FTSE', 1),
        'GDAXI': Stock(YahooCom, '^GDAXI', 1),
        'HSI': Stock(YahooCom, '^HSI', 1),
        'N225': Stock(YahooCom, '^N225', 1),
        'NASDAQ': Stock(YahooCom, '^IXIC', 1),
        'SP500': Stock(YahooCom, '^GSPC', 1),
        #'SSEC': Stock(YahooCom, '000001.SS', 1),
        # 対象の銘柄
        args.target_brand: Stock(YahooJp, args.target_brand, 1)
    }
    print('REMOVE_NEWEST_DAYS {}'.format(REMOVE_NEWEST_DAYS))
    main(stocks, args.target_brand, args.layer1, args.layer2, result_file='results.csv')
