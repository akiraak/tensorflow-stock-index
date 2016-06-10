# -*- coding: utf-8 -*-
import os
from brands import all_brands


with open('results.csv', 'w') as f:
    f.write('Code,Name,Accuracy\n')

exist_codes = []
count = 0
datas = all_brands
for (i, (code, name, _)) in enumerate(datas):
    print '{} / {}: {} {}'.format(count + 1, len(datas), code, name)
    if not code in exist_codes:
        count += 1
        os.system('python goognet.py {}'.format(code))

"""
exist_codes = []
output = ['コード,名称,正解率,データ日数,平均取引高,平均株価上下幅']
with open('results.csv', 'r') as f:
    lines =  f.readlines()[1:]
    for line in lines:
        (code, name, accuracy) = line.strip().split(',')
        print code
        with open('data/YH_JP_{}.csv'.format(code), 'r') as f2:
            prices =  f2.readlines()[1:]
            count = len(prices)

            valumes = []
            gap = []
            for (i, price) in enumerate(prices):
                price_datas = price.split(',')
                valumes.append(int(price_datas[5]))
                if price_datas[1] != '---' and price_datas[4] != '---':
                    price_open = float(price_datas[1])
                    price_close = float(price_datas[4])
                    gap.append(abs(price_close - price_open))
                else:
                    gap.append(0)
            valume_avg = sum(valumes) / count
            gap_avg = int(sum(gap) / count)
            code_output = '[' + code + '](http://stocks.finance.yahoo.co.jp/stocks/detail/?code=' + code + ')'
            output.append(','.join([code_output, name, accuracy, str(count) + '日', str(valume_avg) + '株', str(gap_avg) + '円']))

with open('results2.csv', 'w') as f:
    f.write('\n'.join(output))
"""
