# -*- coding: utf-8 -*-
import os
from download import Download, YahooJp
from goognet import buy_charge


file_path = 'up_expectation_dates.csv'
brand_dates = []

if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        brand_dates = [line.split(',') for line in f.read().split('\n')]

buy_dates = {}
for brand in  brand_dates:
    brand_code = brand[0]
    for date in brand[1:]:
        if not date in buy_dates:
            buy_dates[date] = brand_code

# 元金
money = 10000 * 1000
for date, brand in sorted(buy_dates.items()):
    price = YahooJp(brand).price(date)

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
    print date, money
print money
