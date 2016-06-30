# -*- coding: utf-8 -*-
from download import YahooCom, YahooJp
from goognet import Stock, main

stocks = {
    'DOW': Stock(YahooCom, '^DJI', 1),
    'FTSE': Stock(YahooCom, '^FTSE', 1),
    'GDAXI': Stock(YahooCom, '^GDAXI', 1),
    'HSI': Stock(YahooCom, '^HSI', 1),
    'N225': Stock(YahooCom, '^N225', 1),
    'NASDAQ': Stock(YahooCom, '^IXIC', 1),
    'SP500': Stock(YahooCom, '^GSPC', 1),
    'SSEC': Stock(YahooCom, '000001.SS', 1),
    '2170': Stock(YahooJp, '2170', 1),
}

main(stocks, '2170')
