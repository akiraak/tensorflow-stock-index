# -*- coding: utf-8 -*-
import sys
import os.path
import datetime
import urllib2
import pandas as pd
import lxml.html
import re
import time


dateMatch = re.compile(u'(\d+)年(\d+)月(\d+)日')


class Download(object):
    FROM_YEAR = '1991'
    SAVE_PATH = 'data'

    def __init__(self):
        # ファイル保存用のディレクトリを作成
        if not os.path.isdir(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)
        self.csv = None

    @property
    def filePath(self):
        return ''

    def _fetchCSV(self):
        if os.path.exists(self.filePath):
            #print('fetch CSV for local: ' + self.filePath)
            with open(self.filePath) as f:
                return f.read()
        else:
            return self._download()

    def _download(self):
        return ""

    @property
    def dataframe(self):
        return pd.read_csv(self.filePath).set_index('Date').sort_index()


class YahooCom(Download):
    def __init__(self, code):
        super(YahooCom, self).__init__()
        self.code = code
        self.csv = self._fetchCSV()

    @property
    def filePath(self):
        return os.path.join(self.SAVE_PATH, 'YH_EN_{}.csv'.format(self.code))

    @property
    def _url(self):
        now = datetime.date.today()
        return 'http://chart.finance.yahoo.com/table.csv?s={}&a=0&b=1&c={}&d={}&e={}&f={}&g=d&ignore=.csv'.format(
            self.code,
            self.FROM_YEAR,
            str(now.month - 1),
            str(now.day),
            str(now.year)
        )

    def _download(self):
        print('fetch CSV for url: ' + self._url)
        csv = urllib2.urlopen(self._url).read()
        with open(self.filePath, 'w') as f:
            f.write(csv)
        return csv


class YahooJp(Download):
    def __init__(self, code):
        super(YahooJp, self).__init__()
        self.code = code
        self.csv = self._fetchCSV()

    @property
    def filePath(self):
        return os.path.join(self.SAVE_PATH, 'YH_JP_{}.csv'.format(self.code))

    def _url(self, page):
        now = datetime.date.today()
        return 'http://info.finance.yahoo.co.jp/history/?code={}&sy={}&sm=1&sd=1&ey={}&em={}&ed={}&tm=d&p={}'.format(
            self.code,
            self.FROM_YEAR,
            str(now.year),
            str(now.month),
            str(now.day),
            page
        )

    def _convertPrice(self, tds):
        if len(tds) == 7:
            date = tds[0].text_content()
            dateNum = dateMatch.match(date)
            return [
                '%s-%02d-%02d' % (dateNum.group(1), int(dateNum.group(2)), int(dateNum.group(3))),
                tds[1].text_content().replace(',', ''),
                tds[2].text_content().replace(',', ''),
                tds[3].text_content().replace(',', ''),
                tds[4].text_content().replace(',', ''),
                tds[5].text_content().replace(',', ''),
                tds[6].text_content().replace(',', ''),
            ]
        else:
            return None

    def _getPrices(self, html, divNum):
        prices = []
        root = lxml.html.fromstring(html)
        lines = root.xpath('//*[@id="main"]/div[{}]/table/tr'.format(divNum))
        for line in lines[1:]:
            tds = line.xpath('td')
            price = self._convertPrice(tds)
            if price:
                prices.append(price)
        return prices

    def _downloadPage(self, page):
        html = urllib2.urlopen(self._url(page)).read()
        prices = self._getPrices(html, 5)
        if len(prices) == 0:
            prices = self._getPrices(html, 6)
        return prices

    def _download(self):
        print 'fetch CSV for url: ' + self._url(1),
        sys.stdout.flush()
        page = 1
        prices = [['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        while True:
            print page,
            sys.stdout.flush()
            pagePrices = self._downloadPage(page)
            if len(pagePrices) > 0:
                prices.extend(pagePrices)
                page += 1
                time.sleep(0.1)
            else:
                break
        print ''
        csv = '\n'.join([','.join(price) for price in prices])
        with open(self.filePath, 'w') as f:
            f.write(csv)
        return csv
