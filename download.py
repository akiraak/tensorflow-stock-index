# -*- coding: utf-8 -*-
import sys
import os.path
import datetime
import urllib2
import pandas as pd
import lxml.html
import re
import time
import argparse
from brands import all_brands


dateMatch = re.compile(u'(\d+)年(\d+)月(\d+)日')


class Download(object):
    FROM_YEAR = '1991'
    SAVE_PATH = 'data'

    COL_DATE = 0
    COL_OPEN = 1
    COL_HIGH = 2
    COL_LOW = 3
    COL_CLOSE = 4
    COL_VOLUME = 5
    COL_ADJ_CLOSE = 6

    def __init__(self, auto_upload=False):
        self.auto_upload = auto_upload
        self.prices = []
        # ファイル保存用のディレクトリを作成
        if not os.path.isdir(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)

    @property
    def filePath(self):
        return ''

    def _fetchCSV(self):
        if self.auto_upload and os.path.exists(self.filePath):
            self.prices = self._upload_prices()
            self._save_prices()
        else:
            if not os.path.exists(self.filePath):
                self.prices = self._download_prices()
                self._save_prices()

    def _download_prices(self):
        return []

    def _upload_prices(self):
        return []

    def load_csv(self):
        with open(self.filePath) as f:
            self.prices = [line.split(',') for line in f.read().split('\n')][1:]
    @property
    def dataframe(self):
        if len(self.prices) == 0:
            self.load_csv()
        return pd.read_csv(self.filePath).set_index('Date').sort_index()

    def price(self, str_date):
        if len(self.prices) == 0:
            self.load_csv()
        for price in self.prices:
            if price[0] == str_date:
                return price
        return None

    def _save_prices(self):
        header = [['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        datas = header + self.prices
        csv = '\n'.join([','.join(line) for line in datas])
        with open(self.filePath, 'w') as f:
            f.write(csv)

    @classmethod
    def _str_to_date(cls, str_date):
        date = str_date.split('-')
        return datetime.date(int(date[0]), int(date[1]), int(date[2]))


class YahooCom(Download):
    def __init__(self, code, auto_upload=False):
        super(YahooCom, self).__init__(auto_upload)
        self.code = code
        self._fetchCSV()

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

    def _download_prices(self):
        print('fetch CSV for url: ' + self._url)
        csv = urllib2.urlopen(self._url).read()
        return [line.split(',') for line in csv.split('\n')[1:]]

    def _upload_prices(self):
        return self._download_prices()


class YahooJp(Download):
    def __init__(self, code, auto_upload=False):
        super(YahooJp, self).__init__(auto_upload)
        self.code = code
        self._fetchCSV()

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

    def _download_prices(self, limit_date=None):
        print 'fetch CSV for url: ' + self._url(1),
        sys.stdout.flush()
        page = 1
        prices = []
        fetch = True
        while fetch:
            print page,
            sys.stdout.flush()
            pagePrices = self._downloadPage(page)
            if len(pagePrices) > 0:
                if limit_date:
                    for price in pagePrices:
                        date = self._str_to_date(price[0])
                        if date > limit_date:
                            prices.append(price)
                        else:
                            fetch = False
                            break
                else:
                    prices.extend(pagePrices)
                page += 1
                #time.sleep(0.1)
            else:
                fetch = False
        print ''
        return prices

    def _upload_prices(self):
        with open(self.filePath) as f:
            prices = [line.split(',') for line in f.read().split('\n')]

        if len(prices) >= 2:
            last_date = self._str_to_date(prices[1][0])
            new_prices = self._download_prices(limit_date=last_date)
            new_prices.extend(prices[1:])
            return new_prices
        else:
            return self._download_prices()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', type=int, default=0)
    args = parser.parse_args()

    for (i, (code, name, _)) in enumerate(all_brands[args.skip:]):
        print '{} / {}'.format(args.skip + i + 1, len(all_brands)), code, name
        YahooJp(code)
