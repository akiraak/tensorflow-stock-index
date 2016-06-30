# -*- coding: utf-8 -*-
import os
from download import YahooCom, YahooJp
from goognet import Stock, main

stocks = {
    '2432': Stock(YahooJp, '2432', 1),  # ディーエヌエ
    '3632': Stock(YahooJp, '3632', 1),  # グリー
    '3635': Stock(YahooJp, '3635', 1),  # コーテクＨＤ
    '3639': Stock(YahooJp, '3639', 1),  # ボルテージ
    '3656': Stock(YahooJp, '3656', 1),  # ＫＬａｂ
    '3659': Stock(YahooJp, '3659', 1),  # ネクソン
    '3662': Stock(YahooJp, '3662', 1),  # エイチーム
    '3667': Stock(YahooJp, '3667', 1),  # ｅｎｉｓｈ
    '3672': Stock(YahooJp, '3672', 1),  # オルトＰ
    '3765': Stock(YahooJp, '3765', 1),  # ガンホー
    '3770': Stock(YahooJp, '3770', 1),  # ザッパラス
    '3903': Stock(YahooJp, '3903', 1),  # ｇｕｍｉ
    '4751': Stock(YahooJp, '4751', 1),  # サイバー
    '6460': Stock(YahooJp, '6460', 1),  # セガサミー
    '7832': Stock(YahooJp, '7832', 1),  # バンナムＨＤ
    '7844': Stock(YahooJp, '7844', 1),  # マーベラス
    '9684': Stock(YahooJp, '9684', 1),  # スクエニＨＤ
    '9697': Stock(YahooJp, '9697', 1),  # カプコン
    '9766': Stock(YahooJp, '9766', 1),  # コナミＨＤ
}

# ボルテージの株取引シミュレーション
main(stocks, '3639')
