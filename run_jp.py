# -*- coding: utf-8 -*-
import os
from brands import all_brands, nikkei225, nikkei225_s
from layer_log import LayerLog


brands = nikkei225
#brands = nikkei225_s

layer1 = 512
layer2 = 512

# ログファイル
codes = [code for (code, name, _) in brands]
layerLog = LayerLog('layer_logs', '{}_{}.csv'.format(layer1, layer2), codes)

# 計算する
for i, (code, name, _) in enumerate(brands):
    print '{} / {}: {} {}'.format(i + 1, len(codes), code, name)

    with open(os.path.join('data', 'YH_JP_{}.csv'.format(code)), 'r') as f:
        lines = f.read().split('\n')
        if len(lines) > 2000:
            while not layerLog.is_code_full(code):
                commena = 'python goognet.py {} --layer1={} --layer2={}'.format(code, layer1, layer2)
                os.system(commena)
        else:
            print 'データが少なすぎる'
