# -*- coding: utf-8 -*-
import os
from brands import nikkei225_excellent
from brands import (
    nikkei225_excellent5, nikkei225_excellent10,
    nikkei225_excellent20, nikkei225_excellent30
)


brands = nikkei225_excellent5
#brands = nikkei225_excellent10
#brands = nikkei225_excellent20
#brands = nikkei225_excellent30
layer1 = 512
layer2 = 512


file_path = 'up_expectation_dates.csv'
if os.path.exists(file_path):
    os.remove(file_path)

for i, (code, name, _) in enumerate(brands):
    print '{} / {}: {} {}'.format(i + 1, len(brands), code, name)
    commena = 'python goognet.py {} --layer1={} --layer2={} --load_sess=0'.format(code, layer1, layer2)
    print commena
    os.system(commena)
