# -*- coding: utf-8 -*-
import os
from nikkei225 import nikkei225


with open('results.csv', 'w') as f:
    f.write('Code,Name,Accuracy\n')

for (target_brand_code, target_brand_name, _) in nikkei225:
    os.system('python goognet.py {}'.format(target_brand_code))
    #os.system('python goognet.py SP500')
    #raise
