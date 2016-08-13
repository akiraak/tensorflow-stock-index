# -*- coding: utf-8 -*-
import os


class LayerLog(object):
    col_count = 8
    save_max = 10
    all_col_count = col_count * save_max

    def __init__(self, dir_path, file_name, codes):
        self.dir_path = dir_path
        self.file_path = os.path.join(dir_path, file_name)
        self.codes = codes

    def is_full(self):
        logs = self.load_file()
        if len(logs) == len(self.codes):
            return True
        return False

    def is_code_full(self, code):
        logs = self.load_file()
        if code in logs and len(logs[code]) >= self.all_col_count:
            return True
        return False

    def add(self, code, datas):
        assert(len(datas) == self.col_count)
        datas = [str(data) for data in datas]
        logs = self.load_file()
        if code in logs:
            logs[code].extend(datas)
        else:
            logs[code] = datas
        self.save_file(logs)

    def load_file(self):
        logs = {}
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                datas = [line.split(',') for line in f.read().split('\n')]
                for data in datas:
                    if len(data) >= (1 + self.col_count):
                        logs[data[0]] = data[1:]
        return logs

    def save_file(self, logs):
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)
        with open(self.file_path, 'w') as f:
            data = '\n'.join([','.join([k] + v) for (k, v) in logs.items()])
            f.write(data)
            print('saved')
