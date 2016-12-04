#-*-coding:utf-8-*-

class ChineseSegmentation:

    _B = 0
    _M = 1
    _E = 2
    _S = 3

    def __init__(self):
        self.init_map = {}
        self.b_map = {}
        self.m_map = {}
        self.e_map = {}
        self.s_map = {}
        self.status_map = {}

    # 非空且全为汉字
    @staticmethod
    def all_chinese(text):
        return text and all(u'\u4e00' <= char <= u'\u9fff' for char in text)

    @staticmethod
    def inc_value(m, key):
        cnt = m.setdefault(key, 0)
        m[key] = cnt + 1

    @staticmethod
    def judge_staus():

    def process_line(self, line):
        items = filter(self.all_chinese, line.split(" "))
        is_first = True
        last_status = ""
        for item in items:
            # 处理初始状态
            if is_first:
                is_first = False
                if (len(item == 1)):
                    self.inc_value(self.init_map, "S")
                else:
                    self.inc_value(self.init_map, "B")
            else:
                pass




