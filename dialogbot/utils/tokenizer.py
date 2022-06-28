# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import copy
import logging
import re

import jieba
import jieba.analyse
from jieba import posseg

from dialogbot import config
from loguru import logger

jieba.default_logger.setLevel(logging.ERROR)


def postag(text):
    return posseg.cut(text)


def segment_file(in_file, out_file, word_sep=' ', pos_sep='/', is_pos=True):
    """
    segment input file to output file
    :param in_file:
    :param out_file:
    :param word_sep:
    :param pos_sep:
    :param is_pos: 需要词性标注
    :return:
    """
    jieba.enable_parallel()
    with open(in_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
        count = 0
        for line in fin:
            in_line = line.strip()
            seg_line = ''
            if is_pos:
                words = posseg.lcut(in_line)
                for word, pos in words:
                    seg_line += word + pos_sep + pos + word_sep
            else:
                words = jieba.lcut(in_line)
                for word in words:
                    seg_line += word + word_sep
            fout.write(seg_line + "\n")
            count += 1
    print("segment ok. input file count:", count)


def _load_words(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        words_set = set(f.read().splitlines())
    return words_set


class Tokenizer:
    punctuations_set = _load_words(config.punctuations_path)
    stopwords_set = _load_words(config.stopwords_path)
    user_define_words = _load_words(config.user_define_words_path)
    remove_words_set = _load_words(config.remove_words_path)

    # Init jieba
    jieba.initialize()
    for w in user_define_words:
        jieba.add_word(w, freq=1000000)

    corpus_dict = None
    tfidf_model = None

    url_pattern = re.compile(r"(https|http)://.+?html")
    digit_pattern = re.compile(r"\d+")
    bracket_pattern = re.compile(r"\[.+?\]")

    not_place_set = {
        "京东", "上门", "东西", "拜拜", "满意度", "新旧", "入口", "莫大", "蓝牙", "英伦", "顺顺利利", "哥哥", "立马", "海鲜", "回邮", "太多",
        "长北", "南那", "白跑", "天黑", "天阿", "美华", "华联", "日及", "山山", "京福顺", "卡拿", "太卡", "太大", "千古", "英哥", "两棵树",
        "太累", "包邮", "加半", "中华人名共和国", "六便士", "串联", "非顺丰", "中考", "北冰洋", "下嫩", "安安", "太鲜", "上拉", "入店", "上下水",
        "图京", "之城", "中断", "中武", "伦理", "中道", "之康", "多维度", "黑边", "中爱", "之泰", "锦园店", "三国", "阿门", "肯本", "刚京麦",
        "大黑", "朝霞", "关门大吉", "哥别", "沧桑", "下山", "日京京", "沙沙", "牙牙", "顿顿", "山高", "钱和京", "非买", "上旧", "四科", "西东",
        "上岗", "大山", "福尔马林", "滑黑", "上东", "中上", "内马尔", "中同", "中达", "下欧", "四门", "深春", "正东", "江南春", "入维", "大班",
        "中联", "猫沙", "长卡", "几环", "尾塞", "小桥流水", "澳邮", "上中", "英雄", "镇镇", "如东", "上口", "加邮", "八国", "福利", "台基",
        "那本", "中邮", "六本", "维沙", "中黑", "上美", "加花", "天哇", "远超过", "大拿", "贵干", "苏中", "三本", "酒塞", "七本", "美院",
        "中通", "美人壶加", "中充", "下国", "京伦", "九联", "上马", "美化", "江湖", "黑店", "几米远", "午安", "七哥", "角美", "日春", "几比",
        "确保安全", "壶水", "荷塘月色", "云集", "拉边", "欧克", "中右", "加的京", "上路", "烟嘴", "临证指南", "串口卡", "新建", "安利", "山泉水",
        "苏泊尔", "墨黑", "胶盆", "长达", "商城"
    }

    @classmethod
    def place_recognize(cls, text):
        places = [w for w, flag in posseg.cut(text) if "ns" in flag
                  and len(w) >= 2
                  and w not in cls.not_place_set
                  and "哈" not in w
                  and "之" not in w
                  and "本" not in w
                  and "中" not in w
                  and "嫩" not in w
                  and "大" not in w
                  and "鲜" not in w
                  and "国" not in w
                  and "上" not in w
                  and "确" not in w
                  and "牙" not in w
                  and "壶" not in w
                  and "阿" not in w
                  and "入" not in w
                  and "哥" not in w
                  and "颗" not in w
                  and "的" not in w
                  and "联" not in w
                  and "哇" not in w]

        return places

    @classmethod
    def tokenize(cls,
                 text,
                 filter_punctuations=False,
                 filter_stopwords=False,
                 filter_alpha=False,
                 remove_words=False,
                 normalize_url=False,
                 minimum_tokens_num=1):
        """Tokenize text"""
        try:
            places = cls.place_recognize(text)
            for w in places:
                text = text.replace(w, "[地址x]")
            text = cls.digit_pattern.sub("[数字x]", text)
            if normalize_url:
                text = cls.url_pattern.sub("URL", text)
            tokens = jieba.lcut(text)
            text = " ".join(tokens)
            for s in cls.bracket_pattern.findall(text):
                text = text.replace(s, s.replace(" ", ""))
            text = text.replace(u"# E - s [数字x]", u"#E-s[数字x]")
            text = text.replace(u"# E - s DIGIT [数字x]", u"#E-s[数字x]")
            text = text.replace(u"< s >", "<s>")
            tokens = text.split()
            tokens_copy = copy.copy(tokens)

            # Filter words.
            if filter_punctuations:
                tokens = [w for w in tokens if w not in cls.punctuations_set]
            if filter_stopwords:
                tokens = [w for w in tokens if w not in cls.stopwords_set]
            if filter_alpha:
                tokens = [w for w in tokens if not w.encode("utf-8").isalpha()
                          or w in {"URL"}]
            if remove_words:
                tokens = [w for w in tokens if w not in cls.remove_words_set]

            if len(tokens) < minimum_tokens_num:
                tokens = tokens_copy

            new_tokens = tokens[:1]
            t_len = len(tokens)
            for i in range(1, t_len):
                if tokens[i] != tokens[i - 1]:
                    new_tokens.append(tokens[i])
            return new_tokens
        except Exception as e:
            logger.warning("text=%s, errmsg=%s" % (text, e))
        return [text]

    @classmethod
    def get_keywords(cls, text, size=3):
        return jieba.analyse.textrank(text, topK=size)
