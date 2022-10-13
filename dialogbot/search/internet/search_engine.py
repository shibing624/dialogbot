# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 对百度、Bing 的搜索摘要进行答案的检索
"""

import urllib.parse
import urllib.request
from collections import OrderedDict
from loguru import logger
from dialogbot.search.internet import html_crawler
from dialogbot.utils.tokenizer import postag

baidu_url_prefix = 'https://www.baidu.com/s?ie=utf-8&wd='
bing_url_prefix = 'https://cn.bing.com/search?q='
calendar_url = 'http://open.baidu.com/calendar'
calculator_url = 'http://open.baidu.com/static/calculator/calculator.html'
weather_url = 'http://weathernew.pae.baidu.com'
split_symbol = ["。", "?", ".", "_", "-", ":", "！", "？"]


def split_2_short_text(sentence):
    for i in split_symbol:
        sentence = sentence.replace(i, i + '\t')
    return sentence.split('\t')


def keep_pos_words(query, tags=['n']):
    result = []
    words = postag(query)
    for k in words:
        # 只保留所需词性的词
        if k.flag in tags:
            result.append(k.word)
    return result


class Engine:
    def __init__(self, topk=10):
        self.name = 'engine'
        self.topk = topk
        self.contents = OrderedDict()

    def search(self, query):
        # 检索baidu
        r, baidu_left_text = self.search_baidu(query)
        if r:
            self.contents[query] = r
            return r

        # 检索bing
        r, bing_left_text = self.search_bing(query)
        if r:
            self.contents[query] = r
            return r

        # 检索baidu + bing 的摘要
        r = self._search_other(query, baidu_left_text + bing_left_text)
        if r:
            self.contents[query] = r
            return r
        return r

    def search_baidu(self, query):
        """
        通过baidu检索答案，包括百度知识图谱、百度诗词、百度万年历、百度计算器、百度知道
        :param query:
        :return: list, string
        """
        answer = []
        left_text = ''
        # 抓取百度前10条的摘要
        soup_baidu = html_crawler.get_html_baidu(baidu_url_prefix + urllib.parse.quote(query))
        if not soup_baidu:
            return answer, left_text
        if soup_baidu.title.get_text().__contains__('百度安全验证'):
            logger.warning("爬虫触发百度安全验证")
            return answer, left_text
        for i in range(1, self.topk):
            items = soup_baidu.find(id=i)
            if not items:
                logger.debug("百度找不到答案")
                break
            # Find result by internet
            # 判断是否有mu,如果第一个是百度知识图谱的 就直接命中答案
            if ('mu' in items.attrs) and i == 1:
                r = items.find(class_='op_exactqa_s_answer')
                if r:
                    logger.debug("百度知识图谱找到答案")
                    answer.append(r.get_text().strip())
                    return answer, left_text

            # 古诗词判断
            if ('mu' in items.attrs) and i == 1:
                r = items.find(class_="op_exactqa_detail_s_answer")
                if r:
                    logger.debug("百度诗词找到答案")
                    answer.append(r.get_text().strip())
                    return answer, left_text

            # 万年历 & 日期
            if ('mu' in items.attrs) and i == 1 and items.attrs['mu'].__contains__(calendar_url):
                r = items.find(class_="op-calendar-content")
                if r:
                    logger.debug("百度万年历找到答案")
                    answer.append(r.get_text().strip().replace("\n", "").replace(" ", ""))
                    return answer, left_text

            if ('tpl' in items.attrs) and i == 1 and items.attrs['tpl'].__contains__('calendar_new'):
                r = items.attrs['fk'].replace("6018_", "")
                logger.debug(r)
                if r:
                    logger.debug("百度万年历新版找到答案")
                    answer.append(r)
                    return answer, left_text

            # 计算器
            if ('mu' in items.attrs) and i == 1 and items.attrs['mu'].__contains__(calculator_url):
                r = items.find(class_="op_new_val_screen_result")
                if r:
                    logger.debug("计算器找到答案")
                    answer.append(r.get_text().strip())
                    return answer, left_text

            # 天气
            if ('mu' in items.attrs) and i == 1 and items.attrs['mu'].__contains__(weather_url):
                r = items.find(class_="op_weather4_twoicon_today")
                if r:
                    logger.debug("天气找到答案")
                    answer.append(r.get_text().strip().replace("\n", "").replace(' ', '').strip())
                    return answer, left_text
            # 百度知道
            if ('mu' in items.attrs) and i == 1:
                r = items.find(class_='op_best_answer_question_link')
                if r:
                    zhidao_soup = html_crawler.get_html_zhidao(r['href'])
                    r = zhidao_soup.find(class_='bd answer').find('pre')
                    if not r:
                        r = zhidao_soup.find(class_='bd answer').find(class_='line content').find(
                            class_="best-text mb-10")
                    if r:
                        logger.debug("百度知道找到答案")
                        answer.append(r.get_text().strip().replace("展开全部", "").strip())
                        return answer, left_text

            if items.find("h3") and items.find("h3").find("a"):
                # 百度知道-页面
                if items.find("h3").find("a").get_text().__contains__("百度知道"):
                    url = items.find("h3").find("a")['href']
                    if url:
                        zhidao_soup = html_crawler.get_html_zhidao(url)
                        r = zhidao_soup.find(class_='bd answer')
                        if r:
                            r = r.find('pre')
                            if not r:
                                r = zhidao_soup.find(class_='bd answer').find(class_='line content').find(
                                    class_="best-text mb-10")
                            if r:
                                logger.debug("百度知道找到答案")
                                answer.append(r.get_text().strip().replace("展开全部", "").strip())
                                return answer, left_text

                # 百度百科
                if items.find("h3").find("a").get_text().__contains__("百度百科") and i <=5:
                    url = items.find("h3").find("a")['href']
                    if url:
                        logger.debug("百度百科找到答案")
                        baike_soup = html_crawler.get_html_baike(url)

                        r = baike_soup.find(class_='lemma-summary')
                        if r:
                            answer.append(r.get_text().replace("\n", "").strip())
                            return answer, left_text
            left_text += items.get_text()
        return answer, left_text

    @staticmethod
    def search_bing(query):
        """
        通过bing检索答案，包括bing知识图谱、bing网典
        :param query:
        :return: list, string
        """
        answer = []
        left_text = ''
        # 获取bing的摘要
        soup_bing = html_crawler.get_html_bing(bing_url_prefix + urllib.parse.quote(query))
        # 判断是否在Bing的知识图谱中
        r = soup_bing.find(class_="b_entityTP")

        if r:
            r = r.find_all(class_="b_subModule")
            if r and len(r) > 1 and r[1].find("li"):
                r = r[1].find("li").get_text().strip()
                if r:
                    answer.append(r)
                    logger.debug("Bing知识图谱找到答案")
                    return answer, left_text
        else:
            r = soup_bing.find(id="dict_ans")
            if r:
                bing_list = r.find_all('li')
                for bl in bing_list:
                    temp = bl.get_text()
                    if temp.__contains__("必应网典"):
                        logger.debug("查找Bing网典")
                        url = bl.find("h2").find("a")['href']
                        if url:
                            bingwd_soup = html_crawler.get_html_bingwd(url)
                            r = bingwd_soup.find(class_='bk_card_desc').find("p")
                            if r:
                                r = r.get_text().replace("\n", "").strip()
                                if r:
                                    logger.debug("Bing网典找到答案")
                                    answer.append(r)
                                    return answer, left_text
                left_text += r.get_text()
        if not answer:
            logger.debug("Bing找不到答案")
        return answer, left_text

    def _search_other(self, query, left_text):
        """
        如果 baidu + bing 知识图谱中都没找到答案，那么就分析摘要
        :param query:
        :return: list, string
        """
        answer = []
        # 取名词为核心词
        keywords = keep_pos_words(query)
        # 分句
        sentences = split_2_short_text(left_text.strip())

        # 找到含有关键词的句子, 去除无关的句子
        key_sentences = set()
        for s in sentences:
            for k in keywords:
                if k in s:
                    key_sentences.add(k)

        # 根据问题提取答案
        # 提取人名
        key_persons = self.key_items_by_pos(key_sentences)
        # 候选队列
        candidate_persons = []
        for i, v in enumerate(key_persons):
            # 去除问句中的关键词
            if v[i] not in keywords:
                candidate_persons.append(v)
        if candidate_persons:
            answer.extend(candidate_persons[:3])
        return answer

    @staticmethod
    def key_items_by_pos(sentences, pos='nr'):
        target_dict = {}
        for ks in sentences:
            words = postag(ks)
            for w in words:
                if w.flag == pos:
                    if w.word in target_dict:
                        target_dict[w.word] += 1
                    else:
                        target_dict[w.word] = 1
        # 找出最大词频
        sorted_list = sorted(target_dict.items(), key=lambda item: item[1], reverse=True)
        return sorted_list
