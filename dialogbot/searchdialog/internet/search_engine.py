# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 对百度、Bing 的搜索摘要进行答案的检索
"""

from urllib.request import quote

from dialogbot.searchdialog.internet import html_crawler
from dialogbot.utils.logger import logger
from dialogbot.utils.tokenizer import postag

baidu_url_prefix = 'https://www.bing.com/search?q='
bing_url_prefix = 'https://www.baidu.com/s?wd='
calendar_url = 'http://open.baidu.com/calendar'
calculator_url = 'http://open.baidu.com/static/calculator/calculator.html'


class Engine:
    def __init__(self, query, topk = 10):
        self.name = 'engine'
        self.query = query
        self.keywords = self.get_keywords(query)
        self.topk = topk
        self.text = ''

    @staticmethod
    def get_keywords(query):
        keywords = []
        words = postag(query)
        for k in words:
            # 只保留名词
            if k.flag.__contains__("n"):
                logger.debug(k.word)
                keywords.append(k.word)
        return keywords

    def search_baidu(self):
        answer = []
        # 抓取百度前10条的摘要
        soup_baidu = html_crawler.get_html_baidu(bing_url_prefix + quote(self.query))

        for i in range(1, self.topk):
            if not soup_baidu:
                break
            items = soup_baidu.find(id=i)

            if not items:
                logger.debug("百度摘要找不到答案")
                break
            # 判断是否有mu,如果第一个是百度知识图谱的 就直接命中答案
            if ('mu' in items.attrs) and i == 1:
                r = items.find(class_='op_exactqa_s_answer')
                if not r:
                    logger.debug("百度知识图谱找不到答案")
                else:
                    logger.debug(r.get_text().strip())
                    logger.debug("百度知识图谱找到答案")
                    answer.append(r.get_text().strip())
                    flag = 1
                    break

            # 古诗词判断
            if ('mu' in items.attrs) and i == 1:
                r = items.find(class_="op_exactqa_detail_s_answer")
                if not r:
                    logger.debug("百度诗词找不到答案")
                else:
                    logger.debug(r.get_text())
                    logger.debug("百度诗词找到答案")
                    answer.append(r.get_text().strip())
                    flag = 1
                    break

            # 万年历 & 日期
            if ('mu' in items.attrs) and i == 1 and items.attrs['mu'].__contains__(calendar_url):
                r = items.find(class_="op-calendar-content")
                if not r:
                    logger.debug("百度万年历找不到答案")
                else:
                    logger.debug("百度万年历找到答案")
                    answer.append(r.get_text().strip().replace("\n", "").replace(" ", ""))
                    flag = 1
                    break

            if ('tpl' in items.attrs) and i == 1 and items.attrs['tpl'].__contains__('calendar_new'):
                r = items.attrs['fk'].replace("6018_", "")
                logger.debug(r)
                if not r:
                    logger.debug("百度万年历新版找不到答案")
                else:
                    logger.debug("百度万年历新版找到答案")
                    answer.append(r)
                    flag = 1
                    break

            # 计算器
            if ('mu' in items.attrs) and i == 1 and items.attrs['mu'].__contains__(calculator_url):
                # r = results.find('div').find_all('td')[1].find_all('div')[1]
                r = items.find(class_="op_new_val_screen_result")
                if not r:
                    logger.debug("计算器找不到答案")
                else:
                    logger.debug("计算器找到答案")
                    answer.append(r.get_text().strip())
                    flag = 1
                    break

            # 百度知道答案
            if ('mu' in items.attrs) and i == 1:
                r = items.find(class_='op_best_answer_question_link')
                if not r:
                    logger.debug("百度知道图谱找不到答案")
                else:
                    logger.debug("百度知道图谱找到答案")
                    url = r['href']
                    zhidao_soup = html_crawler.get_html_zhidao(url)
                    r = zhidao_soup.find(class_='bd answer').find('pre')
                    if not r:
                        r = zhidao_soup.find(class_='bd answer').find(class_='line content')
                    answer.append(r.get_text())
                    flag = 1
                    break

            if items.find("h3"):
                # 百度知道
                if items.find("h3").find("a").get_text().__contains__(u"百度知道") and (i == 1 or i == 2):
                    url = items.find("h3").find("a")['href']
                    if not url:
                        logger.debug("百度知道图谱找不到答案")
                        continue
                    else:
                        logger.debug("百度知道图谱找到答案")
                        zhidao_soup = html_crawler.get_html_zhidao(url)

                        r = zhidao_soup.find(class_='bd answer')
                        if not r:
                            continue
                        else:
                            r = r.find('pre')
                            if not r:
                                r = zhidao_soup.find(class_='bd answer').find(class_='line content')
                        answer.append(r.get_text().strip())
                        flag = 1
                        break

                # 百度百科
                if items.find("h3").find("a").get_text().__contains__(u"百度百科") and (i == 1 or i == 2):
                    url = items.find("h3").find("a")['href']
                    if not url:
                        logger.debug("百度百科找不到答案")
                        continue
                    else:
                        logger.debug("百度百科找到答案")
                        baike_soup = html_crawler.get_html_baike(url)

                        r = baike_soup.find(class_='lemma-summary')
                        if not r:
                            continue
                        else:
                            r = r.get_text().replace("\n", "").strip()
                        answer.append(r)
                        flag = 1
                        break
            text += items.get_text()

        if flag == 1:
            return answer



def kwquery(query):
    # 分词 去停用词 抽取关键词
    keywords = []
    words = postag(query)
    for k in words:
        # 只保留名词
        if k.flag.__contains__("n"):
            logger.debug(k.word)
            keywords.append(k.word)

    answer = []
    text = ''
    # 找到答案就置1
    flag = 0

    # 抓取百度前10条的摘要
    soup_baidu = html_crawler.get_html_baidu(bing_url_prefix + quote(query))

    for i in range(1, 10):
        if not soup_baidu:
            break
        results = soup_baidu.find(id=i)

        if not results:
            logger.debug("百度摘要找不到答案")
            break
        # 判断是否有mu,如果第一个是百度知识图谱的 就直接命中答案
        if ('mu' in results.attrs) and i == 1:
            r = results.find(class_='op_exactqa_s_answer')
            if not r:
                logger.debug("百度知识图谱找不到答案")
            else:
                logger.debug(r.get_text().strip())
                logger.debug("百度知识图谱找到答案")
                answer.append(r.get_text().strip())
                flag = 1
                break

        # 古诗词判断
        if ('mu' in results.attrs) and i == 1:
            r = results.find(class_="op_exactqa_detail_s_answer")
            if not r:
                logger.debug("百度诗词找不到答案")
            else:
                logger.debug(r.get_text())
                logger.debug("百度诗词找到答案")
                answer.append(r.get_text().strip())
                flag = 1
                break

        # 万年历 & 日期
        if ('mu' in results.attrs) and i == 1 and results.attrs['mu'].__contains__(calendar_url):
            r = results.find(class_="op-calendar-content")
            if not r:
                logger.debug("百度万年历找不到答案")
            else:
                logger.debug("百度万年历找到答案")
                answer.append(r.get_text().strip().replace("\n", "").replace(" ", ""))
                flag = 1
                break

        if ('tpl' in results.attrs) and i == 1 and results.attrs['tpl'].__contains__('calendar_new'):
            r = results.attrs['fk'].replace("6018_", "")
            logger.debug(r)
            if not r:
                logger.debug("百度万年历新版找不到答案")
            else:
                logger.debug("百度万年历新版找到答案")
                answer.append(r)
                flag = 1
                break

        # 计算器
        if ('mu' in results.attrs) and i == 1 and results.attrs['mu'].__contains__(calculator_url):
            # r = results.find('div').find_all('td')[1].find_all('div')[1]
            r = results.find(class_="op_new_val_screen_result")
            if not r:
                logger.debug("计算器找不到答案")
            else:
                logger.debug("计算器找到答案")
                answer.append(r.get_text().strip())
                flag = 1
                break

        # 百度知道答案
        if ('mu' in results.attrs) and i == 1:
            r = results.find(class_='op_best_answer_question_link')
            if not r:
                logger.debug("百度知道图谱找不到答案")
            else:
                logger.debug("百度知道图谱找到答案")
                url = r['href']
                zhidao_soup = html_crawler.get_html_zhidao(url)
                r = zhidao_soup.find(class_='bd answer').find('pre')
                if not r:
                    r = zhidao_soup.find(class_='bd answer').find(class_='line content')
                answer.append(r.get_text())
                flag = 1
                break

        if results.find("h3"):
            # 百度知道
            if results.find("h3").find("a").get_text().__contains__(u"百度知道") and (i == 1 or i == 2):
                url = results.find("h3").find("a")['href']
                if not url:
                    logger.debug("百度知道图谱找不到答案")
                    continue
                else:
                    logger.debug("百度知道图谱找到答案")
                    zhidao_soup = html_crawler.get_html_zhidao(url)

                    r = zhidao_soup.find(class_='bd answer')
                    if not r:
                        continue
                    else:
                        r = r.find('pre')
                        if not r:
                            r = zhidao_soup.find(class_='bd answer').find(class_='line content')
                    answer.append(r.get_text().strip())
                    flag = 1
                    break

            # 百度百科
            if results.find("h3").find("a").get_text().__contains__(u"百度百科") and (i == 1 or i == 2):
                url = results.find("h3").find("a")['href']
                if not url:
                    logger.debug("百度百科找不到答案")
                    continue
                else:
                    logger.debug("百度百科找到答案")
                    baike_soup = html_crawler.get_html_baike(url)

                    r = baike_soup.find(class_='lemma-summary')
                    if not r:
                        continue
                    else:
                        r = r.get_text().replace("\n", "").strip()
                    answer.append(r)
                    flag = 1
                    break
        text += results.get_text()

    if flag == 1:
        return answer

    # 获取bing的摘要
    soup_bing = html_crawler.get_html_bing(bing_url_prefix + quote(query))
    # 判断是否在Bing的知识图谱中
    # bingbaike = soup_bing.find(class_="b_xlText b_emphText")
    bingbaike = soup_bing.find(class_="bm_box")

    if bingbaike:
        if bingbaike.find_all(class_="b_vList")[1]:
            if bingbaike.find_all(class_="b_vList")[1].find("li"):
                logger.debug("Bing知识图谱找到答案")
                answer.append(bingbaike.get_text())
                return answer
    else:
        logger.debug("Bing知识图谱找不到答案")
        results = soup_bing.find(id="b_results")
        bing_list = results.find_all('li')
        for bl in bing_list:
            temp = bl.get_text()
            if temp.__contains__(u" - 必应网典"):
                logger.debug("查找Bing网典")
                url = bl.find("h2").find("a")['href']
                if not url:
                    logger.debug("Bing网典找不到答案")
                    continue
                else:
                    logger.debug("Bing网典找到答案")
                    bingwd_soup = html_crawler.get_html_bingwd(url)

                    r = bingwd_soup.find(class_='bk_card_desc').find("p")
                    if not r:
                        continue
                    else:
                        r = r.get_text().replace("\n", "").strip()
                    answer.append(r)
                    flag = 1
                    break

        if flag == 1:
            return answer

        text += results.get_text()

    # 如果再两家搜索引擎的知识图谱中都没找到答案，那么就分析摘要
    if flag == 0:
        # 分句
        cutlist = [u"。", u"?", u".", u"_", u"-", u":", u"！", u"？"]
        temp = ''
        sentences = []
        for i in range(0, len(text)):
            if text[i] in cutlist:
                if temp == '':
                    continue
                else:
                    # logger.debug(temp
                    sentences.append(temp)
                temp = ''
            else:
                temp += text[i]

        # 找到含有关键词的句子,去除无关的句子
        key_sentences = {}
        for s in sentences:
            for k in keywords:
                if k in s:
                    key_sentences[s] = 1

        # 根据问题制定规则

        # 识别人名
        target_dict = {}
        for ks in key_sentences:
            # logger.debug(ks
            words = postag(ks)
            for w in words:
                logger.debug("=====")
                logger.debug(w.word)
                if w.flag == "nr":
                    if w.word in target_dict:
                        target_dict[w.word] += 1
                    else:
                        target_dict[w.word] = 1

        # 找出最大词频
        sorted_lists = sorted(target_dict, reverse=True)
        # logger.debug(len(target_list)
        # 去除问句中的关键词
        sorted_lists2 = []
        # 候选队列
        for i, st in enumerate(sorted_lists):
            # logger.debug(st[0]
            if st[0] in keywords:
                continue
            else:
                sorted_lists2.append(st)

        logger.debug("返回前n个词频")
        answer = []
        for i, st in enumerate(sorted_lists2):
            if i < 3:
                answer.append(st[0])

    return answer


if __name__ == '__main__':
    query = "姚明老婆是谁"
    ans = kwquery(query)
    logger.debug(ans)
