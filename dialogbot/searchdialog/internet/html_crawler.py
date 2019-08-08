# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import requests
from bs4 import BeautifulSoup



def get_html_zhidao(url):
    """
    获取百度知道的页面
    :param url:
    :return:
    """
    headers = {'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686)Gecko/20071127 Firefox/2.0.0.11'}
    result = BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")
    return result


def get_html_baike(url):
    """
    获取百度百科的页面
    :param url:
    :return:
    """
    headers = {'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686)Gecko/20071127 Firefox/2.0.0.11'}
    return BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")


def get_html_bingwd(url):
    """
    获取Bing网典的页面
    :param url:
    :return:
    """
    headers = {'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686)Gecko/20071127 Firefox/2.0.0.11'}
    return BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")


def get_html_baidu(url):
    """
    获取百度搜索的结果
    :param url:
    :return:
    """
    headers = {'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686)Gecko/20071127 Firefox/2.0.0.11'}
    return BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")


def get_html_bing(url):
    """
    获取Bing搜索的结果
    :param url:
    :return:
    """
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'}
    return BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")
