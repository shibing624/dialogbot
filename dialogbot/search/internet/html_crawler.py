# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.16 Safari/537.36",
}


def get_html_zhidao(url):
    """
    获取百度知道的页面
    :param url:
    :return:
    """
    result = BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")
    return result


def get_html_baike(url):
    """
    获取百度百科的页面
    :param url:
    :return:
    """
    return BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")


def get_html_bingwd(url):
    """
    获取Bing网典的页面
    :param url:
    :return:
    """
    return BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")


def get_html_baidu(url):
    """
    获取百度搜索的结果
    :param url:
    :return:
    """
    return BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")


def get_html_bing(url):
    """
    获取Bing搜索的结果
    :param url:
    :return:
    """
    return BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")
