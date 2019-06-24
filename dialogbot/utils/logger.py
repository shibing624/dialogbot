# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import logging


def get_logger(name, log_file=None):
    """
    logger
    :param name: 模块名称
    :param log_file: 日志文件，如无则输出到标准输出
    :return:
    """
    formatter = logging.Formatter('[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s',
                                  datefmt='%m%d%Y %I:%M:%sS')
    if not log_file:
        handle = logging.StreamHandler()
    else:
        handle = logging.FileHandler(log_file)
    handle.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(handle)
    logger.setLevel(logging.DEBUG)
    return logger


def log_print(text, log_file=None, level='INFO'):
    from dialogbot import config
    log_file = log_file if log_file else config.log_file
    logger = get_logger(__name__, log_file)
    print(text)
    level = level.upper()
    if level == 'INFO':
        logger.info(text)
    elif level == 'ERROR':
        logger.error(text)
    elif level == 'DEBUG':
        logger.debug(text)
    elif level == 'WARNING' or level == 'WARN':
        logger.warning(text)
    else:
        logger.info(text)


def start_heartbeat(interval=60, logger=None):
    import time
    import threading

    def print_time():
        t = time.strftime('%Y-%m-%d %H:%M:%S - heartbeat', time.localtime(time.time()))
        print(t)
        if logger:
            logger.info(t)
        timer = threading.Timer(interval, print_time)
        timer.start()

    timer = threading.Timer(interval, print_time)
    timer.start()