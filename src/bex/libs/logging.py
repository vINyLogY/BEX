# coding: utf-8
"""Interface to logging package.
"""
import logging
import os
import sys
from typing import Literal, Optional


class Logger(object):
    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self,
                 filename: Optional[str] = None,
                 level: Literal['debug', 'info', 'warning', 'error', 'critical'] = 'info',
                 stream_fmt: Optional[str] = None,
                 file_fmt: str = '%(message)s'):
        if filename is None:
            # Use the same name of the main script as the default name
            filename = os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.log'
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self.levels[level])
        if stream_fmt is not None:
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter(stream_fmt))
            self.logger.addHandler(sh)
        th = logging.FileHandler(filename=filename, mode='w', encoding='utf-8')
        th.setFormatter(logging.Formatter(file_fmt))
        self.logger.addHandler(th)
