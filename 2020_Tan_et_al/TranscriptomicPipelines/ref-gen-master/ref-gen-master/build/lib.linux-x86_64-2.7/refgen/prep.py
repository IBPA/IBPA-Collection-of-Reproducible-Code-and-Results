"""Preprocessing object for reference genome pipeline"""

import os

from .const import FtpConst
from .const import SepConst
from refgen.utils import ftp

class PreprocessConst(object):

    DATA_DIR        = 'data'
    DATA_GFF        = 'gff'
    DATA_SUMMARY    = 'summary.txt'

class Preprocessor(object):
    """preprocessor object for reference genome extractor module

    parameters
    ==========
    update_summary  : (bool, default = False) if True, it will redownload
        summary file from NCBI server which takes longer time. However, it will
        download a summary regardless for the first run of meta-omics

    attributes
    ==========
    path_dir        : (str) absolute path to parent dir
    path_data       : (str) absolute path to data dir
    path_summary    : (str) absolute path to summary file"""

    def __init__(self, update_summary = False):
        self.update_summary = update_summary
        self._preprocess()

    def _download_summary(self):
        ftp.download(
            FtpConst.HOST,
            FtpConst.PATH_SUMMARY,
            self.path_summary,
            't')

    def _init_data(self):
        """initialize an empty directory for the first time, unless specified"""

        if not os.path.exists(self.path_data):
            os.mkdir(self.path_data)
            self._download_summary()
        elif self.update_summary:
            self._download_summary()

    def _preprocess(self):
        self.path_dir       = os.path.dirname(os.path.abspath(__file__))
        self.path_data      = SepConst.SLASH.join([
            self.path_dir,
            PreprocessConst.DATA_DIR])
        self.path_summary   = SepConst.SLASH.join([
            self.path_data,
            PreprocessConst.DATA_SUMMARY])

        self._init_data()