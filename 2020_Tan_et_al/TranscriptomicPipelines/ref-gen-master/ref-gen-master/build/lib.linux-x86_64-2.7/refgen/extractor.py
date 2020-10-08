"""extractor object for reference genome pipeline"""

import pandas as pd

from .prep import Preprocessor
from .const import FtpConst
from .const import SepConst
from refgen.utils import ftp

class ExtractorConst(object):

    SUMM_COL_IDX_ASSEMBLY_ACC   = 0
    SUMM_COL_IDX_REFSEQ_CAT     = 4
    SUMM_COL_IDX_SPECIES        = 7
    SUMM_COL_IDX_LEVEL          = 11
    SUMM_COL_IDX_RELEASE        = 12
    SUMM_COL_IDX_GENOME         = 13
    SUMM_COL_IDX_GBRS           = 17
    SUMM_COL_IDX_FTP            = 19
    SUMM_COL_NAME_ASSEMBLY_ACC  = 'asm_acc'
    SUMM_COL_NAME_REFSEQ_CAT    = 'refseq_cat'
    SUMM_COL_NAME_SPECIES       = 'name'
    SUMM_COL_NAME_LEVEL         = 'level'
    SUMM_COL_NAME_RELEASE       = 'release'
    SUMM_COL_NAME_GENOME        = 'genome'
    SUMM_COL_NAME_GBRS          = 'gbrs'
    SUMM_COL_NAME_FTP           = 'ftp'

    REFSEQ_CAT_REP_GEN          = 'representative genome'
    REFSEQ_CAT_REF_GEN          = 'reference genome'
    LEVEL_COMP                  = 'Complete Genome'
    LEVEL_CHRO                  = 'Chromosome'
    LEVEL_SCAF                  = 'Scaffold'
    LEVEL_CONT                  = 'Contig'
    RELEASE_MAJOR               = 'Major'
    RELEASE_MINOR               = 'Minor'
    RELEASE_PATCH               = 'Patch'
    GENOME_FULL                 = 'Full'
    GENOME_PARTIAL              = 'Partial'
    NA                          = 'na'

    FILE_EXT_TYPE               = 'genomic'
    FILE_EXT_FORMAT             = 'gff'
    FILE_EXT_SUFFIX             = 'gz'

class Extractor(object):
    """extractor class for reference genome

    parameters
    ==========
    update_summary  : (bool, default = False) if True, it will redownload
        summary file from NCBI server which takes longer time. However, it will
        download a summary regardless for the first run of meta-omics

    attributes
    ==========
    prep            : (object) preprocessor"""

    def __init__(self, update_summary = False):
        self.prep = Preprocessor(update_summary)

    def _load_summary(self):
        """load assembly summary file to memory"""

        return pd.read_csv(
            self.prep.path_summary,
            header  = 1,
            sep     = SepConst.TAB,
            usecols = [
                ExtractorConst.SUMM_COL_IDX_ASSEMBLY_ACC,
                ExtractorConst.SUMM_COL_IDX_REFSEQ_CAT,
                ExtractorConst.SUMM_COL_IDX_SPECIES,
                ExtractorConst.SUMM_COL_IDX_LEVEL,
                ExtractorConst.SUMM_COL_IDX_RELEASE,
                ExtractorConst.SUMM_COL_IDX_GENOME,
                ExtractorConst.SUMM_COL_IDX_GBRS,
                ExtractorConst.SUMM_COL_IDX_FTP],
            names   = [
                ExtractorConst.SUMM_COL_NAME_ASSEMBLY_ACC,
                ExtractorConst.SUMM_COL_NAME_REFSEQ_CAT,
                ExtractorConst.SUMM_COL_NAME_SPECIES,
                ExtractorConst.SUMM_COL_NAME_LEVEL,
                ExtractorConst.SUMM_COL_NAME_RELEASE,
                ExtractorConst.SUMM_COL_NAME_GENOME,
                ExtractorConst.SUMM_COL_NAME_GBRS,
                ExtractorConst.SUMM_COL_NAME_FTP])

    def find_refgen(self, term, output = None):
        """look for the best refence genome candidate

        input:
            term                    : (str) str
            output                  : (str OR None) if None, return list
                otherwise download to the specified path
        output:
            (list of tuple OR None) : (assembly, name, ftp_url)"""

        df = self._load_summary()

        # 1. remove rows with gbrs, ftp columns of na
        df = df[df.ftp != ExtractorConst.NA]
        df = df[df.gbrs != ExtractorConst.NA]

        # 2. choose rows related to the term
        df = df[df.name.str.contains(term, regex = False) |
                df.name.str.contains(term.capitalize(), regex = False)]

        # 3. prioritize refseq_cat: representative -> reference -> na
        df_ = df[df.refseq_cat == ExtractorConst.REFSEQ_CAT_REP_GEN]
        if df_.shape[0] == 0:
            df_ = df[df.refseq_cat == ExtractorConst.REFSEQ_CAT_REF_GEN]
            if df_.shape[0] == 0:
                df_ = df[df.refseq_cat == ExtractorConst.NA]
        df = df_

        # 4. prioritize level: complete -> chromesome -> scaffold >- config
        df_ = df[df.level == ExtractorConst.LEVEL_COMP]
        if df_.shape[0] == 0:
            df_ = df[df.level == ExtractorConst.LEVEL_CHRO]
            if df_.shape[0] == 0:
                df_ = df[df.level == ExtractorConst.LEVEL_SCAF]
                if df_.shape[0] == 0:
                    df_ = df[df.level == ExtractorConst.LEVEL_CONT]
        df = df_

        # 5. prioritize release: major -> minor
        df_ = df[df.release == ExtractorConst.RELEASE_MAJOR]
        if df_.shape[0] == 0:
            df_ = df[df.release == ExtractorConst.RELEASE_MINOR]
            if df_.shape[0] == 0:
                df_ = df[df.release == ExtractorConst.RELEASE_PATCH]
        df = df_

        # 6. prioritize genome: full -> partial
        df_ = df[df.genome == ExtractorConst.GENOME_FULL]
        if df_.shape[0] == 0:
            df_ = df[df.genome == ExtractorConst.GENOME_PARTIAL]
        df = df_

        if output is None:
            return list(zip(df.asm_acc.tolist(), df.name.tolist(),
                df.ftp.tolist()))
        else:
            pd.concat([df.asm_acc, df.name, df.ftp], axis = 1).to_csv(output)

    def extract(self, ftp_url, output = None):
        """find the best reference genome of the term from the NCBI server and
        download the binary file to the specified local path

        input:
            ftp_url     : (str) used path directly from find_refgen
            output      : (str, default PATH = '.')"""

        # pick the first candid and parse the ftp path
        ftp_dir_path    = ftp_url[FtpConst.URL_PREFIX_LEN:]
        file_prefix     = ftp_dir_path.split(SepConst.SLASH)[-1]

        file_ext        = SepConst.DOT.join([
            ExtractorConst.FILE_EXT_TYPE,
            ExtractorConst.FILE_EXT_FORMAT,
            ExtractorConst.FILE_EXT_SUFFIX])

        file_gff_name   = SepConst.UNDERSCORE.join([
            file_prefix,
            file_ext])

        file_path       = SepConst.SLASH.join([
            ftp_dir_path,
            file_gff_name])

        if output is None:
            output  = '.'

        ftp.download(
            FtpConst.HOST,
            file_path,
            output,
            'b')