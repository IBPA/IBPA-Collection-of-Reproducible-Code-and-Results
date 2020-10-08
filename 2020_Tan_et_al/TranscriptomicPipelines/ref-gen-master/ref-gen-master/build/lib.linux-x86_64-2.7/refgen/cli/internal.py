"""internal use only, will be removed in the future"""

import sys

from refgen import Extractor

def ref_find():
    term = sys.argv[1]
    output = sys.argv[2]

    extr = Extractor()
    return extr.find_refgen(term, output)

def ref_download():
    ftp_url = sys.argv[1]
    output = sys.argv[2]

    extr = Extractor()
    extr.extract(ftp_url, output)