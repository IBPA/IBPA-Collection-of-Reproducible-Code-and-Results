from Bio import Entrez

from .list import split_list

class EntrezConst(object):

    # database abbr
    GEO             = 'gds'
    SRA             = 'sra'
    PUBMED          = 'pubmed'

    # dict tags
    D_IDLIST        = 'IdList'
    D_COUNT         = 'Count'

    # separaters
    SEP_ID          = ','

    # parameters
    SEARCH_MAX      =  100000
    SUMMARY_MAX     =  10000

def set_config(email, key):
    """set email and api key configuration for Entrez"""

    Entrez.email    = email
    Entrez.api_key  = key

def get_esearch(db, query):
    """get IDs returned by esearch"""

    id_list = []
    start   = 0
    while True:
        handle = Entrez.esearch(
            db          = db,
            retmax      = EntrezConst.SEARCH_MAX,
            retstart    = start,
            term        = query)
        record = Entrez.read(handle)
        id_list.extend(record[EntrezConst.D_IDLIST])
        if len(id_list) == int(record[EntrezConst.D_COUNT]):
            break
        else:
            start += ret_max
    return id_list

def get_esummary(db, id_list):
    """get data returned by esummary"""

    data_list = []
    id_batch_list = split_list(id_list, EntrezConst.SUMMARY_MAX)
    for batch in id_batch_list:
        handle = Entrez.esummary(
            db = db,
            id = EntrezConst.SEP_ID.join([str(id_) for id_ in batch]))
        records = Entrez.parse(handle)
        data_list.extend(list(records))
    return data_list