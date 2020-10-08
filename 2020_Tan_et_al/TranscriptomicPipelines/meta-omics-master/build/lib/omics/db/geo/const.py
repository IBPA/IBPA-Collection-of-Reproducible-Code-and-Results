class FetcherConst(object):

    DB_GEO                  = 'GEO'
    
    # Entrez parameters
    TRANS                   = 'transcriptomics'
    MICRO                   = 'microbiomics'

    # exp types by omics
    ET_TRANS                = 'Expression profiling'
    # ET_GENOME
    # ET_MICRO
    # ET_PROTE

class ParserConst(object):

    ACC_PRE_LEN             =  3
    UID_LEN                 =  9
    UID_FILL                = '0'
    UID_PRE_GPL             = '1'
    UID_PRE_GSM             = '3'
    
    # str separators
    SEP_SPECIES             = '; '
    SEP_EXP_TYPE            = '; '
    SEP_GPL                 = ';'
    EMPTY                   = ''
    
    # dict tags
    D_TITLE                 = 'title'
    D_DESCRIPTION           = 'summary'
    D_PUB_PMID              = 'PubMedIds'
    D_PUB_TITLE             = 'Title'
    D_PUB_AUTHOR            = 'AuthorList'
    D_PUB_DATE              = 'PubDate'
    D_TECH_ACC              = 'GPL'
    D_TECH_NAME             = 'title'
    D_ACC                   = 'Accession'
    D_ACC_SECONDARY         = 'ExtRelations'
    D_ACC_SECONDARY_TYPE    = 'RelationType'
    D_ACC_SECONDARY_VALUE   = 'TargetObject'
    D_ACC_SECONDARY_SRA     = 'SRA'
    D_SAMPLE                = 'Samples'
    D_SPECIES               = 'taxon'
    D_EXP_TYPE              = 'gdsType'

    # tech data retrieval consts
    TECH_ID_PREFIX          = 'GPL'