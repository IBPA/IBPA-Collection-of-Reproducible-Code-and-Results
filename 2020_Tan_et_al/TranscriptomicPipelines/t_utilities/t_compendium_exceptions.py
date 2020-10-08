#Definition of metadata exceptions
class FailedToWriteMergedMetadataTable(Exception):
    pass

class FailedToWriteMergedDataMatrix(Exception):
    pass
    
class FailedToWriteImputedDataMatrix(Exception):
    pass
    
class FailedToWriteNormalizedDataMatrix(Exception):
    pass