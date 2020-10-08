class JoinResultsException(Exception):
    pass

class InvalidFastqFilePathException(Exception):
    pass
    
class FailedToDownloadSRAFileException(Exception):
    pass
    
class FastqdumpFailedException(Exception):
    pass    
 
class FastqFileNotFoundException(Exception):
    pass 
    
class Bowtie2AlignmentFailedException(Exception):
    pass
    
class RSeQCInferExperimentFailedException(Exception):
    pass
    
class HTSeqCountFailedException(Exception):
    pass