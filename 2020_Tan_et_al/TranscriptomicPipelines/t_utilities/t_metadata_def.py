import sys
if (sys.version_info < (3, 0)):
    import t_metadata_exceptions
else:
    from . import t_metadata_exceptions
#t_metadata_def.py
#Define each field of metadata table
from enum import Enum
import datetime

import pandas as pd

class MetadataTableColumns(Enum):
    __order__ = 'CHANNEL_NUM FIRST_CHANNEL SECOND_CHANNEL SOURCE_TYPE SERIES_ID PLATFORM_ID REMOVED USED_DATA PAIRED STRANDED BG_AVAILABLE USED_VALUE VALUE_VALID PMID AUTHOR DATE'
    CHANNEL_NUM         = "channel_num"
    FIRST_CHANNEL       = "first_channel"
    SECOND_CHANNEL      = "second_channel"
    SOURCE_TYPE         = "source_type"
    SERIES_ID           = "series_id"
    PLATFORM_ID         = "platform_id"
    REMOVED             = "removed"
    USED_DATA           = "used_data"
    PAIRED              = "paired"
    STRANDED            = "stranded"
    BG_AVAILABLE        = "bg_available"
    USED_VALUE          = "used_value"
    VALUE_VALID         = "value_valid"
    PMID                = "pmid"
    AUTHOR              = "author"
    DATE                = "date"
    

class Separator(Enum):
    CHANNEL_NUM     = "_"
    SERIES          = "/"

class SourceType(Enum):
    GEO             = "GEO"
    ARRAYEXPRESS    = "ARRAYEXPRESS"
    SRA             = "SRA"
    MICROARRAY      = "MICROARRAY"
    SEQUENCING      = "SEQUENCING"
    ALL             = "ALL"
    
class ChannelNum(Enum):
    ONE             = "ONE"
    TWO             = "TWO"
    NA              = "NA"
    
class Channel(Enum):
    CY3             = "cy3"
    CY5             = "cy5"
    UNKNOWN         = "unknown"
    NA              = "NA"
    

class RemovedIndicator(Enum):
    TRUE            = "TRUE"
    FALSE           = "FALSE"
    
class UsedDataType(Enum):
    GEOSOFT         = "GEOSOFT"
    RAW             = "RAW"
    SEQUENCING      = "SEQUENCING"
    
class PairedType(Enum):
    PAIRED          = "PAIRED"
    UNPAIRED        = "UNPAIRED"
    NA              = "NA"
    
class StrandedType(Enum):
    STRANDED        = "STRANDED"
    UNSTRANDED      = "UNSTRANDED"
    NA              = "NA"
    
class BGAvailableIndicator(Enum):
    TRUE            = "TRUE"
    FALSE           = "FALSE"
    NA              = "NA"
    
class UsedValueType(Enum):
    MEDIAN          = "MEDIAN"
    MEAN            = "MEAN"
    UNKNOWN         = "UNKNOWN"
    NA              = "NA"
    
class ValueValidIndicator(Enum):
    TRUE            = "TRUE"
    FALSE           = "FALSE"
    


class SampleID:
    #For SRA data, we have to take care of cases with multiple runs...
    def __init__(   self, 
                    source_type = SourceType.GEO.value,
                    channel_num  = ChannelNum.TWO.value,
                    channel = Channel.CY3.value,
                    series_id = "", #Only for arrayexpress data
                    experiment_id = ""
                    ):
        self.source_type = source_type
        self.channel_num = channel_num
        self.channel = channel
        self.series_id = series_id
        self.experiment_id = experiment_id
        self.id = ""
        
    def create_id(self):
        if self.source_type == SourceType.GEO.value:
            self.id = add_channel_surfix(self.experiment_id)
            
        elif self.source_type == SourceType.ARRAYEXPRESS.value:
            if not series_id:
                raise t_metadata_exceptions.FailedToCreateID("Series id should be provided to create id for a ArrayExpress entry")
            self.id = add_channel_surfix(self.series_id + Separator.SERIES.value + self.experiment_id)
            
        elif self.source_type == SourceType.SRA.value or self.source_type == SourceType.SEQUENCING.value:
            #SRA data:
            #NOTE: One experiment should be mapped into only ONE entry even there are more than one runs
            self.id = self.experiment_id
            
        else:
            raise t_metadata_exceptions.FailedToCreateID("Invalid Source Type")
        
    
    def add_channel_surfix(self, prefix_id):
        if self.channel == Channel.CY3.value:
            result = prefix_id + Separator.CHANNEL_NUM.value + Channel.CY3.value
        elif self.channel == Channel.CY5.value:
            result = prefix_id + Separator.CHANNEL_NUM.value + Channel.CY5.value
        else:
            result = prefix_id
        return(result)
        
        
class MetadataEntry:
    #To create one new entry:
    #1. Initiate the new Sample ID after you got the necessary information
    #2. Fill the necessary elements
    def __init__(self,
                sample_id = SampleID(), #Should be prepared well before you call the constructor
                channel_num = ChannelNum.TWO.value,
                first_channel = Channel.CY3.value,
                second_channel = Channel.CY5.value,
                source_type = SourceType.GEO.value,
                series_id = "",
                platform_id = "",
                removed = RemovedIndicator.FALSE.value,
                used_data = UsedDataType.GEOSOFT.value,
                paired = PairedType.NA.value,
                stranded = StrandedType.NA.value,
                bg_available = BGAvailableIndicator.TRUE.value,
                used_value = UsedValueType.MEDIAN.value,
                value_valid = ValueValidIndicator.TRUE.value,
                pmid = [], #Should be prepared well before you call the constructor
                author = [], #Should be prepared well before you call the constructor
                date = datetime.date(1,1,1) #Should be prepared well before you call the constructor
                ):
        
        self.sample_id = sample_id
        self.channel_num = channel_num
        self.first_channel = first_channel
        self.second_channel = second_channel
        self.source_type = source_type
        self.series_id = series_id
        self.platform_id = platform_id
        self.removed = removed
        self.used_data = used_data
        self.paired = paired
        self.stranded = stranded
        self.bg_available = bg_available
        self.used_value = used_value
        self.value_valid = value_valid
        self.pmid = pmid
        self.author = author
        self.date = date
        
    def to_df(self):
        columns = [e.value for e in MetadataTableColumns]
        
    
        df = pd.DataFrame([ [self.channel_num,
                            self.first_channel,
                            self.second_channel,
                            self.source_type,
                            self.series_id,
                            self.platform_id,
                            self.removed,
                            self.used_data,
                            self.paired,
                            self.stranded,
                            self.bg_available,
                            self.used_value,
                            self.value_valid,
                            "|".join(self.pmid),
                            "|".join(self.author),
                            self.date.strftime("%m/%d/%Y")]
                            ],
                            index = [self.sample_id.id],
                            columns = columns)
        return df
        
    def set_remove_ind(self):
        self.removed = RemovedIndicator.TRUE.value
