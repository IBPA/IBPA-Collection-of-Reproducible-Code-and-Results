import sys
if (sys.version_info < (3, 0)):
    import t_metadata_def
    import t_metadata_exceptions
else:
    from . import t_metadata_def
    from . import t_metadata_exceptions

import datetime
import pandas as pd


class TranscriptomeMetadata:
    def __init__(   self, 
                    query_date = datetime.date(1,1,1),
                    query_string = "",
                    query_ids = []
                    ):
        self.source_type = None #{Microarray, Sequencing, All}
        self.query_date = query_date #A datetime object
        self.query_string = query_string
        self.query_ids = query_ids
        self.entries = {} #(A List with entries)
        self.entries_removed = {}

        
    def configure_sra(self):
        self.source_type = t_metadata_def.SourceType.SRA.value
        
    def configure_sequencing(self):
        self.source_type = t_metadata_def.SourceType.SEQUENCING.value
        
    def new_sequencing_entry(self, 
                                platform_id, 
                                series_id, 
                                experiment_id):
        if self.source_type != t_metadata_def.SourceType.SRA.value and self.source_type != t_metadata_def.SourceType.SEQUENCING.value:
            raise t_metadata_exceptions.FailedToCreateID('Incorrect SourceType for calling this entry constructor!')
        
        sample_id = t_metadata_def.SampleID(self.source_type, 
                                            t_metadata_def.ChannelNum.NA.value,
                                            t_metadata_def.Channel.NA.value,
                                            series_id, #SRPXXXXXX
                                            experiment_id)
        sample_id.create_id()
        if sample_id in self.entries.keys():
            #Duplicate
            return
            
        metadata_entry = t_metadata_def.MetadataEntry(  sample_id,                                     
                                                        channel_num = t_metadata_def.ChannelNum.NA.value,
                                                        first_channel = t_metadata_def.Channel.NA.value,
                                                        second_channel = t_metadata_def.Channel.NA.value,
                                                        source_type = self.source_type,
                                                        series_id = series_id,
                                                        platform_id = platform_id,
                                                        removed = t_metadata_def.RemovedIndicator.FALSE.value,
                                                        used_data = t_metadata_def.UsedDataType.SEQUENCING.value,
                                                        paired = t_metadata_def.PairedType.NA.value,
                                                        stranded = t_metadata_def.StrandedType.NA.value,
                                                        bg_available = t_metadata_def.BGAvailableIndicator.NA.value,
                                                        used_value = t_metadata_def.UsedValueType.NA.value,
                                                        value_valid = t_metadata_def.ValueValidIndicator.TRUE.value,
                                                        pmid = [], #Should be prepared well before you call the constructor
                                                        author = [], #Should be prepared well before you call the constructor
                                                        date = datetime.date(1900,1,1) #Should be prepared well before you call the constructor
                                                        )
        self.entries[sample_id.id] = metadata_entry
        
    def update_sequencing_entry_data_dependent_info(self, experiment_id, paired = False, stranded = False):
        if paired == True:
            self.entries[experiment_id].paired = t_metadata_def.PairedType.PAIRED.value
        else:
            self.entries[experiment_id].paired = t_metadata_def.PairedType.UNPAIRED.value
            
        if stranded == True:
            self.entries[experiment_id].stranded = t_metadata_def.StrandedType.STRANDED.value
        else:
            self.entries[experiment_id].stranded = t_metadata_def.StrandedType.UNSTRANDED.value
        
    def get_table(self):
        df = []
        for exp in self.entries:
            df.append(self.entries[exp].to_df())
        if len(df) == 0:
            return None
        return pd.concat(df)
        
    