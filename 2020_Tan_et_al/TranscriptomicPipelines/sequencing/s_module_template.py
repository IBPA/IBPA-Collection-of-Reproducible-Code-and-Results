
class SequencingModule:
    def __init__(self, owner):
        self.owner = owner
        
    def get_parameter_set(self):
        return self.owner.get_parameter_set()
        
    def get_t_gene_annotation(self):
        return self.owner.get_t_gene_annotation()
        
    def get_general_parameters(self):
        return self.owner.get_general_parameters()
        
    def get_general_constant(self):
        return self.owner.get_general_constant()
        
    def get_parallel_engine(self):
        return self.owner.get_parallel_engine()
        
class SequencingSubModule(SequencingModule):
    def __init__(self, owner):
        self.owner = owner
        
    def get_s_query_id(self):
        return self.owner.get_s_query_id()
        
    def get_s_metadata(self):
        return self.owner.get_s_metadata()
        
    def get_s_data(self):
        return self.owner.get_s_data()
        
    def get_bowtie2_parameters(self):
        return self.owner.get_bowtie2_parameters()
    
    def get_sratool_parameters(self):
        return self.owner.get_sratool_parameters()
        
    def get_rseqc_parameters(self):
        return self.owner.get_rseqc_parameters()
        
    def get_htseq_parameters(self):
        return self.owner.get_htseq_parameters()