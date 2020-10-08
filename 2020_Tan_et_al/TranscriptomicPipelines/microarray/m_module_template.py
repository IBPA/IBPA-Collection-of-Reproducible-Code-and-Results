class MicroarrayModule:
    def __init__(self, owner):
        self.owner = owner
        
    def get_parameter_set(self):
        return self.owner.get_parameter_set()
        
    def get_m_query_id(self):
        return self.owner.get_m_query_id()
        
    def get_t_metadata(self):
        return self.owner.get_t_metadata()
        
    def set_t_metadata(self,metadata):
        return self.owner.set_t_metadata(metadata)
        
    def get_t_gene_annotation(self):
        return self.owner.get_t_gene_annotation()