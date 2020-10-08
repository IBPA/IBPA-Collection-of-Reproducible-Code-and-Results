class ValidationModule:
    def __init__(self, owner):
        self.owner = owner
        
    def get_t_metadata(self):
        return self.owner.get_t_metadata()
        
    def get_t_gene_annotation(self):
        return self.owner.get_t_gene_annotation()