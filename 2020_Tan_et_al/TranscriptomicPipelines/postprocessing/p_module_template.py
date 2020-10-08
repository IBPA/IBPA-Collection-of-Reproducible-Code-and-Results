class PostprocessingModule:
    def __init__(self, owner):
        self.owner = owner
        
    def get_parameter_set(self):
        return self.owner.get_parameter_set()
        
    def get_t_compendium_collections(self):
        return self.owner.get_t_compendium_collections()
        
    def get_m_compendium(self):
        return self.owner.get_m_compendium()
        
    def get_s_compendium(self):
        return self.owner.get_s_compendium()
        

class PostprocessingSubModule(PostprocessingModule):
    def __init__(self, owner):
        self.owner = owner