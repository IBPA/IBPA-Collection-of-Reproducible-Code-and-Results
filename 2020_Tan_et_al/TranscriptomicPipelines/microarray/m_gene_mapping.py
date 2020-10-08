from . import m_module_template

class MicroarrayGeneMapping(m_module_template.MicroarrayModule):
    def __init__(self, owner):
        self.owner = owner
        
    def add_tag(self):
        self.data = None #Fake
        
    def map_gene(self):
        self.data = None #Fake
    
    def merge_different_platform_data(self):
        self.data = None #Fake