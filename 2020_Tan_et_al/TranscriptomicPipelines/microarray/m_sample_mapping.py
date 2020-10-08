from . import m_module_template

class MicroarraySampleMapping(m_module_template.MicroarrayModule):
    def __init__(self, owner):
        self.owner = owner
        
    def split_channel(self):
        self.data = None #Fake
        
    def merge_same_platform_data(self):
        self.data = None #Fake