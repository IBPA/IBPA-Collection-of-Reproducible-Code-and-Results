from . import m_module_template

class MicroarrayRetrieval(m_module_template.MicroarrayModule):
    def __init__(self, owner):
        self.owner = owner
        
    def download_metadata(self):
        self.data = None #Fake
        
    def complete_data_independent_metadata(self):
        self.data = None #Fake
        
    def filter_entry(self):
        self.data = None #Fake
        
    def download_data(self):
        self.data = None #Fake