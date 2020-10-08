from . import m_module_template

class MicroarrayExtraction(m_module_template.MicroarrayModule):
    def __init__(self, owner):
        self.owner = owner
        
    def extract_column(self):
        self.data = None
        
    def complete_data_dependent_metadata(self):
        self.data = None
        
    def correct_background_intensities(self):
        self.data = None