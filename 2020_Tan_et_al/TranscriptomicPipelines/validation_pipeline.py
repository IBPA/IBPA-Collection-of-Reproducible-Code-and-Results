import sys
if (sys.version_info < (3, 0)):
    sys.path.insert(0, "validation")
    import supervised
    import unsupervised
    import v_module_template
else:
    import validation.supervised as supervised
    import validation.unsupervised as unsupervised
    import validation.v_module_template as v_module_template

class ValidationPipeline(v_module_template.ValidationModule):
    def __init__(self, owner):
        self.owner = owner
        self.supervised_validation = supervised.SupervisedValidation(self)
        self.unsupervised_validation = unsupervised.UnsupervisedValidation(self)
       
        self.configure_parameter_set()
       
    def configure_parameter_set_all(self):
        self.configure_parameter_set()
        self.supervised_validation.configure_parameter_set()
        self.unsupervised_validation.configure_parameter_set()
       
    def configure_parameter_set(self):
        return
        
    def run_validation_pipeline(self, input_corr_path = None, input_knowledge_capture_groupping_path = None, input_knowledge_capture_gene_list_path = None):
        self.unsupervised_validation.validate_data()
        print(input_corr_path)
        if input_corr_path:
            self.supervised_validation.correlation_validation(input_corr_path)
            
        if input_knowledge_capture_groupping_path and input_knowledge_capture_gene_list_path:
            self.supervised_validation.knowledge_capture_validation(input_knowledge_capture_groupping_path, input_knowledge_capture_gene_list_path)
            
            
        