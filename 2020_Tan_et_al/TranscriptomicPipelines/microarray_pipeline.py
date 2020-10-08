import microarray.m_data_retrieval as m_data_retrieval
import microarray.m_value_extraction as m_value_extraction
import microarray.m_sample_mapping as m_sample_mapping
import microarray.m_gene_mapping as m_gene_mapping
import microarray.m_module_template as m_module_template


class MicroarrayPipeline(m_module_template.MicroarrayModule):
    def __init__(self, owner, m_query_id, m_compendium):
        self.owner = owner
        
        self.m_query_id = m_query_id
        self.m_compendium = m_compendium
        
        self.m_data_retrieval = m_data_retrieval.MicroarrayRetrieval(self)
        self.m_value_extraction = m_value_extraction.MicroarrayExtraction(self)
        self.m_sample_mapping = m_sample_mapping.MicroarraySampleMapping(self)
        self.m_gene_mapping = m_gene_mapping.MicroarrayGeneMapping(self)
        
    
        
        
