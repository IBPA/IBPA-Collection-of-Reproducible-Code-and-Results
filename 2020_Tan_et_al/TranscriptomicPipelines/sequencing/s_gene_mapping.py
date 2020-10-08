import sys
if (sys.version_info < (3, 0)):
    import s_module_template
    import s_gene_mapping_exceptions
else:
    from . import s_module_template
    from . import s_gene_mapping_exceptions

class SequencingGeneMappingParameters:
    def __init__(self, 
                    drop_unnamed_genes = False,
                    use_gene_names = False,
                    data_matrix_table_path = 'SequencingDataMatrix.csv',
                    gene_mapping_table_path = 'SequencingGeneMappingTable.csv'
                ):
        self.drop_unnamed_genes = drop_unnamed_genes
        self.use_gene_names = use_gene_names
        self.data_matrix_table_path = data_matrix_table_path
        self.gene_mapping_table_path = gene_mapping_table_path
        
class SequencingGeneMappingResults:
    def __init__(self):
        self.original_data_matrix = None
        
    def update_original_data_matrix(self, original_data_matrix):
        self.original_data_matrix = original_data_matrix

class SequencingGeneMapping(s_module_template.SequencingSubModule):
    def __init__(self, owner):
        self.owner = owner
        self.s_sample_mapping_results = owner.get_s_sample_mapping_results()
        self.parameters = SequencingGeneMappingParameters()
        self.results = SequencingGeneMappingResults()
        
        self.configure_parameter_set()
        
    def do_gene_mapping(self):
        self.map_gene()
        
    def configure_parameter_set(self):
        parameter_set = self.get_parameter_set()
        self.parameters.drop_unnamed_genes              = parameter_set.s_gene_mapping_parameters_drop_unnamed_genes
        self.parameters.use_gene_names                  = parameter_set.s_gene_mapping_parameters_use_gene_names
        self.parameters.data_matrix_table_path          = parameter_set.s_gene_mapping_parameters_data_matrix_table_path
        self.parameters.gene_mapping_table_path         = parameter_set.s_gene_mapping_parameters_gene_mapping_table_path
        
    def map_gene(self):
        gene_mapping_table = self.owner.get_t_gene_annotation().get_gene_mapping_table()
        colname_id = self.owner.get_t_gene_annotation().get_gene_mapping_table_colname_id()
        colname_gene_name = self.owner.get_t_gene_annotation().get_gene_mapping_table_colname_gene_name()
        
        gene_mapping_table_selected = gene_mapping_table[[colname_id,colname_gene_name]]
        try:
            gene_mapping_table_selected.to_csv(self.parameters.gene_mapping_table_path)
        except Exception as e:
            raise s_gene_mapping_exceptions.FailedToWriteGeneMappingTable('Failed to write gene mapping table!')
        
        count_reads_matrix = self.s_sample_mapping_results.count_reads_matrix
        indices = count_reads_matrix.index.tolist()
        
        if self.parameters.use_gene_names == True:
            gene_mapping_table_selected_dict = {}
            for index, row in gene_mapping_table_selected.iterrows():
                gene_mapping_table_selected_dict[row[colname_id]] = row[colname_gene_name]
            #USE GENE NAMES ==> rpoS, spoT, ...
            #NOT USE GENE NAMES ==> STMXXXX
            for i in range(len(indices)):
                indices[i] = self.find_gene_name(indices[i], gene_mapping_table_selected_dict)
            
        count_reads_matrix.index = indices
        self.results.update_original_data_matrix(count_reads_matrix)
        
        
        #Update the compendium part
        s_data = self.get_s_data()
        s_data.update_ori_data_matrix(count_reads_matrix, self.parameters.data_matrix_table_path)
        s_data.output_ori_data_matrix()

            
    def find_gene_name(self, index, gene_mapping_table_dict):
        if gene_mapping_table_dict[index] != "":
            return gene_mapping_table_dict[index]
        else:
            return index
            
            
        
        