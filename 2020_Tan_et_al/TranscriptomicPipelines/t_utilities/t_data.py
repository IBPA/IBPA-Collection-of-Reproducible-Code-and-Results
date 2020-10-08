import sys
if (sys.version_info < (3, 0)):
    import t_data_exceptions
else:
    from . import t_data_exceptions

class TranscriptomeData:
    def __init__(self,test = ""):
        self.ori_data_matrix = None
        self.ori_data_matrix_path = ""
        self.test = test
        
        #Later you can impute the data matrix from each pipeline, not only merged matrix :)
        
    def update_ori_data_matrix(self, ori_data_matrix, ori_data_matrix_path):
        if self.test == 1:
            raise Exception('Why!!??')
        self.ori_data_matrix_path = ori_data_matrix_path
        self.ori_data_matrix = ori_data_matrix
        
    def output_ori_data_matrix(self):
        if self.ori_data_matrix is None:
            raise t_data_exceptions.OriDataMatrixIsNotReady('Original data matrix is not ready!')
            
        if self.ori_data_matrix_path == "":
            raise t_data_exceptions.InvalidOriDataMatrixPath('Invalid original data matrix path!')
        
        try:
            self.ori_data_matrix.to_csv(self.ori_data_matrix_path)
        except Exception as e:
            raise t_data_exceptions.FailedToWriteOriDataMatrix('Failed to write original data matrix')
    