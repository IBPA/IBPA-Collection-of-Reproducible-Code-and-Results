import numpy as np
class ValidationModule:
    def __init__(self, owner):
        self.owner = owner
        
    def get_parameter_set(self):
        return self.owner.get_parameter_set()
        
    def get_t_metadata(self):
        return self.owner.get_t_metadata()
        
    def get_t_gene_annotation(self):
        return self.owner.get_t_gene_annotation()
        
    def get_t_compendium_collections(self):
        return self.owner.get_t_compendium_collections()
        

class ValidationSubModule(ValidationModule):
    def __init__(self, owner):
        self.owner = owner

    def get_noise(self, matrix):
        #Get the noise by permute the data matrix
        noise = matrix
        noise = noise.reshape((matrix.shape[0]*matrix.shape[1],1))
        noise = np.random.permutation(noise)
        noise = noise.reshape(matrix.shape)

        return noise
