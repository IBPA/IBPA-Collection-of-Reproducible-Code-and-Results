"""Parent class for all parser subclasses"""

class BaseParser(object):

    def __init__(self):
        pass

    def parse_title(self, data):
        pass

    def parse_publication(self, data):
        """get publication title, pmid, date, author information"""
        pass

    def parse_species(self, data):
        pass

    def parse_tech(self, data):
        """get experiment technology name and ID"""
        pass

    def parse_sample(self, data):
        """get sample IDs and secondary IDs if existed"""
        pass

    def parse_patients_num(self, data):
        pass

    def parse_patients_ctry(self, data):
        pass

    def parse_description(self, data):
        pass

    def parse_keywords(self, data):
        pass

    def parse_accession(self, data):
        """get secondary IDs if exsited"""
        pass

    def parse_exp_type(self, data):
        pass

    def parse(self, data):
        pass