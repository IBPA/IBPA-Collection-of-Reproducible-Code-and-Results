"""fetcher object for NCBI GEO"""

from Bio import Entrez

from .const import FetcherConst
from .parser import Parser
from omics.db._base import BaseFetcher
from omics.utils.entrez import EntrezConst
from omics.utils.entrez import set_config
from omics.utils.entrez import get_esearch
from omics.utils.entrez import get_esummary

class Fetcher(BaseFetcher):

    def __init__(self, email=None, api_key=None):
        super(Fetcher, self).__init__(
            email,
            api_key)
        self.parser = Parser()

    def _get_query(self, term):
        """compose query for NCBI for series data only"""
        
        return term + ' AND "gse"[Entry Type]'

    def get(self, term):
        """get data related to the term from server

        input:
            term : (str)
        output:
            data : (list of dict)"""

        set_config(self.email, self.api_key)
        id_list = get_esearch(EntrezConst.GEO, self._get_query(term))
        data    = get_esummary(EntrezConst.GEO, id_list)
        return data

    def fetch(self, term):
        """return parsed metadata related to the term

        input:
            term        : (str)
        output:
            metadata    : (list of lists)"""

        metadata = []
        for row in zip(*self.parser.parse(self.get(term))):
            metadata.append([FetcherConst.DB_GEO] + list(row))
        return metadata