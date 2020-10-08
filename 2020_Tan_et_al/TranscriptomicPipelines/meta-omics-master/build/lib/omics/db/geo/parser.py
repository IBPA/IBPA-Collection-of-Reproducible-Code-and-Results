"""Parser object for NCBI GEO"""

from .const import ParserConst
from omics.db._base import BaseParser
from omics.utils.entrez import EntrezConst
from omics.utils.entrez import get_esummary

class Parser(BaseParser):

    def _convert_gpl_to_uid(self, gpl):
        """gpl is a string of GEO platform ID with only the numerical part"""
        
        fill_str = ParserConst.UID_FILL * (ParserConst.UID_LEN - len(gpl) - 1)
        return ParserConst.UID_PRE_GPL + fill_str + gpl

    def _convert_gsm_to_uid(self, gsm):
        """gsm is a string of GEO sample ID with prefix GSM"""

        gsm_suffix  = gsm[ParserConst.ACC_PRE_LEN:]
        fill_str    = ParserConst.UID_FILL * (ParserConst.UID_LEN - len(gsm_suffix) - 1)
        return ParserConst.UID_PRE_GSM + fill_str + gsm_suffix

    def parse_title(self, data):
        title_list = []
        for row in data:
            title_list.append(row[ParserConst.D_TITLE])
        return title_list

    def parse_description(self, data):
        description_list = []
        for row in data:
            description_list.append(row[ParserConst.D_DESCRIPTION])
        return description_list

    def parse_publication(self, data):
        """get publication title, pmid, date, author information"""

        pmid_list       = []
        pub_title_list  = []
        date_list       = []
        author_list     = []
        
        # get pmid_list directly first
        pmid_concat = []
        pmid_count  = []
        for row in data:
            pmids = [int(id_) for id_ in row[ParserConst.D_PUB_PMID]]
            pmid_list.append(pmids)
            pmid_concat.extend(pmids)
            pmid_count.append(len(pmids))

        # get other metadata by searching in pubmed database
        data_pub = get_esummary(EntrezConst.PUBMED, pmid_concat)

        # allocate and map the concat data to pmid_list
        start = 0
        for count in pmid_count:
            pub_title_batch = []
            date_batch      = []
            author_batch    = []
            for row in data_pub[start:start + count]:
                pub_title_batch.append(row[ParserConst.D_PUB_TITLE])
                date_batch.append(row[ParserConst.D_PUB_DATE])
                author_batch.append(row[ParserConst.D_PUB_AUTHOR])
            pub_title_list.append(pub_title_batch)
            date_list.append(date_batch)
            author_list.append(author_batch)
            start += count
        return pmid_list, pub_title_list, date_list, author_list

    def parse_tech(self, data):
        """get experiment technology name and ID"""
        
        tech_id_list    = []
        tech_name_list  = []
        
        # get tech_id_list directly from data
        tech_uid_concat  = []
        tech_uid_count   = []
        for row in data:
            tech_id_str = row[ParserConst.D_TECH_ACC]
            tech_id_raw = tech_id_str.split(ParserConst.SEP_GPL)
            tech_id     = [id_ for id_ in tech_id_raw if len(id_)]
            tech_uid_concat.extend(self._convert_gpl_to_uid(id_) for id_ in tech_id)
            tech_uid_count.append(len(tech_id))
            tech_id_list.append([ParserConst.TECH_ID_PREFIX + id_ for id_ in tech_id])
        
        # get tech_name_list using tech_uid_concat and tech_uid_count
        tech_data = get_esummary(EntrezConst.GEO, tech_uid_concat)
        start = 0
        for count in tech_uid_count:
            tech_data_batch = tech_data[start:start + count]
            tech_name_list.append([row[ParserConst.D_TECH_NAME] for row in tech_data_batch])
            start += count
        return tech_id_list, tech_name_list

    def parse_accession(self, data):
        """get secondary IDs if exsited"""
        
        accession_list = []
        for row in data:
            acc_primary         = row[ParserConst.D_ACC]
            acc_secondary_list  = []
            for acc in row[ParserConst.D_ACC_SECONDARY]:

                # append SRP acc if found
                if acc[ParserConst.D_ACC_SECONDARY_TYPE] == ParserConst.D_ACC_SECONDARY_SRA:
                    acc_secondary_list.append(acc[ParserConst.D_ACC_SECONDARY_VALUE])
                
                # alarm me if NCBI changes in the future
                else:
                    raise Exception("Developping bugs. ExtRalations have more than SRA accession")
            accession_list.append([acc_primary] + acc_secondary_list)
        return accession_list

    def parse_sample(self, data):
        """get sample IDs"""

        sample_list = []

        # get all GSM IDs directly
        gsm_acc_concat = []
        gsm_uid_concat = []
        gsm_count = []
        for row in data:
            gsm_list = []
            for sample_data in row[ParserConst.D_SAMPLE]:
                gsm_list.append(sample_data[ParserConst.D_ACC])
            gsm_acc_concat.extend(gsm_list)
            gsm_uid_concat.extend([self._convert_gsm_to_uid(gsm) for gsm in gsm_list])
            gsm_count.append(len(gsm_list))

        # get SRX IDs corresponding to related GSM IDs using gsm_uid_concat and gsm_count
        srx_data = get_esummary(EntrezConst.GEO, gsm_uid_concat)
        srx_acc_concat = []
        for row in srx_data:
            found = 0
            for acc in row[ParserConst.D_ACC_SECONDARY]:

                # append SRX acc if found
                if acc[ParserConst.D_ACC_SECONDARY_TYPE] == ParserConst.D_ACC_SECONDARY_SRA:
                    srx_acc_concat.append(acc[ParserConst.D_ACC_SECONDARY_VALUE])
                    found = 1
                # alarm me if NCBI changes in the future
                else:
                    raise Exception("Developping bugs. ExtRalations have more than SRA accession")
            if not found:
                srx_acc_concat.append(ParserConst.EMPTY)

        # map GSM and SRX together
        start = 0
        for count in gsm_count:
            sample_batch = []
            gsm_batch = gsm_acc_concat[start:start + count]
            srx_batch = srx_acc_concat[start:start + count]
            for gsm, srx in zip(gsm_batch, srx_batch):
                if srx != ParserConst.EMPTY:
                    sample_batch.append((gsm, srx))
                else:
                    sample_batch.append(gsm)
            sample_list.append(sample_batch)
            start += count
        return sample_list

    def parse_species(self, data):
        species_list = []
        for row in data:
            species_str = row[ParserConst.D_SPECIES]
            species_list.append(species_str.split(ParserConst.SEP_SPECIES))
        return species_list

    def parse_exp_type(self, data):
        exp_type_list = []
        for row in data:
            exp_type_str = row[ParserConst.D_EXP_TYPE]
            exp_type_list.append(exp_type_str.split(ParserConst.SEP_EXP_TYPE))
        return exp_type_list

    def parse_keywords(self, data):
        return ''

    def parse_patients_num(self, data):
        return ''

    def parse_patients_ctry(self, data):
        return ''

    def parse(self, data):
        accession_list                  = self.parse_accession(data)
        title_list                      = self.parse_title(data)
        description_list                = self.parse_description(data)
        pmid_list, pub_title_list,\
            date_list, author_list      = self.parse_publication(data)
        tech_id_list, tech_name_list    = self.parse_tech(data)
        sample_list                     = self.parse_sample(data)
        species_list                    = self.parse_species(data)
        exp_type_list                   = self.parse_exp_type(data)

        return accession_list, title_list, description_list, pmid_list, pub_title_list,\
            date_list, author_list, tech_id_list, tech_name_list, sample_list, species_list,\
            exp_type_list