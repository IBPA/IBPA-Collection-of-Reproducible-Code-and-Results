import pickle
import json
import sys
from transcriptomic_pipeline import GeneralParameters
from transcriptomic_pipeline import GeneralConstant

from sequencing_pipeline import *

if (sys.version_info < (3, 0)):
    sys.path.insert(0, "sequencing")
    import s_sample_mapping
else:
    import sequencing.s_sample_mapping

if __name__ == "__main__":
    sample_mapping_worker = pickle.load(open(sys.argv[1], 'rb'))

    sample_mapping_worker.do_run()

    sample_mapping_worker.results.done = True
    json.dump(sample_mapping_worker.results.save_result_to_dict(), open(sys.argv[2], 'w'))
