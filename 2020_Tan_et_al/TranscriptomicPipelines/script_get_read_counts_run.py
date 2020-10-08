import pickle
import json
import sys
from transcriptomic_pipeline import GeneralParameters
from transcriptomic_pipeline import GeneralConstant

from sequencing_pipeline import *

if (sys.version_info < (3, 0)):
    sys.path.insert(0, "sequencing")
    import s_value_extraction
else:
    import sequencing.s_value_extraction

if __name__ == "__main__":
    value_extraction_worker = pickle.load(open(sys.argv[1], 'rb'))

    value_extraction_worker.do_run()
    value_extraction_worker.results.done = True
    print("DONE!")
    print(sys.argv[2])
    json.dump(value_extraction_worker.results.save_result_to_dict(), open(sys.argv[2], 'w'))