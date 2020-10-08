
import json
import pickle
import sys
import os

sys.path.insert(0,'sequencing')

if __name__ == '__main__':
    target_dir = sys.argv[1]
    files = os.listdir(target_dir)

    for cur_file in files:
        if cur_file.startswith('results_s_sample_mapping') and cur_file.endswith('.dat'):
            print(cur_file)
            tmp = pickle.load(open(target_dir.replace('/','') + '/' + cur_file,'rb'))
            
            new_json_obj = {}
            for cur_key in tmp.__dict__.keys():
                if type(getattr(tmp, cur_key)) is not dict:
                    new_json_obj[cur_key] = getattr(tmp, cur_key)
                else:
                    if cur_key == 'merged_count_reads_result':
                        tmp_result = getattr(tmp, cur_key)
                        new_result = {}
                        for e in tmp_result:
                            new_result[e] = tmp_result[e].to_csv()
                    else:
                        tmp_result = getattr(tmp, cur_key)
                        new_result = {}
                        for e in tmp_result:
                            new_result[e] = tmp_result[e]
                    new_json_obj[cur_key] = new_result

            json.dump(new_json_obj, open(target_dir.replace('/','') + '/' + cur_file.replace('dat','json'),'w'))



