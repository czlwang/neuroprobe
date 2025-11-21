import os
from glob import glob as glob
import random
import json

data_root = "/storage/czw/braintreebank_data"

def stem_electrode_name(name):
    #names look like 'O1aIb4', 'O1aIb5', 'O1aIb6', 'O1aIb7'
    #names look like 'T1b2
    reverse_name = reversed(name)
    found_stem_end = False
    stem, num = [], []
    for c in reversed(name):
        if c.isalpha():
            found_stem_end = True
        if found_stem_end:
            stem.append(c)
        else:
            num.append(c)
    return ''.join(reversed(stem)), int(''.join(reversed(num)))

def get_all_electrodes(subject, data_root=None):
    '''
        returns list of electrodes in this subject and trial
        NOTE: the order of these labels is important. Their position corresponds with a row in data.h5
    '''
    electrode_labels_file = glob(os.path.join(data_root, "electrode_labels", subject, "electrode_labels.json"))
    assert len(electrode_labels_file)==1
    electrode_labels_file = electrode_labels_file[0]
    with open(electrode_labels_file, "r") as f:
        electrode_labels = json.load(f)
    strip_string = lambda x: x.replace("*","").replace("#","").replace("_","")
    electrode_labels = [strip_string(e) for e in electrode_labels]
    return electrode_labels

def get_all_laplacian_electrodes(elec_list):
    stems = [stem_electrode_name(e) for e in elec_list]
    def has_nbrs(stem, stems):
        (x,y) = stem
        return ((x,y+1) in stems) and ((x,y-1) in stems)
    laplacian_stems = [x for x in stems if has_nbrs(x, stems)]
    electrodes = [f'{x}{y}' for (x,y) in laplacian_stems]
    return electrodes

with open("/storage/czw/neuroprobe/neuroprobe/lite_electrodes.json", "r") as f:
    neuroprobe_lite_electrodes = json.load(f)

seed = 0
random.seed(seed)
selection = {}

localization_root = os.path.join(data_root, "localization")
all_localization_dfs = {}
for fpath in glob.glob(f'{localization_root}/*'):
    subject = os.path.split(fpath)[1].split(".")[0]
    all_localization_dfs[subject] = pd.read_csv(fpath)

import pdb; pdb.set_trace()

for subject in neuroprobe_lite_electrodes:
    print(subject, len(neuroprobe_lite_electrodes[subject]))
    subject_id = "sub_" + subject[len("btbank"):]
    laplacian_electrodes = get_all_laplacian_electrodes(get_all_electrodes(subject_id, data_root))

    #btbank_key = "btbank" + subject.split("_")[1]
    #electrodes = clean_electrodes[subject]
    #print(len(electrodes))
    select_k = len(neuroprobe_lite_electrodes[subject])
    print(len(laplacian_electrodes))
    if select_k <= len(laplacian_electrodes):
        random_electrodes = random.sample(laplacian_electrodes, select_k)
    else:
        print(f"{subject} not enough electrodes")
        random_electrodes = neuroprobe_lite_electrodes[subject]
    selection[subject] = random_electrodes

with open(f"/storage/czw/neuroprobe/neuroprobe/random_lite_selection_seed_{seed}.json", "w") as f:
    clean_electrodes = json.dump(selection, f)




