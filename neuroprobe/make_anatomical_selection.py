import pandas as pd
import os
from glob import glob as glob
import random
import json

data_root = "/storage/czw/braintreebank_data"

#https://bookdown.org/u0243256/tbicc/freesurfer.html
dk2text_d = {'supramarginal': 'supramarginal',
             'postcentral': 'postcentral',
             'medialorbitofrontal': 'medial orbitofrontal',
             'caudalmiddlefrontal': 'caudal middle frontal',
             'posteriorcingulate': 'posterior cingulate',
             'middletemporal': 'middle temporal',
             'superiortemporal': 'superior temporal',
             'bankssts': 'bankssts',
             'superiorparietal': 'superior parietal',
             'precuneus': 'precuneus',
             'entorhinal': 'entorhinal',
             'parstriangularis': 'pars triangularis',
             'precentral': 'precentral',
             'parsorbitalis': 'pars orbitalis',
             'insula': 'insula',
             'parahippocampal': 'parahippocampal',
             'inferiortemporal': 'inferior temporal',
             'parsopercularis': 'pars opercularis',
             'fusiform': 'fusiform',
             'transversetemporal': 'transverse temporal',
             'superiorfrontal': 'superior frontal',
             'paracentral': 'paracentral',
             'lateralorbitofrontal': 'lateral orbitofrontal',
             'caudalanteriorcingulate': 'caudal anterior cingulate',
             'inferiorparietal': 'inferior parietal',
             'rostralanteriorcingulate': 'rostral anterior cingulate',
             'isthmuscingulate': 'isthmus cingulate',
             'temporalpole': 'temporal pole',
             'rostralmiddlefrontal': 'rostral middle frontal',
             'amygdala': 'Amygdala',
             'hippocampus': 'Hippocampus',
             'inf-lat-vent': 'Inf. Lat. Vent.',
             'putamen': 'putamen',
             'unknown': 'unknown'}

dk_names = ["superior frontal", "rostral middle frontal", "caudal middle frontal", "pars opercularis",
           "pars triangularis", "pars orbitalis", "lateral orbitofrontal", "medial orbitofrontal",
           "precentral", "paracentral", "frontal pole", "superior parietal", "inferior parietal",
           "supramarginal", "postcentral", "precuneus", "superior temporal", "middle temporal",
           "inferior temporal", "bankssts", "fusiform", "transverse temporal",
           "entorhinal","temporal pole", "parahippocampal", "lateral occipital", "lingual", "cuneus",
           "pericalcarine","rostral anterior cingulate", "caudal anterior cingulate",
           "posterior cingulate","isthmus cingulate", "insula"]

region_ids = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6]
name2region_id = {n:x for x,n in zip(region_ids, dk_names)}
region_id2region_name = {x+1:n for x,n in enumerate(["Frontal", "Parietal", "Temporal", "Occipital", "Cingulate", "Insula"])}

def dk2text(label):
    l = label.replace('ctx-','')
    l = l.replace('rh-','')
    l = l.replace('lh-','')
    l = l.replace('Left-','')
    l = l.replace('Right-','')
    l = l.lower()
    return dk2text_d[l]

def dk2region(label):
    dk_name = dk2text(label)
    if dk_name in name2region_id:
        return region_id2region_name[name2region_id[dk_name]]
    return dk_name

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
for fpath in glob(f'{localization_root}/sub_*/*.csv'):
    subject = fpath.split("/")[-2]
    local_df = pd.read_csv(fpath)
    local_df["common_name"] = [dk2region(x) for x in local_df.DesikanKilliany]
    all_localization_dfs[subject] = local_df

for subject in neuroprobe_lite_electrodes:
    subject_id = "sub_" + subject[len("btbank"):]
    laplacian_electrodes = get_all_laplacian_electrodes(get_all_electrodes(subject_id, data_root))

    #only select temporal and frontal electrodes
    region_elecs = all_localization_dfs[subject_id][all_localization_dfs[subject_id].common_name.str.contains("Temporal") | all_localization_dfs[subject_id].common_name.str.contains("Frontal")] 
    region_elecs = set(region_elecs.Electrode).intersection(laplacian_electrodes)

    select_k = len(neuroprobe_lite_electrodes[subject])
    print(len(region_elecs), select_k)
    #prioritize the region elecs
    if select_k <= len(laplacian_electrodes):
        random_region_electrodes = random.sample(list(region_elecs), min(select_k,len(region_elecs)))
        random_filler_electrodes = random.sample(list(set(laplacian_electrodes).difference(region_elecs)), max(select_k-len(region_elecs), 0))
        random_electrodes = random_region_electrodes + random_filler_electrodes
        assert len(set(random_electrodes)) == select_k
    else:
        print(f"{subject} not enough electrodes")
        random_electrodes = neuroprobe_lite_electrodes[subject]
    selection[subject] = random_electrodes

with open(f"/storage/czw/neuroprobe/neuroprobe/anatomical_lite_selection_seed_{seed}.json", "w") as f:
    clean_electrodes = json.dump(selection, f)
