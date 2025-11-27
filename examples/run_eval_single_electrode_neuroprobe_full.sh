#!/bin/bash
#SBATCH --job-name=e_p_lite
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2  
#SBATCH --mem=16G
#####SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#####SBATCH --constraint=24GB
#SBATCH --exclude=dgx001,dgx002
#SBATCH --array=1-780
#SBATCH --output data/logs/%A_%a.out # STDOUT
#SBATCH --error data/logs/%A_%a.err # STDERR
#SBATCH -p use-everything

nvidia-smi

export PYTHONUNBUFFERED=1
#export ROOT_DIR_BRAINTREEBANK=/om2/user/zaho/braintreebank/braintreebank/
export ROOT_DIR_BRAINTREEBANK=/storage/czw/braintreebank_data_popt_preprocessed/
#export ROOT_DIR_BRAINTREEBANK=/storage/czw/braintreebank_data/
source .venv/bin/activate

# Use the BTBENCH_LITE_SUBJECT_TRIALS from btbench_config.py
declare -a subjects=(1 1 2 2 3 3 4 4 7 7 10 10)
declare -a trials=(1 2 0 4 0 1 0 1 0 1 0 1)

declare -a eval_names=(
    "onset"
    "speech"
    "frame_brightness"
    "global_flow"
    "local_flow"
    "face_num"
    "volume"
    "pitch"
    "delta_volume"
    "gpt2_surprisal"
    "word_length"
    "word_gap"
    "word_index"
    "word_head_pos"
    "word_part_speech"
)
# to make it sequential, just aggregate the eval_names separating with a comma
# eval_names=(
#     $(IFS=,; echo "${eval_names[*]}")
# )

declare -a preprocess=(
    #'stft_absangle', # magnitude and phase after FFT
    #'stft_realimag' # real and imaginary parts after FFT
    #'stft_abs' # just magnitude after FFT ("spectrogram")
    'none' # no preprocessing, just raw voltage

    #'remove_line_noise' # remove line noise from the raw voltage
    #'downsample_200' # downsample to 200 Hz
    #'downsample_200-remove_line_noise' # downsample to 200 Hz and remove line noise
)

declare -a splits_type=(
    "WithinSession"
    # "CrossSession"
)

declare -a classifier_type=(
    "linear"
    #"cnn"
    #"transformer"
)

# Calculate indices for this task
#EVAL_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % ${#eval_names[@]} ))
#PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} % ${#subjects[@]} ))
#PREPROCESS_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} % ${#preprocess[@]} ))
#SPLITS_TYPE_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#preprocess[@]} % ${#splits_type[@]} ))
#CLASSIFIER_TYPE_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#preprocess[@]} / ${#splits_type[@]} % ${#classifier_type[@]} ))
PREPROCESS_IDX=0 #TODO
SPLITS_TYPE_IDX=0 #TODO
CLASSIFIER_TYPE_IDX=0 #TODO

# Get subject, trial and eval name for this task
for EVAL_IDX in {0..14};
do
for PAIR_IDX in {0..11};
do
EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}
PREPROCESS=${preprocess[$PREPROCESS_IDX]}
SPLITS_TYPE=${splits_type[$SPLITS_TYPE_IDX]}
CLASSIFIER_TYPE=${classifier_type[$CLASSIFIER_TYPE_IDX]}
#save_dir="data/brainbert_results/single_electrode_eval_results_${SPLITS_TYPE}"
#save_dir="data/brainbert_results/debug_brainbert${SPLITS_TYPE}"#TODO
save_dir="data/brainbert_results/brainbert_5s_${SPLITS_TYPE}"

echo "Running eval for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL, preprocess $PREPROCESS, classifier $CLASSIFIER_TYPE"
echo "Save dir: $save_dir"
echo "Split type: $SPLITS_TYPE"

# Add the -u flag to Python to force unbuffered output

python -u examples/eval_single_electrode_brainbert.py \
       --eval_name $EVAL_NAME     \
       --subject_id $SUBJECT     \
       --trial_id $TRIAL     \
       --preprocess.type $PREPROCESS     \
       --verbose     \
       --save_dir $save_dir     \
       --split_type $SPLITS_TYPE     \
       --classifier_type $CLASSIFIER_TYPE \
       --bin_size_seconds=5.0 \
       --bins_start_before_word_onset_seconds=2.5 \
       --bins_end_after_word_onset_seconds=2.5 \
       --lite

#python -u examples/eval_single_electrode.py \
#    --eval_name $EVAL_NAME \
#    --subject_id $SUBJECT \
#    --trial_id $TRIAL \
#    --preprocess.type $PREPROCESS \
#    --verbose \
#    --save_dir $save_dir \
#    --split_type $SPLITS_TYPE \
#    --classifier_type $CLASSIFIER_TYPE
done
done
