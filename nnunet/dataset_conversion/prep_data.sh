export nnUNet_raw_data_base="/arc/project/st-rohling-1/data/kidney/raw_data"
export nnUNet_preprocessed="/scratch/st-rohling-1/nnUNet/preprocessed_data"
export RESULTS_FOLDER="/scratch/st-rohling-1/nnUNet/results"
python select_files.py
python prep_capsule_data.py
python prep_region_data.py