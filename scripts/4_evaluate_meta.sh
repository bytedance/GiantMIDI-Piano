#!/bin/bash
WORKSPACE="./workspace"

# Select and write 200 files from 60000+ downloaded audios to a csv to evaluate piano solo detection precision and recall.
python3 evaluate_meta.py create_subset200_eval_csv --workspace=$WORKSPACE

# Evaluate piano solo detection precision and recall.
python3 evaluate_meta.py plot_piano_solo_p_r_f1 \
    --subset200_eval_with_labels_path="./manual_labels/subset200_eval_with_labels.csv"

# Select and write 200 files from GiantMIDI-Piano to a csv to evaluate piano pieces accuracy.
python3 evaluate_meta.py create_subset200_piano_solo_eval_csv --workspace=$WORKSPACE

# Evaluate piano pieces accruacy.
python3 evaluate_meta.py piano_solo_meta_accuracy \
    --subset200_piano_solo_eval_with_labels_path="./manual_labels/subset200_piano_solo_eval_with_labels.csv" \
    --surname_in_youtube_title

# Evaluate piano pieces performance ratio.
python3 evaluate_meta.py piano_solo_performed_ratio \
    --subset200_piano_solo_eval_with_labels_path="./manual_labels/subset200_piano_solo_eval_with_labels.csv" \
    --surname_in_youtube_title

# Evaluate individual composers piano solo meta accuracy.
python3 evaluate_meta.py individual_composer_piano_solo_meta_accuracy \
    --workspace=$WORKSPACE \
    --surname="Bach" \
    --firstname="Johann Sebastian" \
    --surname_in_youtube_title
