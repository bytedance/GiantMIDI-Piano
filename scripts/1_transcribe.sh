#!/bin/bash
WORKSPACE="./workspace"

# Calculate the piano solo probabilities of downloaded mp3s. This may take several days.
python3 audios_to_midis.py calculate_piano_solo_prob \
	--workspace=$WORKSPACE \
	--mp3s_dir=$WORKSPACE"/mp3s"

# Transcribe all mp3s to midi files. Users could split the transcription into parts to speed up the transcription.
python3 audios_to_midis.py transcribe_piano --workspace=$WORKSPACE --mp3s_dir=$WORKSPACE"/mp3s" --midis_dir=$WORKSPACE"/midis" --begin_ind=0 --end_index=30000
python3 audios_to_midis.py transcribe_piano --workspace=$WORKSPACE --mp3s_dir=$WORKSPACE"/mp3s" --midis_dir=$WORKSPACE"/midis" --begin_ind=30000 --end_index=60000
python3 audios_to_midis.py transcribe_piano --workspace=$WORKSPACE --mp3s_dir=$WORKSPACE"/mp3s" --midis_dir=$WORKSPACE"/midis" --begin_ind=60000 --end_index=90000
python3 audios_to_midis.py transcribe_piano --workspace=$WORKSPACE --mp3s_dir=$WORKSPACE"/mp3s" --midis_dir=$WORKSPACE"/midis" --begin_ind=90000 --end_index=120000
python3 audios_to_midis.py transcribe_piano --workspace=$WORKSPACE --mp3s_dir=$WORKSPACE"/mp3s" --midis_dir=$WORKSPACE"/midis" --begin_ind=120000 --end_index=150000