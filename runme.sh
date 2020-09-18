#!/bin/bash
WORKSPACE="./workspace"

###### Please follow README.md to acquire GiantMIDI-Piano. This runme.sh file
# is only used to prepare the scripts, csv files, and plot figures for paper.

# Download IMSLP html pages
python3 dataset.py download_imslp_htmls --workspace=$WORKSPACE

# Download wikipedia html pages of composers
python3 dataset.py download_wikipedia_htmls --workspace=$WORKSPACE

# Create GiantMIDI-Piano meta csv
python3 dataset.py create_meta_csv --workspace=$WORKSPACE

# Search and append YouTube titles and IDs to meta csv
python3 dataset.py search_youtube --workspace=$WORKSPACE

# Calculate and append the similarity between YouTube titles and IMSLP music names to meta csv
python3 dataset.py calculate_similarity --workspace=$WORKSPACE

# Download all mp3s. Users could split the downloading into parts to speed up the downloading. E.g.,
python3 dataset.py download_youtube --workspace=$WORKSPACE --begin_index=0 --end_index=30000
python3 dataset.py download_youtube --workspace=$WORKSPACE --begin_index=30000 --end_index=60000
python3 dataset.py download_youtube --workspace=$WORKSPACE --begin_index=60000 --end_index=90000
python3 dataset.py download_youtube --workspace=$WORKSPACE --begin_index=90000 --end_index=120000
python3 dataset.py download_youtube --workspace=$WORKSPACE --begin_index=12000 --end_index=150000

######
# Calculate the piano solo probabilities of downloaded mp3s. This may take several days.
python3 audios_to_midis.py calculate_piano_solo_prob --workspace=$WORKSPACE --mp3s_dir=$WORKSPACE"/mp3s"

# Transcribe all mp3s to midi files. Users could split the transcription into parts to speed up the transcription. E.g.,
python3 audios_to_midis.py transcribe_piano --workspace=$WORKSPACE --mp3s_dir=$WORKSPACE"/mp3s" --midis_dir=$WORKSPACE"/midis" --begin_ind=0 --end_index=30000
python3 audios_to_midis.py transcribe_piano --workspace=$WORKSPACE --mp3s_dir=$WORKSPACE"/mp3s" --midis_dir=$WORKSPACE"/midis" --begin_ind=30000 --end_index=60000
python3 audios_to_midis.py transcribe_piano --workspace=$WORKSPACE --mp3s_dir=$WORKSPACE"/mp3s" --midis_dir=$WORKSPACE"/midis" --begin_ind=60000 --end_index=90000
python3 audios_to_midis.py transcribe_piano --workspace=$WORKSPACE --mp3s_dir=$WORKSPACE"/mp3s" --midis_dir=$WORKSPACE"/midis" --begin_ind=90000 --end_index=120000
python3 audios_to_midis.py transcribe_piano --workspace=$WORKSPACE --mp3s_dir=$WORKSPACE"/mp3s" --midis_dir=$WORKSPACE"/midis" --begin_ind=120000 --end_index=150000

###### Create train & validation & test split ######
python3 create_split.py create_piano_split --workspace=$WORKSPACE

###### Plot figures for paper ######
# Calculate the number of piano pieces, nationalites, etc.
python3 calculate_statistics.py meta_info --workspace=$WORKSPACE

# Plot number of works of composers
python3 calculate_statistics.py plot_composer_works_num --workspace=$WORKSPACE

# This may takes a few hours
python3 calculate_statistics.py calculate_music_events_from_midi --workspace=$WORKSPACE

# Plot total piece durations of composers
python3 calculate_statistics.py plot_composer_durations --workspace=$WORKSPACE

# Plot histogram of all notes
python3 calculate_statistics.py plot_note_histogram --workspace=$WORKSPACE

# Plot mean and standard values of notes of different composers
python3 calculate_statistics.py plot_mean_std_notes --workspace=$WORKSPACE

# plot mean and standor values of number of notes per second
python3 calculate_statistics.py plot_notes_per_second_mean_std --workspace=$WORKSPACE

# Plot note historgram of selected composers
python3 calculate_statistics.py plot_selected_composers_note_histogram --workspace=$WORKSPACE

# Plot chroma of selected composers
python3 calculate_statistics.py plot_selected_composers_chroma --workspace=$WORKSPACE

# Plot note intervals of selected composers
python3 calculate_statistics.py plot_selected_composers_intervals --workspace=$WORKSPACE

# Plot chords statistics of selected composers
python3 calculate_statistics.py plot_selected_composers_chords --workspace=$WORKSPACE --n_chords=3
python3 calculate_statistics.py plot_selected_composers_chords --workspace=$WORKSPACE --n_chords=4

###### Align GiantMIDI-Piano and MAESTRO with sequenced MIDI files ######
# Download and compile alignment tool
wget https://midialignment.github.io/AlignmentTool_v190813.zip
cd AlignmentTool_v190813 
./compile.sh
cd ..

# Align and plot results for paper
python3 evaluate_transcribed_midis.py align
python3 evaluate_transcribed_midis.py plot_box_plot