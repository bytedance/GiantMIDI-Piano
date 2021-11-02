#!/bin/bash
WORKSPACE="./workspace"

###### Plot figures for paper ######
# Calculate the number of piano pieces, nationalites, etc.
python3 calculate_statistics.py meta_info --workspace=$WORKSPACE --surname_in_youtube_title

# Plot number of works of composers.
python3 calculate_statistics.py plot_composer_works_num --workspace=$WORKSPACE --surname_in_youtube_title

# Plot total piece durations of composers.
python3 calculate_statistics.py plot_composer_durations --workspace=$WORKSPACE --surname_in_youtube_title

# Plot nationalities.
python3 calculate_statistics.py plot_nationalities

# Collect notes. This may takes a few hours.
python3 calculate_statistics.py calculate_music_events_from_midi --workspace=$WORKSPACE

# Plot histogram of all notes.
python3 calculate_statistics.py plot_note_histogram --workspace=$WORKSPACE --surname_in_youtube_title

# Plot note historgram of selected composers.
python3 calculate_statistics.py plot_selected_composers_note_histogram --workspace=$WORKSPACE --surname_in_youtube_title

# Plot mean and standard values of notes of different composers.
python3 calculate_statistics.py plot_mean_std_notes --workspace=$WORKSPACE --surname_in_youtube_title

# plot mean and standor values of number of notes per second.
python3 calculate_statistics.py plot_notes_per_second_mean_std --workspace=$WORKSPACE --surname_in_youtube_title

# Plot chroma of selected composers.
python3 calculate_statistics.py plot_selected_composers_chroma --workspace=$WORKSPACE --surname_in_youtube_title

# Plot note intervals of selected composers.
python3 calculate_statistics.py plot_selected_composers_intervals --workspace=$WORKSPACE --surname_in_youtube_title

# Plot chords statistics of selected composers.
python3 calculate_statistics.py plot_selected_composers_chords --workspace=$WORKSPACE --n_chords=3 --surname_in_youtube_title
python3 calculate_statistics.py plot_selected_composers_chords --workspace=$WORKSPACE --n_chords=4 --surname_in_youtube_title
