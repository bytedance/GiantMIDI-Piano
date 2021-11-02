#!/bin/bash
WORKSPACE="./workspace"

###### Create train & validation & test split ######
python3 create_split.py create_piano_split --workspace=$WORKSPACE

###### Copy curated MIDI files to a new subset ######
python3 create_split.py create_surname_checked_subset --workspace=$WORKSPACE
