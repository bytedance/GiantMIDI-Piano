#!/bin/bash
WORKSPACE="./workspace"

###### This script is used to download IMSLP webpages, create .csv files that 
# contain information of YouTube links to download, and download audios from YouTube. 

# Download IMSLP html pages.
python3 dataset.py download_imslp_htmls --workspace=$WORKSPACE

# Download wikipedia html pages of composers.
python3 dataset.py download_wikipedia_htmls --workspace=$WORKSPACE

# Create GiantMIDI-Piano meta csv.
python3 dataset.py create_meta_csv --workspace=$WORKSPACE

# Search and append YouTube titles and IDs to meta csv.
python3 dataset.py search_youtube --workspace=$WORKSPACE

# Calculate and append the similarity between YouTube titles and IMSLP music names to meta csv.
python3 dataset.py calculate_similarity --workspace=$WORKSPACE

# Download all mp3s. Users could split the downloading into parts to speed up the downloading.
python3 dataset.py download_youtube --workspace=$WORKSPACE --begin_index=0 --end_index=30000
python3 dataset.py download_youtube --workspace=$WORKSPACE --begin_index=30000 --end_index=60000
python3 dataset.py download_youtube --workspace=$WORKSPACE --begin_index=60000 --end_index=90000
python3 dataset.py download_youtube --workspace=$WORKSPACE --begin_index=90000 --end_index=120000
python3 dataset.py download_youtube --workspace=$WORKSPACE --begin_index=12000 --end_index=150000