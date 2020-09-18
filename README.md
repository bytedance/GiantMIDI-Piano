# GiantMIDI-Piano

GiantMIDI-Piano [1] is a classical piano MIDI dataset include 10,848 MIDI files of 2,784 composers. GiantMIDI-Piano are transcribed from live recordings with a high-resolution piano transcription system [2]. To acquire GiantMIDI-Piano, users need to: 1) Download audio recordings by themselves using our provided .csv file; 2) Transcribe audio recordings to MIDI files using our released piano transcription system. 

Here is the demo of GiantMIDI-Piano: https://www.youtube.com/watch?v=5U-WL0QvKCg

## Install requirements
```
pip install -r requirements.txt
```

## Download audio recordings
Download audio recordings from YouTube using the following scripts. Approximately 10,848 audio recordings can be downloaded. There can be audios no longer downloadable.

```
WORKSPACE="./workspace"
mkdir -p $WORKSPACE
cp "resources/full_music_pieces_youtube_similarity_pianosoloprob.csv" $WORKSPACE/"full_music_pieces_youtube_similarity_pianosoloprob.csv"

# Download all mp3s. Users could split the downloading into parts to speed up the downloading. E.g.,
python3 dataset.py download_youtube_piano_solo --workspace=$WORKSPACE --begin_index=0 --end_index=30000
python3 dataset.py download_youtube_piano_solo --workspace=$WORKSPACE --begin_index=30000 --end_index=60000
python3 dataset.py download_youtube_piano_solo --workspace=$WORKSPACE --begin_index=60000 --end_index=90000
python3 dataset.py download_youtube_piano_solo --workspace=$WORKSPACE --begin_index=90000 --end_index=120000
python3 dataset.py download_youtube_piano_solo --workspace=$WORKSPACE --begin_index=12000 --end_index=150000
```

The downloaded mp3 files look like:

<pre>
mp3s_piano_solo (10,848 files)
├── Aaron, Michael, Piano Course, V8WvKK-1b2c.mp3
├── Aarons, Alfred E., Brother Bill, Giet2Krl6Ww.mp3
└── ...
</pre>

## Transcribe
Trainscribe audio recordings to MIDI files.

```
# Transcribe all mp3s to midi files. Users could split the transcription into parts to speed up the transcription. E.g.,
python3 audios_to_midis.py transcribe_piano --workspace=$WORKSPACE --mp3s_dir=$WORKSPACE"/mp3s_piano_solo" --midis_dir=$WORKSPACE"/midis" --begin_ind=0 --end_index=30000
python3 audios_to_midis.py transcribe_piano --workspace=$WORKSPACE --mp3s_dir=$WORKSPACE"/mp3s_piano_solo" --midis_dir=$WORKSPACE"/midis" --begin_ind=30000 --end_index=60000
python3 audios_to_midis.py transcribe_piano --workspace=$WORKSPACE --mp3s_dir=$WORKSPACE"/mp3s_piano_solo" --midis_dir=$WORKSPACE"/midis" --begin_ind=60000 --end_index=90000
python3 audios_to_midis.py transcribe_piano --workspace=$WORKSPACE --mp3s_dir=$WORKSPACE"/mp3s_piano_solo" --midis_dir=$WORKSPACE"/midis" --begin_ind=90000 --end_index=120000
python3 audios_to_midis.py transcribe_piano --workspace=$WORKSPACE --mp3s_dir=$WORKSPACE"/mp3s_piano_solo" --midis_dir=$WORKSPACE"/midis" --begin_ind=120000 --end_index=150000
```

The transcribed MIDI files look like:

<pre>
midis (10,848 files)
├── Aaron, Michael, Piano Course, V8WvKK-1b2c.mid
├── Abel, Frederic, Lola Polka, SLNJF0uiqRw.mid
└── ...
</pre>

The transcription of all audio recordings may take around one week using a single GPU card.

## Previews of transcribed MIDI files
We provide a few MIDI files of GiantMIDI-Piano for preview at the [midis_preview](midis_preview) directory.

## FAQ
If users met "Too many requests! Sleep for 3600 s" when downloading, it means that YouTube has limited the number of videos for downloading. Users could either 1) Wait until YouTube unblock your IP (1 days or a few weeks), or 2) try to use another machine with a different IP for downloading.

If users still have difficulty when downloading or transcribing, please contact **Qiuqiang Kong** for support without hesitation.

## Contact
Qiuqiang Kong, kongqiuqiang@bytedance.com

## Cite
[1] Qiuqiang Kong, Bochen Li, Jitong Chen, Yuxuan Wang, GiantMIDI-Piano: A MIDI dataset of classical piano music, 2020

[2] Qiuqiang Kong, Bochen Li, Xuchen Song, Yuxuan Wang, High resolution piano transcription by regressing onset and offset times, 2020

## License
Apache 2.0