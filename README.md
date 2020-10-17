# GiantMIDI-Piano

GiantMIDI-Piano [1] is a classical piano MIDI dataset include 10,854 MIDI files of 2,786 composers. GiantMIDI-Piano are transcribed from live recordings with a high-resolution piano transcription system [2].

Here is the demo of GiantMIDI-Piano: https://www.youtube.com/watch?v=5U-WL0QvKCg

Transcribed MIDI files of GiantMIDI-Piano can be viewed at [midis_preview](midis_preview) directory.

If users would like to access all 10,854 MIDI files (185 Mb), please contact kongqiuqiang@bytedance.com.

Optionally, users could download and transcribe GiantMIDI-Piano by themselves as follows: 1) Download audio recordings using our provided .csv file in this repo; 2) Transcribe audio recordings into MIDI files using our released piano transcription system. 

## Install requirements
Install PyTorch (>=1.4) following https://pytorch.org/.

```
pip install -r requirements.txt
```

## Download audio recordings
Download audio recordings from YouTube using the following scripts. Approximately 10,854 audio recordings can be downloaded. There can be audios no longer downloadable.

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
mp3s_piano_solo (10,854 files)
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
midis (10,854 files)
├── Aaron, Michael, Piano Course, V8WvKK-1b2c.mid
├── Abel, Frederic, Lola Polka, SLNJF0uiqRw.mid
└── ...
</pre>

The transcription of all audio recordings may take around 10 days on a single GPU card.

## FAQ
If users met "Too many requests! Sleep for 3600 s" when downloading, it means that YouTube has limited the number of videos for downloading. Users could either 1) Wait until YouTube unblock your IP (1 days or a few weeks), or 2) try to use another machine with a different IP for downloading.

If users still have difficulty when downloading or transcribing, please contact Qiuqiang Kong for support without hesitation.

## Contact
Qiuqiang Kong, kongqiuqiang@bytedance.com

## Cite
[1] Qiuqiang Kong, Bochen Li, Jitong Chen, and Yuxuan Wang. "GiantMIDI-Piano: A large-scale MIDI dataset for classical piano music." arXiv preprint arXiv:2010.07061 (2020). [[pdf]](https://arxiv.org/pdf/2010.07061.pdf)

[2] Qiuqiang Kong, Bochen Li, Xuchen Song, Yuan Wan, and Yuxuan Wang. "High-resolution Piano Transcription with Pedals by Regressing Onsets and Offsets Times." arXiv preprint arXiv:2010.01815 (2020). [[pdf]](https://arxiv.org/pdf/2010.01815)

## License
Apache 2.0