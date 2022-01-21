import os
import sys
import numpy as np
import argparse
import time
import librosa
import torch
import piano_transcription_inference

import piano_detection_model
from utilities import get_filename
from dataset import read_csv_to_meta_dict, write_meta_dict_to_csv
import config


def calculate_piano_solo_prob(args):
    """Calculate the piano solo probability of all downloaded mp3s, and append
    the probability to the meta csv file.
    """
    # Arguments & parameters
    workspace = args.workspace 
    mp3s_dir = args.mp3s_dir
    mini_data = args.mini_data

    sample_rate = piano_detection_model.SR

    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''

    # Paths
    similarity_csv_path = os.path.join(workspace, 
        '{}full_music_pieces_youtube_similarity.csv'.format(prefix))

    piano_prediction_path = os.path.join(workspace, 
        '{}full_music_pieces_youtube_similarity_pianosoloprob.csv'.format(prefix))

    # Meta info
    meta_dict = read_csv_to_meta_dict(similarity_csv_path)

    meta_dict['piano_solo_prob'] = []
    meta_dict['audio_name'] = []
    meta_dict['audio_duration'] = []
    count = 0

    piano_solo_detector = piano_detection_model.PianoSoloDetector()

    for n in range(len(meta_dict['surname'])):
        mp3_path = os.path.join(mp3s_dir, '{}, {}, {}, {}.mp3'.format(
            meta_dict['surname'][n], meta_dict['firstname'][n], 
            meta_dict['music'][n], meta_dict['youtube_id'][n]).replace('/', '_'))

        if os.path.exists(mp3_path):
            (audio, _) = librosa.core.load(mp3_path, sr=sample_rate, mono=True)
            
            try:
                probs = piano_solo_detector.predict(audio)
                prob = np.mean(probs)
            except:
                prob = 0
    
            print(n, mp3_path, prob)
            meta_dict['audio_name'].append(get_filename(mp3_path))
            meta_dict['piano_solo_prob'].append(prob)
            meta_dict['audio_duration'].append(len(audio) / sample_rate)
        else:
            meta_dict['piano_solo_prob'].append('')
            meta_dict['audio_name'].append('')
            meta_dict['audio_duration'].append('')

    write_meta_dict_to_csv(meta_dict, piano_prediction_path)
    print('Write out to {}'.format(piano_prediction_path))
    

def transcribe_piano(args):
    """Transcribe piano solo mp3s to midi files.
    """

    # Arguments & parameters
    workspace = args.workspace
    mp3s_dir = args.mp3s_dir
    midis_dir = args.midis_dir
    begin_index = args.begin_index
    end_index = args.end_index
    mini_data = args.mini_data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''

    # Paths
    csv_path = os.path.join('./resources/full_music_pieces_youtube_similarity_pianosoloprob_split.csv')

    os.makedirs(midis_dir, exist_ok=True)

    # Meta info
    meta_dict = read_csv_to_meta_dict(csv_path)

    # Transcriptor
    transcriptor = piano_transcription_inference.PianoTranscription(device=device)

    count = 0
    transcribe_time = time.time()
    audios_num = len(meta_dict['surname'])

    for n in range(begin_index, min(end_index, audios_num)):
        
        if meta_dict['giant_midi_piano'][n] and int(meta_dict['giant_midi_piano'][n]) == 1:
            count += 1
            
            mp3_path = os.path.join(mp3s_dir, '{}.mp3'.format(meta_dict['audio_name'][n]))
            print(n, mp3_path)
            midi_path = os.path.join(midis_dir, '{}.mid'.format(meta_dict['audio_name'][n]))

            (audio, _) = piano_transcription_inference.load_audio(mp3_path, 
                sr=piano_transcription_inference.sample_rate, mono=True)

            try:
                # Transcribe
                transcribed_dict = transcriptor.transcribe(audio, midi_path)
            except:
                print('Failed for this audio!') 

    print('Time: {:.3f} s'.format(time.time() - transcribe_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Plot statistics
    parser_predict_piano = subparsers.add_parser('calculate_piano_solo_prob')
    parser_predict_piano.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_predict_piano.add_argument('--mp3s_dir', type=str, required=True, help='')
    parser_predict_piano.add_argument('--mini_data', action='store_true', default=False)

    parser_transcribe_piano = subparsers.add_parser('transcribe_piano')
    parser_transcribe_piano.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_transcribe_piano.add_argument('--mp3s_dir', type=str, required=True, help='')
    parser_transcribe_piano.add_argument('--midis_dir', type=str, required=True, help='')
    parser_transcribe_piano.add_argument('--begin_index', type=int, required=True, help='')
    parser_transcribe_piano.add_argument('--end_index', type=int, required=True, help='')
    parser_transcribe_piano.add_argument('--mini_data', action='store_true', default=False)

    # Parse arguments
    args = parser.parse_args()

    if args.mode == 'calculate_piano_solo_prob':
        calculate_piano_solo_prob(args)

    elif args.mode == 'transcribe_piano':
        transcribe_piano(args)

    else:
        raise Exception('Error argument!')