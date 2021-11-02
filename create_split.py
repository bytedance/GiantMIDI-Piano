import argparse
import os

from dataset import read_csv_to_meta_dict, write_meta_dict_to_csv


def create_piano_split(args):
    """Add 'giant_midi_piano', 'split', and 'surname_in_youtube_title' flags 
    to csv file and write out the csv file. The ratio of validation, test, train
    subsets are 1:1:8.

    Args:
        workspace: str

    Returns:
        NoReturn
    """
    # arguments & parameters
    workspace = args.workspace
    
    # Paths
    piano_prediction_path = os.path.join(workspace, 
        'full_music_pieces_youtube_similarity_pianosoloprob.csv')

    split_path = os.path.join(workspace, 
        'full_music_pieces_youtube_similarity_pianosoloprob_split.csv')

    # Meta info to be downloaded
    meta_dict = read_csv_to_meta_dict(piano_prediction_path)
    giant_midi_pianos = []
    splits = []
    surname_in_youtube_titles = []

    i = 0

    # Add 'giant_midi_piano', 'split', and 'surname_in_youtube_title' flags to .csv file.
    for n in range(len(meta_dict['surname'])):

        if meta_dict['piano_solo_prob'][n] == "":
            giant_midi_pianos.append("")
            splits.append("")
            surname_in_youtube_titles.append("")

        else:
            if float(meta_dict['piano_solo_prob'][n]) >= 0.5:

                giant_midi_pianos.append(1)

                if i == 0:
                    splits.append('validation')
                elif i == 1:
                    splits.append('test')
                else:
                    splits.append('train')
                i += 1

            else:
                giant_midi_pianos.append(0)
                splits.append("")

            if meta_dict['surname'][n] in meta_dict['youtube_title'][n]:
                surname_in_youtube_titles.append(1)
            else:
                surname_in_youtube_titles.append(0)

        # Reset i if moved to next composer
        if n > 0:
            previous_name = '{}, {}'.format(meta_dict['surname'][n - 1], meta_dict['surname'][n - 1])
            current_name = '{}, {}'.format(meta_dict['surname'][n], meta_dict['surname'][n])
            if previous_name != current_name:
                i = 0

        if i == 10:
            i = 0

    meta_dict['giant_midi_piano'] = giant_midi_pianos
    meta_dict['split'] = splits
    meta_dict['surname_in_youtube_title'] = surname_in_youtube_titles

    write_meta_dict_to_csv(meta_dict, split_path)
    print('Write csv to {}'.format(split_path))


def create_surname_checked_subset(args):
    """Select MIDI files whose YouTube titles must contain composer surnames. 
    This procedure will select and filter 7,236 MIDI files from 10,854 MIDI files.

    Args:
        workspace: str

    Returns:
        NoReturn
    """
    # arguments & parameters
    workspace = args.workspace

    # paths
    midis_dir = os.path.join(workspace, "midis")
    surname_checked_midis_dir = os.path.join(workspace, "surname_checked_midis")
    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob_split.csv')

    os.makedirs(surname_checked_midis_dir, exist_ok=True)

    # Read csv file.
    meta_dict = read_csv_to_meta_dict(csv_path)
    audios_num = len(meta_dict['audio_name'])

    count = 0

    for n in range(audios_num):
        if meta_dict['giant_midi_piano'][n] != "" and int(meta_dict['giant_midi_piano'][n]) == 1:
            if int(meta_dict['surname_in_youtube_title'][n]) == 1:
                midi_name = "{}.mid".format(meta_dict['audio_name'][n])
                midi_path = os.path.join(midis_dir, midi_name)
                surname_checked_midi_path = os.path.join(surname_checked_midis_dir, midi_name)
                exec_str = 'cp "{}" "{}"'.format(midi_path, surname_checked_midi_path)
                print(count, exec_str)
                os.system(exec_str)
                count += 1

    print("Copy {} surname checked midi files to {}".format(count, surname_checked_midis_dir))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_piano_split = subparsers.add_parser('create_piano_split')
    parser_create_piano_split.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    parser_create_surname_checked_subset = subparsers.add_parser('create_surname_checked_subset')
    parser_create_surname_checked_subset.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')    

    # Parse arguments
    args = parser.parse_args()

    if args.mode == 'create_piano_split':
        create_piano_split(args)

    elif args.mode == "create_surname_checked_subset":
        create_surname_checked_subset(args)

    else:
        raise Exception('Error argument!')