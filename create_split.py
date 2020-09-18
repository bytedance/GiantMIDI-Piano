import argparse
import os

from dataset import read_csv_to_meta_dict, write_meta_dict_to_csv


def create_piano_split(args):
    """Validation, test, train: 1:1:8
    """
    # Arguments & parameters
    workspace = args.workspace
    
    # Paths
    piano_prediction_path = os.path.join(workspace, 
        'full_music_pieces_youtube_similarity_pianosoloprob.csv')

    split_path = os.path.join(workspace, 
        'full_music_pieces_youtube_similarity_pianosoloprob_split.csv')

    # Meta info to be downloaded
    meta_dict = read_csv_to_meta_dict(piano_prediction_path)
    splits = []

    i = 0
    for n in range(len(meta_dict['surname'])):
        
        if float(meta_dict['piano_solo_prob'][n]) >= 0.5:
            if i == 0:
                splits.append('validation')
            elif i == 1:
                splits.append('test')
            else:
                splits.append('train')
            i += 1
        else:
            splits.append('none')

        # Reset i if moved to next composer
        if n > 0:
            previous_name = '{}, {}'.format(meta_dict['surname'][n - 1], meta_dict['surname'][n - 1])
            current_name = '{}, {}'.format(meta_dict['surname'][n], meta_dict['surname'][n])
            if previous_name != current_name:
                i = 0

        if i == 10:
            i = 0

    meta_dict['split'] = splits
    write_meta_dict_to_csv(meta_dict, split_path)
    print('Write csv to {}'.format(split_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_piano_split = subparsers.add_parser('create_piano_split')
    parser_create_piano_split.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    # Parse arguments
    args = parser.parse_args()

    if args.mode == 'create_piano_split':
        create_piano_split(args)

    else:
        raise Exception('Error argument!')