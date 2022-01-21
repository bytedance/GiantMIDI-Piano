import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from dataset import read_csv_to_meta_dict, write_meta_dict_to_csv


def create_subset200_eval_csv(args):
    r"""Select 200 files from 60,724 downloaded files to evaluate the precision, 
    recall of piano solo detection.

    Args:
        workspace: str

    Returns:
        None
    """
    workspace = args.workspace
    eval_num = 200

    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob_split.csv')

    output_path = os.path.join('subset_csvs_for_evaluation', 'subset200_eval.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    meta_dict = read_csv_to_meta_dict(csv_path)

    audios_num = len(meta_dict['surname'])
    indexes = []

    for n in range(audios_num):
        if float(meta_dict['similarity'][n]) > 0.6:
            indexes.append(n)

    skip_num = len(indexes) // eval_num
    eval_indexes = indexes[0 : : skip_num][0 : eval_num]

    new_meta_dict = {key: [] for key in meta_dict.keys()}
    new_meta_dict['index_in_csv'] = []

    for index in eval_indexes:
        for key in meta_dict.keys():
            new_meta_dict[key].append(meta_dict[key][index])

        new_meta_dict['index_in_csv'].append(index)

    new_meta_dict['piano_solo'] = [''] * eval_num
    new_meta_dict['electronic_piano'] = [''] * eval_num
    new_meta_dict['sequenced'] = [''] * eval_num

    write_meta_dict_to_csv(new_meta_dict, output_path)
    print('Write out to {}'.format(output_path))


def plot_piano_solo_p_r_f1(args):
    r"""Plot piano solo detection precision, recall, and F1 score.

    Args:
        subset200_eval_with_labels_path: str
        surname_in_youtube_title: bool

    Returns:
        None
    """

    # arguments & paramteres
    subset200_eval_with_labels_path = args.subset200_eval_with_labels_path
    surname_in_youtube_title = args.surname_in_youtube_title

    # paths
    out_fig_path = os.path.join('results', 'piano_solo_p_r_f1.pdf')

    meta_dict = read_csv_to_meta_dict(subset200_eval_with_labels_path)

    audios_num = len(meta_dict['surname'])

    precs = []
    recalls = []
    thresholds = []
    f1s = []

    for threshold in np.arange(0, 0.99, 0.1):

        tp, fn, fp, tn = 0, 0, 0, 0

        for n in range(audios_num):
            if meta_dict['audio_name'][n] == '':
                flag = False
            else:
                if surname_in_youtube_title and int(meta_dict['surname_in_youtube_title'][n]) == 0:
                    flag = False
                else:
                    flag = True

            if flag:
                if float(meta_dict['piano_solo_prob'][n]) >= threshold:
                    pred = 1
                else:
                    pred = 0

                target = int(meta_dict['piano_solo'][n])

                if target == 1 and pred == 1:
                    tp += 1
                if target == 1 and pred == 0:
                    fn += 1
                if target == 0 and pred == 1:
                    fp += 1
                if target == 0 and pred == 0:
                    tn += 1
    
        prec = tp / np.clip(tp + fp, 1e-8, np.inf)
        recall = tp / np.clip(tp + fn, 1e-8, np.inf)
        f1 = 2 * prec * recall / (prec + recall)

        precs.append(prec)
        recalls.append(recall)
        thresholds.append(threshold)
        f1s.append(f1)

        if threshold == 0.5:
            print('Threshold: {:.3f}, TP: {}, FN: {}, FP: {}, TN: {}'.format(threshold, tp, fn, fp, tn))

    print('Total num: {}'.format(tp + fn + fp + tn))
    print('Thresholds: {}'.format(thresholds))
    print('Precisions: {}'.format(precs))
    print('Recalls: {}'.format(recalls))
    print('F1s: {}'.format(f1s))

    N = len(thresholds)
    fontsize = 14

    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(5, 3))
    axs.scatter(np.arange(N), precs, color='blue')
    axs.scatter(np.arange(N), recalls, color='green')
    axs.scatter(np.arange(N), f1s, color='red')
    line_p, = axs.plot(precs, label='Precision', linestyle='--', color='blue')
    line_r, = axs.plot(recalls, label='Recall', linestyle='-.', color='green')
    line_f1, = axs.plot(f1s, label='F1', linestyle='-', color='red')
    axs.set_ylim(0., 1.02)
    axs.set_xlabel(r"Thresholds", fontsize=fontsize)
    axs.set_ylabel('Scores', fontsize=fontsize)

    axs.legend(handles=[line_p, line_r, line_f1], loc=4)
    axs.xaxis.set_ticks(np.arange(N))
    axs.xaxis.set_ticklabels(['{:.2f}'.format(e) for e in thresholds], rotation=0)
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    
    os.makedirs(os.path.dirname(out_fig_path), exist_ok=True)
    plt.savefig(out_fig_path)
    print('Write out to {}'.format(out_fig_path))


def create_subset200_piano_solo_eval_csv(args):
    r"""Select 200 pieces from GiantMIDI-Piano to evaluate the music piece accuracy.

    Args:
        workspace: str

    Returns:
        None
    """

    # arguments & parameters
    workspace = args.workspace
    eval_num = 200

    # paths
    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob_split.csv')
    output_path = os.path.join('subset_csvs_for_evaluation', 'subset200_piano_solo_eval.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    meta_dict = read_csv_to_meta_dict(csv_path)

    audios_num = len(meta_dict['surname'])
    indexes = []

    for n in range(audios_num):
        if meta_dict['giant_midi_piano'][n] != '' and int(meta_dict['giant_midi_piano'][n]) == 1:
            indexes.append(n)

    skip_num = len(indexes) // eval_num
    eval_indexes = indexes[0 : : skip_num][0 : eval_num]

    new_meta_dict = {key: [] for key in meta_dict.keys()}
    new_meta_dict['index_in_csv'] = []

    for index in eval_indexes:
        for key in meta_dict.keys():
            new_meta_dict[key].append(meta_dict[key][index])

        new_meta_dict['index_in_csv'].append(index)

    new_meta_dict['meta_correct'] = [''] * eval_num
    new_meta_dict['sequenced'] = [''] * eval_num

    write_meta_dict_to_csv(new_meta_dict, output_path)
    print('Write out to {}'.format(output_path))


def piano_solo_meta_accuracy(args):
    r"""Calcualte piano piece accuracy from 200 files from GiantMIDI-Piano.

    Args:
        subset200_piano_solo_eval_with_labels_path: str
        youtube_title_contain_surname: bool

    Returns:
        None
    """

    # arguments & parameters
    subset200_piano_solo_eval_with_labels_path = args.subset200_piano_solo_eval_with_labels_path
    surname_in_youtube_title = args.surname_in_youtube_title

    meta_dict = read_csv_to_meta_dict(subset200_piano_solo_eval_with_labels_path)

    audios_num = len(meta_dict['surname'])

    tp, fp = 0, 0

    for n in range(audios_num):

        if meta_dict['audio_name'][n] == '':
            flag = False
        else:
            if surname_in_youtube_title and int(meta_dict['surname_in_youtube_title'][n]) == 0:
                flag = False
            else:
                flag = True

        if flag:
            if int(meta_dict['meta_correct'][n]) == 1:
                tp += 1
            else:
                fp += 1

    precision = tp / (tp + fp)

    print('Correct rate: {} / {}, {}'.format(tp, tp + fp, precision))


def piano_solo_performed_ratio(args):
    r"""Calcualte piano piece accuracy from 200 files from GiantMIDI-Piano.

    Args:
        subset200_piano_solo_eval_with_labels_path: str
        youtube_title_contain_surname: bool

    Returns:
        None
    """

    # arguments & parameters
    subset200_piano_solo_eval_with_labels_path = args.subset200_piano_solo_eval_with_labels_path
    surname_in_youtube_title = args.surname_in_youtube_title

    meta_dict = read_csv_to_meta_dict(subset200_piano_solo_eval_with_labels_path)

    audios_num = len(meta_dict['surname'])

    tp, fp = 0, 0

    for n in range(audios_num):

        if meta_dict['audio_name'][n] == '':
            flag = False
        else:
            if surname_in_youtube_title and int(meta_dict['surname_in_youtube_title'][n]) == 0:
                flag = False
            else:
                flag = True

        if flag:
            if int(meta_dict['sequenced'][n]) == 1:
                tp += 1
            else:
                fp += 1

    sequenced_ratio = tp / (tp + fp)

    print('Performance ratio: {:.3f}'.format(1. - sequenced_ratio))


def individual_composer_piano_solo_meta_accuracy(args):
    """Calcualte individual composer piano solo meta accuracy. Composers include:

    Bach, Johann Sebastian
    Mozart, Wolfgang Amadeus
    Beethoven, Ludwig van
    Chopin, Frédéric
    Liszt, Franz
    Debussy, Claude

    Args:
        workspace: str
        surname: str
        firstname: str
        surname_in_youtube_title: bool
    
    Returns:
        None
    """

    workspace = args.workspace
    surname = args.surname
    firstname = args.firstname
    surname_in_youtube_title = args.surname_in_youtube_title

    eval_num = 200

    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob_split.csv')

    meta_dict = read_csv_to_meta_dict(csv_path)

    audios_num = len(meta_dict['surname'])
    indexes = []
    tp, fp = 0, 0

    for n in range(audios_num):
        if meta_dict['giant_midi_piano'][n] != '' and int(meta_dict['giant_midi_piano'][n]) == 1:

            match_composer = (surname == meta_dict['surname'][n] and firstname == meta_dict['firstname'][n])

            if surname_in_youtube_title and int(meta_dict['surname_in_youtube_title'][n]) == 0:
                flag = False
            else:
                flag = True

            if flag:
                if surname in meta_dict['youtube_title'][n] or match_composer:
                    if match_composer:
                        tp += 1
                    else:
                        fp += 1

    accuracy = tp / (tp + fp)
    print('Match: {}, Accuracy: {}'.format(tp, fp))
    print('Accuracy: {:.3f}'.format(accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_subset200_eval_csv = subparsers.add_parser('create_subset200_eval_csv')
    parser_create_subset200_eval_csv.add_argument('--workspace', type=str, required=True)

    parser_plot_piano_solo_p_r_f1 = subparsers.add_parser('plot_piano_solo_p_r_f1')
    parser_plot_piano_solo_p_r_f1.add_argument('--subset200_eval_with_labels_path', type=str, required=True)
    parser_plot_piano_solo_p_r_f1.add_argument('--surname_in_youtube_title', action='store_true', default=False)

    parser_create_subset200_piano_solo_eval_csv = subparsers.add_parser('create_subset200_piano_solo_eval_csv')
    parser_create_subset200_piano_solo_eval_csv.add_argument('--workspace', type=str, required=True)

    parser_piano_solo_meta_accuracy = subparsers.add_parser('piano_solo_meta_accuracy')
    parser_piano_solo_meta_accuracy.add_argument('--subset200_piano_solo_eval_with_labels_path', type=str, required=True)
    parser_piano_solo_meta_accuracy.add_argument('--surname_in_youtube_title', action='store_true', default=False)

    parser_piano_solo_performed_ratio = subparsers.add_parser('piano_solo_performed_ratio')
    parser_piano_solo_performed_ratio.add_argument('--subset200_piano_solo_eval_with_labels_path', type=str, required=True)
    parser_piano_solo_performed_ratio.add_argument('--surname_in_youtube_title', action='store_true', default=False)

    parser_individual_composer_piano_solo_meta_accuracy = subparsers.add_parser('individual_composer_piano_solo_meta_accuracy')
    parser_individual_composer_piano_solo_meta_accuracy.add_argument('--workspace', type=str, required=True)
    parser_individual_composer_piano_solo_meta_accuracy.add_argument('--surname', type=str, required=True)
    parser_individual_composer_piano_solo_meta_accuracy.add_argument('--firstname', type=str, required=True)
    parser_individual_composer_piano_solo_meta_accuracy.add_argument('--surname_in_youtube_title', action='store_true', default=False)

    args = parser.parse_args()

    if args.mode == 'create_subset200_eval_csv':
        create_subset200_eval_csv(args)

    elif args.mode == 'plot_piano_solo_p_r_f1':
        plot_piano_solo_p_r_f1(args)

    elif args.mode == 'create_subset200_piano_solo_eval_csv':
        create_subset200_piano_solo_eval_csv(args)

    elif args.mode == 'piano_solo_meta_accuracy':
        piano_solo_meta_accuracy(args)

    elif args.mode == 'piano_solo_performed_ratio':
        piano_solo_performed_ratio(args)

    elif args.mode == 'individual_composer_piano_solo_meta_accuracy':
        individual_composer_piano_solo_meta_accuracy(args)

    else:
        raise Exception('Incorrect argument!')