import os
import csv
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt


def align(args):
    """
    1. Align MAESTRO pieces with sequenced MIDIs. 
    2. Align transcribed pieces with sequenced MIDIs.
    The alignment toolbox is based on [1].

    [1] Nakamura, E., Yoshii, K. and Katayose, H., 2017. Performance Error 
    Detection and Post-Processing for Fast and Accurate Symbolic Music 
    Alignment. In ISMIR (pp. 347-353).
    """
    csv_path = 'midis_for_evaluation/groundtruth_maestro_giantmidi-piano.csv'
    align_tools_dir = './AlignmentTool_v190813'

    os.makedirs('_tmp', exist_ok=True)
    os.makedirs('aligned_results', exist_ok=True)

    with open(csv_path, 'r') as fr:
        reader = csv.reader(fr, delimiter='\t')
        lines = list(reader)

    lines = lines[1:]   # Remove header
    align_time = time.time()

    for n, line in enumerate(lines):
        [piece_name, gt_name, maestro_name, giantmidi_name] = line
        print(n, piece_name)

        # Copy MIDI files
        cmd = 'cp "midis_for_evaluation/ground_truth/{}" "{}/{}"; '.format(gt_name, align_tools_dir, gt_name)
        cmd += 'cp "midis_for_evaluation/maestro/{}" "{}/{}"; '.format(maestro_name, align_tools_dir, maestro_name)
        cmd += 'cp "midis_for_evaluation/giantmidi-piano/{}" "{}/{}"; '.format(giantmidi_name, align_tools_dir, giantmidi_name)
        print(cmd)
        os.system(cmd)

        # Align
        cmd = 'cd {}; '.format(align_tools_dir)
        cmd += './MIDIToMIDIAlign.sh {} {}; '.format(gt_name[0 : -4], maestro_name[0 : -4])
        cmd += './MIDIToMIDIAlign.sh {} {}; '.format(gt_name[0 : -4], giantmidi_name[0 : -4])
        cmd += 'cd ..; '
        print(cmd)
        os.system(cmd)

        # Copy aligned results
        cmd = 'cp {}/{}_corresp.txt aligned_results/{}_corresp.txt; '.format(
            align_tools_dir, maestro_name[0 : -4], maestro_name[0 : -4])
        cmd += 'cp {}/{}_corresp.txt aligned_results/{}_corresp.txt; '.format(
            align_tools_dir, giantmidi_name[0 : -4], giantmidi_name[0 : -4])
        print(cmd)
        os.system(cmd)

    print('Finished! {:.3f} s'.format(time.time() - align_time))


def get_stats(csv_path):
    """Parse aligned results csv file to get results.

    Args:
      csv_path: str, aligned result path, e.g., xx_corresp.txt

    Returns:
      stat_dict, dict, keys: true positive (TP), deletion (D), insertion (I), 
        substitution (S), error rate (ER), ground truth number (N)
    """
    with open(csv_path, 'r') as fr:
        reader = csv.reader(fr, delimiter='\t')
        lines = list(reader)

    lines = lines[1 :]

    TP, D, I, S = 0, 0, 0, 0
    align_counter = []
    ref_counter = []

    for line in lines:
        line = line[0 : -1]
        [alignID, _, _, alignPitch, _, refID, _, _, refPitch, _] = line

        if alignID != '*' and refID != '*':
            if alignPitch == refPitch:
                TP += 1
            else:
                S += 1

        if alignID == '*':
            D += 1

        if refID == '*':
            I += 1

    N = TP + D + S
    ER = (D + I + S) / N

    print('TP: {}, D: {}, I: {}, S: {}'.format(TP, D, I, S))
    print('ER: {:.4f}'.format(ER))

    stat_dict = {'TP': TP, 'D': D, 'I': I, 'S': S, 'ER': ER, 'N': N}
    return stat_dict


def plot_box_plot(args):
    """Plot box plot of aligned results.
    """

    # Paths
    csv_path = 'midis_for_evaluation/groundtruth_maestro_giantmidi-piano.csv'
    fig_path = 'results/transcribed_metrics_box_plot.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    with open(csv_path, 'r') as fr:
        reader = csv.reader(fr, delimiter='\t')
        lines = list(reader)
    lines = lines[1:]

    piece_names = []
    maestro_stats = []
    giantmidi_stats = []

    # Collect statistics
    for n, line in enumerate(lines):
        [piece_name, gt_name, maestro_name, giantmidi_name] = line
        piece_names.append(piece_name)

        print('------ {}, {} ------'.format(n, piece_name))
        print('Maestro:')
        csv_path = 'aligned_results/{}_corresp.txt'.format(maestro_name[: -4])
        maestro_stats.append(get_stats(csv_path))

        print('GiantMIDI-Piano:')
        csv_path = 'aligned_results/{}_corresp.txt'.format(giantmidi_name[: -4])
        giantmidi_stats.append(get_stats(csv_path))

    # Plot
    fig, axs = plt.subplots(1,3, figsize=(12, 4))
    metrics_num = 4     # S, D, I, ER

    # Plot box plot of Maestro and GiantMIDI-Piano metrics
    for i, stats in enumerate([maestro_stats, giantmidi_stats]):
        metrics_mat = []    # (pieces_num, 4)
        for key in ['S', 'D', 'I']:
            metrics_mat.append([e[key] / e['N'] for e in stats])
        metrics_mat.append([e['ER'] for e in stats])
        metrics_mat = np.array(metrics_mat).T
        
        axs[i].set_ylim(0, 0.3)
        axs[i].boxplot(metrics_mat)
        axs[i].xaxis.set_ticks(np.arange(0, 5))
        axs[i].xaxis.set_ticklabels(['', 'S', 'D', 'I', 'ER'])

        for j in range(metrics_num):
            y = metrics_mat[:, j]
            x = np.random.normal(j + 1, 0.04, size=len(y))
            axs[i].plot(x, y, 'r.', alpha=0.2)

        if i == 0:
            maestro_metrics_mat = metrics_mat
        elif i == 1:
            giantmidi_metrics_mat = metrics_mat

    # Plot relative difference
    relative_diff = giantmidi_metrics_mat - maestro_metrics_mat
    axs[2].set_ylim(0., 0.3)
    axs[2].boxplot(relative_diff)
    axs[2].xaxis.set_ticks(np.arange(0, 5))
    axs[2].xaxis.set_ticklabels(['', 'S', 'D', 'I', 'ER'])

    for j in range(metrics_num):
        y = relative_diff[:, j]
        x = np.random.normal(j + 1, 0.04, size=len(y))
        axs[2].plot(x, y, 'r.', alpha=0.2)

    axs[0].set_title('Maestro')
    axs[1].set_title('GiantMIDI-Piano')
    axs[2].set_title('Relative difference')

    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save to {}'.format(fig_path))

    print('------ GiantMIDI-Piano sorted ERs ------')
    ers = giantmidi_metrics_mat[:, 3]
    # ers = maestro_metrics_mat[:, 3]
    sorted_indexes = np.argsort(ers)
    for n in range(len(sorted_indexes)):
        print('{}, ER: {:.3f}'.format(np.array(piece_names)[sorted_indexes[n]], 
            np.array(ers)[sorted_indexes[n]]))

    import crash
    asdf
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_align = subparsers.add_parser('align')
    parser_plot_box_plot = subparsers.add_parser('plot_box_plot')

    args = parser.parse_args()

    if args.mode == 'align':
        align(args)

    elif args.mode == 'plot_box_plot':
        plot_box_plot(args)

    else:
        raise Exception('Incorrect argument!')