import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utilities import read_midi, TargetProcessor
from dataset import read_csv_to_meta_dict


note_names = [r'$\mathregular{A_{0}}$', r'$\mathregular{A{\sharp}_{0}}$', r'$\mathregular{B_{0}}$', 
        r'$\bf{C_{1}}$', r'$\mathregular{C{\sharp}_{1}}$', r'$\mathregular{D_{1}}$', r'$\mathregular{D{\sharp}_{1}}$', r'$\mathregular{E_{1}}$', r'$\mathregular{F_{1}}$', r'$\mathregular{F{\sharp}_{1}}$', r'$\mathregular{G_{1}}$', r'$\mathregular{G{\sharp}_{1}}$', r'$\mathregular{A_{1}}$', r'$\mathregular{A{\sharp}_{1}}$', r'$\mathregular{B_{1}}$', 
        r'$\bf{C_{2}}$', r'$\mathregular{C{\sharp}_{2}}$', r'$\mathregular{D_{2}}$', r'$\mathregular{D{\sharp}_{2}}$', r'$\mathregular{E_{2}}$', r'$\mathregular{F_{2}}$', r'$\mathregular{F{\sharp}_{2}}$', r'$\mathregular{G_{2}}$', r'$\mathregular{G{\sharp}_{2}}$', r'$\mathregular{A_{2}}$', r'$\mathregular{A{\sharp}_{2}}$', r'$\mathregular{B_{2}}$', 
        r'$\bf{C_{3}}$', r'$\mathregular{C{\sharp}_{3}}$', r'$\mathregular{D_{3}}$', r'$\mathregular{D{\sharp}_{3}}$', r'$\mathregular{E_{3}}$', r'$\mathregular{F_{3}}$', r'$\mathregular{F{\sharp}_{3}}$', r'$\mathregular{G_{3}}$', r'$\mathregular{G{\sharp}_{3}}$', r'$\mathregular{A_{3}}$', r'$\mathregular{A{\sharp}_{3}}$', r'$\mathregular{B_{3}}$', 
        r'$\bf{C_{4}}$', r'$\mathregular{C{\sharp}_{4}}$', r'$\mathregular{D_{4}}$', r'$\mathregular{D{\sharp}_{4}}$', r'$\mathregular{E_{4}}$', r'$\mathregular{F_{4}}$', r'$\mathregular{F{\sharp}_{4}}$', r'$\mathregular{G_{4}}$', r'$\mathregular{G{\sharp}_{4}}$', r'$\mathregular{A_{4}}$', r'$\mathregular{A{\sharp}_{4}}$', r'$\mathregular{B_{4}}$', 
        r'$\bf{C_{5}}$', r'$\mathregular{C{\sharp}_{5}}$', r'$\mathregular{D_{5}}$', r'$\mathregular{D{\sharp}_{5}}$', r'$\mathregular{E_{5}}$', r'$\mathregular{F_{5}}$', r'$\mathregular{F{\sharp}_{5}}$', r'$\mathregular{G_{5}}$', r'$\mathregular{G{\sharp}_{5}}$', r'$\mathregular{A_{5}}$', r'$\mathregular{A{\sharp}_{5}}$', r'$\mathregular{B_{5}}$', 
        r'$\bf{C_{6}}$', r'$\mathregular{C{\sharp}_{6}}$', r'$\mathregular{D_{6}}$', r'$\mathregular{D{\sharp}_{6}}$', r'$\mathregular{E_{6}}$', r'$\mathregular{F_{6}}$', r'$\mathregular{F{\sharp}_{6}}$', r'$\mathregular{G_{6}}$', r'$\mathregular{G{\sharp}_{6}}$', r'$\mathregular{A_{6}}$', r'$\mathregular{A{\sharp}_{6}}$', r'$\mathregular{B_{6}}$', 
        r'$\bf{C_{7}}$', r'$\mathregular{C{\sharp}_{7}}$', r'$\mathregular{D_{7}}$', r'$\mathregular{D{\sharp}_{7}}$', r'$\mathregular{E_{7}}$', r'$\mathregular{F_{7}}$', r'$\mathregular{F{\sharp}_{7}}$', r'$\mathregular{G_{7}}$', r'$\mathregular{G{\sharp}_{7}}$', r'$\mathregular{A_{7}}$', r'$\mathregular{A{\sharp}_{7}}$', r'$\mathregular{B_{7}}$', 
        r'$\bf{C_{8}}$']

chroma_names = [r'$\bf{C}$', r'$\mathregular{C{\sharp}}$', r'$\mathregular{D}$', r'$\mathregular{D{\sharp}}$', r'$\mathregular{E}$', r'$\mathregular{F}$', r'$\mathregular{F{\sharp}}$', r'$\mathregular{G}$', r'$\mathregular{G{\sharp}}$', r'$\mathregular{A}$', r'$\mathregular{A{\sharp}}$', r'$\mathregular{B}$']

frames_per_second = 100
begin_note = 21
classes_num = 88


def meta_info(args):
    """Calculate statistics of number of music pieces, nationalities, birth, 
    etc."""

    # Arugments & parameters
    workspace = args.workspace

    # Paths
    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob.csv')
    statistics_path = os.path.join(workspace, 'statistics.pkl')
    os.makedirs(os.path.dirname(statistics_path), exist_ok=True)

    meta_dict = read_csv_to_meta_dict(csv_path)
    """keys: ['surname', 'firstname', 'music', 'nationality', 'birth', 'death', 
    'youtube_title', 'youtube_id', 'similarity', 'piano_solo_prob', 'audio_name']"""

    for key in meta_dict.keys():
        meta_dict[key] = np.array(meta_dict[key])

    # Filter piano solo
    indexes = np.where(meta_dict['piano_solo_prob'].astype(np.float32) >= 0.5)[0]
    print('Music pieces num: {}'.format(len(indexes)))

    # Composers
    full_names = []

    for idx in indexes:
        full_names.append('{}, {}'.format(meta_dict['surname'][idx], meta_dict['firstname'][idx]))

    composers = np.array(list(set(full_names)))
    print('Composers num: {}'.format(len(composers)))

    # Number of works
    works_dict = {composer: {'audio_names': [], 'nationality': None, 
        'birth': None, 'death': None} for composer in composers}
    
    for idx in indexes:
        composer = '{}, {}'.format(meta_dict['surname'][idx], meta_dict['firstname'][idx])
        works_dict[composer]['audio_names'].append(meta_dict['audio_name'][idx])
        works_dict[composer]['nationality'] = meta_dict['nationality'][idx]
        works_dict[composer]['birth'] = meta_dict['birth'][idx]
        works_dict[composer]['death'] = meta_dict['death'][idx]

    number_of_works = np.array([len(works_dict[composer]['audio_names']) for composer in composers])
    statistics_dict = {'composers': composers, 'number_of_piano_works': number_of_works}

    # Sort by number of works
    sorted_idx = np.argsort(number_of_works)[::-1]
    sorted_list = []
    
    for idx in sorted_idx:
        composer = composers[idx]
        sorted_list.append([composer, len(works_dict[composer]['audio_names']), 
            works_dict[composer]['nationality'], works_dict[composer]['birth'], 
            works_dict[composer]['death']])
    """E.g., [..., ['Schmitt, Florent', 132, 'French', '1870', '1958'], ...]"""

    # Count by nationalities
    nationalities = [e[2] for e in sorted_list]
    unique_nationalities = list(set(nationalities))
    nationalities_count = []
    for na in unique_nationalities:
        nationalities_count.append(nationalities.count(na))

    _idxes = np.argsort(nationalities_count)[::-1]
    unique_nationalities = np.array(unique_nationalities)[_idxes]
    nationalities_count = np.array(nationalities_count)[_idxes]
    print('-------- Nationalities --------')
    print('Nationalities:', unique_nationalities)
    print('Count:', nationalities_count)

    # Plot nationalities
    fig_path = 'results/nationalities.pdf'
    N = len(nationalities_count)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.set_xlim(-1, N - 1)
    ax.set_ylim(0, 200)
    ax.set_xlabel('Nationalities', fontsize=14)
    ax.set_ylabel('Number of composers', fontsize=14)
    ax.bar(np.arange(N - 1), nationalities_count[1 : N], align='center', color='C0', alpha=1)
    ax.xaxis.set_ticks(np.arange(N - 1))
    ax.xaxis.set_ticklabels(unique_nationalities[1 : N], rotation=90, fontsize=12)
    ax.yaxis.grid(color='k', linestyle='--', linewidth=0.3)  # only horizontal grid
    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save fig to {}'.format(fig_path))

    # Year
    births = [int(e[3]) // 100 for e in sorted_list  if e[3] != 'unknown']
    unique_births = list(set(births))
    births_count = []
    for na in unique_births:
        births_count.append(births.count(na))

    _idxes = np.argsort(births_count)[::-1]
    unique_births = np.array(unique_births)[_idxes]
    births_count = np.array(births_count)[_idxes]
    print('-------- Birth centery --------')
    print('Birth centuries:', unique_births)
    print('Count:', births_count)

    # Lifespan
    lifespan = [int(e[4]) - int(e[3]) for e in sorted_list  if e[3] != 'unknown']
    unique_lifespan = list(set(lifespan))
    lifespan_count = []
    for na in unique_lifespan:
        lifespan_count.append(lifespan.count(na))

    _idxes = np.argsort(unique_lifespan)
    unique_lifespan = np.array(unique_lifespan)[_idxes]
    lifespan_count = np.array(lifespan_count)[_idxes]
    print('-------- Lifespan --------')
    print('Life span (years):', unique_lifespan)
    print('Count:', lifespan_count)

    # Dump statistics to disk
    pickle.dump(statistics_dict, open(statistics_path, 'wb'))
    print('Save to {}'.format(statistics_path))


def plot_composer_works_num(args):

    def _get_composer_works_num(meta_dict, indexes, composers):
        """Get the number of works of composers.

        Args:
          meta_dict, dict, keys: ['surname', 'firstname', 'music', 'nationality', 
              'birth', 'death', 'youtube_title', 'youtube_id', 'similarity', 
              'piano_solo_prob', 'audio_name']
          indexes: 1darray, e.g., [0, 2, 5, 6, ...]
          composers: list

        Returns:
          number_of_works: (composers_num,)
          sorted_indexes: (composers_num,)
        """
        # Composers
        full_names = []

        # Number of works
        works_dict = {composer: 0 for composer in composers}
        
        for idx in indexes:
            composer = '{}, {}'.format(meta_dict['surname'][idx], meta_dict['firstname'][idx])
            if composer in composers:
                works_dict[composer] += 1

        number_of_works = np.array([works_dict[composer] for composer in composers])

        # Sort by number of works
        sorted_indexes = np.argsort(number_of_works)[::-1]

        return number_of_works, sorted_indexes

    # Arugments & parameters
    workspace = args.workspace

    # Paths
    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob.csv')
    fig_path = 'results/composer_works_num.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    meta_dict = read_csv_to_meta_dict(csv_path)
    """keys: ['surname', 'firstname', 'music', 'nationality', 'birth', 'death', 
    'youtube_title', 'youtube_id', 'similarity', 'piano_solo_prob', 'audio_name']"""

    for key in meta_dict.keys():
        meta_dict[key] = np.array(meta_dict[key])

    # Filter by indexes, larger 1e-6 indicates audio has been downloaded, larger than 0.5 indicates piano solo
    all_indexes = np.where(meta_dict['piano_solo_prob'].astype(np.float32) >= 1e-6)[0]
    piano_indexes = np.where(meta_dict['piano_solo_prob'].astype(np.float32) >= 0.5)[0]

    # Get composer names
    piano_composers = []
    for idx in piano_indexes:
        piano_composers.append('{}, {}'.format(meta_dict['surname'][idx], meta_dict['firstname'][idx]))
    piano_composers = list(set(piano_composers))

    # Get composer works number
    composer_works_num_full, _ = _get_composer_works_num(meta_dict, all_indexes, piano_composers)
    composer_works_num_piano, sorted_indexes = _get_composer_works_num(meta_dict, piano_indexes, piano_composers)

    # Plot
    top_composers = 100
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    ax.set_xlim(-1, top_composers)
    ax.set_ylim(0, 300)
    ax.set_ylabel('Number of works', fontsize=15)
    line1 = ax.bar(np.arange(top_composers), np.array(composer_works_num_full)[sorted_indexes[0 : top_composers]], 
        align='center', color='pink', alpha=0.5, label='Full works')
    line2 = ax.bar(np.arange(top_composers), np.array(composer_works_num_piano)[sorted_indexes[0 : top_composers]], 
        align='center', color='C0', alpha=1, label='Piano works')
    ax.xaxis.set_ticks(np.arange(top_composers))
    ax.xaxis.set_ticklabels(np.array(piano_composers)[sorted_indexes[0 : top_composers]], rotation=90, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.yaxis.grid(color='k', linestyle='--', linewidth=0.3)
    ax.legend(handles=[line1, line2], fontsize=15, loc=1, framealpha=1.)
    plt.tight_layout(0, 0, 0)
    plt.savefig(fig_path)
    print('Save fig to {}'.format(fig_path))


def plot_composer_durations(args):

    def _get_composer_durations(meta_dict, indexes, composers):
        """Get the number of works of composers.

        Args:
          meta_dict, dict, keys: ['surname', 'firstname', 'music', 'nationality', 
              'birth', 'death', 'youtube_title', 'youtube_id', 'similarity', 
              'piano_solo_prob', 'audio_name', 'audio_duration']
          indexes: 1darray, e.g., [0, 2, 5, 6, ...]
          composers: list

        Returns:
          durations: (composers_num,)
          sorted_indexes: (composers_num,)
        """
        # Composers
        full_names = []

        # Number of works
        durations_dict = {composer: 0 for composer in composers}
        
        for idx in indexes:
            composer = '{}, {}'.format(meta_dict['surname'][idx], meta_dict['firstname'][idx])
            if composer in composers:
                durations_dict[composer] += float(meta_dict['audio_duration'][idx]) / 3600

        durations = np.array([durations_dict[composer] for composer in composers])

        # Sort by number of works
        sorted_indexes = np.argsort(durations)[::-1]

        return durations, sorted_indexes

    # Arugments & parameters
    workspace = args.workspace

    # Paths
    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob.csv')
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')
    fig_path = 'results/composer_durations.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    meta_dict = read_csv_to_meta_dict(csv_path)
    """keys: ['surname', 'firstname', 'music', 'nationality', 'birth', 'death', 
    'youtube_title', 'youtube_id', 'similarity', 'piano_solo_prob', 'audio_name']"""

    for key in meta_dict.keys():
        meta_dict[key] = np.array(meta_dict[key])

    # Filter by indexes, larger 1e-6 indicates audio has been downloaded, larger than 0.5 indicates piano solo
    all_indexes = np.where(meta_dict['piano_solo_prob'].astype(np.float32) >= 1e-6)[0]
    piano_indexes = np.where(meta_dict['piano_solo_prob'].astype(np.float32) >= 0.5)[0]

    # Get composer names
    piano_composers = []
    for idx in piano_indexes:
        piano_composers.append('{}, {}'.format(meta_dict['surname'][idx], meta_dict['firstname'][idx]))
    piano_composers = list(set(piano_composers))

    # Get composer works number
    composer_durations_full, _ = _get_composer_durations(meta_dict, all_indexes, piano_composers)
    composer_durations_piano, sorted_indexes = _get_composer_durations(meta_dict, piano_indexes, piano_composers)

    # Plot
    N = 100
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    ax.set_xlim(-1, N)
    ax.set_ylim(0, 40)
    ax.set_ylabel('Durations (h)', fontsize=15)
    line1 = ax.bar(np.arange(N), np.array(composer_durations_full)[sorted_indexes[0 : N]], 
        align='center', color='pink', alpha=0.5, label='Full works')
    line2 = ax.bar(np.arange(N), np.array(composer_durations_piano)[sorted_indexes[0 : N]], 
        align='center', color='C0', alpha=1, label='Piano works')
    ax.xaxis.set_ticks(np.arange(N))
    ax.xaxis.set_ticklabels(np.array(piano_composers)[sorted_indexes[0 : N]], rotation=90, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.yaxis.grid(color='k', linestyle='--', linewidth=0.3)
    ax.legend(handles=[line1, line2], fontsize=15, loc=1, framealpha=1.)
    plt.tight_layout(0, 0, 0)
    plt.savefig(fig_path)
    print('Save fig to {}'.format(fig_path))


def calculate_music_events_from_midi(args):

    # Arugments & parameters
    workspace = args.workspace

    midis_dir = os.path.join(workspace, "midis")
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')
    composer = 'Beethoven'

    midi_names = sorted(os.listdir(midis_dir))
    print('{} num: {}'.format(composer, len(midi_names)))

    all_music_events_dict = {}
    
    for n, midi_name in enumerate(midi_names):
        print(n, midi_name)
        midi_path = os.path.join(midis_dir, midi_name)

        midi_dict = read_midi(midi_path)

        segment_seconds = midi_dict['midi_event_time'][-1]

        target_processor = TargetProcessor(segment_seconds, frames_per_second, 
            begin_note, classes_num)

        (note_events, pedal_events) = target_processor.process(
            0, midi_dict['midi_event_time'], midi_dict['midi_event'])

        bare_name = os.path.splitext(midi_name)[0]

        all_music_events_dict[bare_name] = {
            'note_events': note_events, 
            'pedal_events': pedal_events, 
            'segment_seconds': segment_seconds}

    pickle.dump(all_music_events_dict, open(all_music_events_path, 'wb'))
    print('Dump to {}'.format(all_music_events_path))


def plot_note_histogram(args):
    
    # Arugments & parameters
    workspace = args.workspace

    # Paths
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')
    fig_path = 'results/note_histogram.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # Load events
    all_music_events = pickle.load(open(all_music_events_path, 'rb'))
    print('Load finish.')

    all_names = sorted([name for name in all_music_events.keys()])
    all_piano_notes = []

    for name in all_names:
        note_events = all_music_events[name]['note_events']
        piano_notes = [note_event['midi_note'] - begin_note for note_event in note_events]
        all_piano_notes += piano_notes

    counts = count_notes(all_piano_notes)

    fig, ax = plt.subplots(1, 1, figsize=(20, 3))
    ax.bar(np.arange(classes_num), counts, align='center')
    ax.xaxis.set_ticks(np.arange(classes_num))
    ax.xaxis.set_ticklabels(note_names, fontsize=10)
    ax.yaxis.set_ticks([0, 2e5, 4e5, 6e5, 8e5, 1e6, 1.2e6])
    ax.yaxis.set_ticklabels(['0', '200,000', '400,000', '600,000', '800,000', 
        '1,000,000', '1,200,000'], fontsize=10)
    ax.set_xlim(-1, classes_num)
    ax.set_ylabel('Number of notes', fontsize=15)
    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save to {}'.format(fig_path))

    print('Total notes: {}'.format(counts))


def plot_mean_std_notes(args):

    # Arugments & parameters
    workspace = args.workspace

    # Paths
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')
    meta_path = os.path.join(workspace, 'statistics.pkl')
    fig_path = 'results/note_mean_std.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # Load meta
    statistics_dict = pickle.load(open(meta_path, 'rb'))
    sorted_idxes = np.argsort(statistics_dict['number_of_piano_works'])[::-1]
    composers = statistics_dict['composers'][sorted_idxes[0 : 100]]

    # Load events
    all_music_events = pickle.load(open(all_music_events_path, 'rb'))
    print('Load finish.')

    all_names = sorted([name for name in all_music_events.keys()])

    stat_dict = {}

    # Calculate the mean and standard of notes of all composers
    for composer in composers:
        names = []

        for name in all_names:
            [surname, firstname] = name.split(', ')[0 : 2]
            if surname == composer.split(', ')[0] and firstname == composer.split(', ')[1]:
                names.append(name)

        all_piano_notes = []

        for name in names:
            note_events = all_music_events[name]['note_events']
            piano_notes = [note_event['midi_note'] - begin_note for note_event in note_events]
            all_piano_notes += piano_notes

        counts = count_notes(all_piano_notes)

        mean_ = np.sum(counts * np.arange(classes_num)) / np.sum(counts)
        std_ = np.sqrt(np.sum((np.arange(classes_num) - mean_) ** 2 * counts) / np.sum(counts))

        stat_dict[composer] = {'mean': mean_, 'std': std_}

    # Sort by ascending order
    mean_array = np.array([stat_dict[composer]['mean'] for composer in composers])
    std_array = np.array([stat_dict[composer]['std'] for composer in composers])
    sorted_idxes = np.argsort(mean_array)
    
    mean_array = mean_array[sorted_idxes]
    std_array = std_array[sorted_idxes]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    top_composers = 100
    ax.plot(std_array[sorted_idxes], c='r')
    (markerline, stemlines, baseline) = ax.stem(mean_array)
    # ax.setp(markerline, color='r', alpha=0.3)
    ax.fill_between(np.arange(top_composers), mean_array - std_array / 2, mean_array + std_array / 2, alpha=0.3)
    ax.plot(39 * np.ones(top_composers), linestyle='--', c='k')
    ax.xaxis.set_ticks(np.arange(top_composers))
    ax.xaxis.set_ticklabels(composers[sorted_idxes], rotation=90, fontsize=13)
    ax.yaxis.set_ticks([15, 27, 39, 51, 63])
    ax.yaxis.set_ticklabels(['$C_{2}$', '$C_{3}$', '$C_{4}$', '$C_{5}$', '$C_{6}$'], fontsize=13)
    ax.set_ylabel('Note names', fontsize=15)
    ax.set_xlim(-1, 100)
    ax.set_ylim((20, 60))
    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save to {}'.format(fig_path))


def mean_of_histogram(x):
    return np.sum(x * np.arange(len(x))) / np.sum(x)


def std_of_histogram(x):
    mean_ = mean_of_histogram(x)
    return np.sqrt(np.sum((np.arange(len(x)) - mean_) ** 2 * x) / np.sum(x))


def plot_notes_per_second_mean_std(args):

    # Arugments & parameters
    workspace = args.workspace

    # Paths
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')
    meta_path = os.path.join(workspace, 'statistics.pkl')
    fig_path = 'results/notes_per_second_mean_std.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # Load meta
    statistics_dict = pickle.load(open(meta_path, 'rb'))
    sorted_idxes = np.argsort(statistics_dict['number_of_piano_works'])[::-1]
    composers = statistics_dict['composers'][sorted_idxes[0 : 100]]

    # Load events
    all_music_events = pickle.load(open(all_music_events_path, 'rb'))
    print('Load finish.')

    all_names = sorted([name for name in all_music_events.keys()])

    stat_dict = {}

    # Calculate the mean and standard of notes of all composers
    for composer in composers:
        names = []

        for name in all_names:
            [surname, firstname] = name.split(', ')[0 : 2]
            if surname == composer.split(', ')[0] and firstname == composer.split(', ')[1]:
                names.append(name)

        notes_per_second = np.zeros(100)

        for name in names:
            note_events = all_music_events[name]['note_events']

            onset_times = np.array([note_event['onset_time'] for note_event in note_events])

            # Calculate the number of notes in 1 second
            tmp = onset_times.astype(np.int32)
            unique, counts = np.unique(tmp, return_counts=True)
            for e in counts:
                notes_per_second[e] += 1

        mean_ = mean_of_histogram(notes_per_second)
        std_ = std_of_histogram(notes_per_second)
        stat_dict[composer] = {'mean': mean_, 'std': std_}

    # Sort by ascending order
    mean_array = np.array([stat_dict[composer]['mean'] for composer in composers])
    std_array = np.array([stat_dict[composer]['std'] for composer in composers])
    sorted_idxes = np.argsort(mean_array)
    
    mean_array = mean_array[sorted_idxes]
    std_array = std_array[sorted_idxes]

    # Plot
    top_composers = 100
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    (markerline, stemlines, baseline) = ax.stem(mean_array)
    # ax.setp(stemlines, color='b', alpha=0.3)
    ax.fill_between(np.arange(top_composers), mean_array - std_array / 2, mean_array + std_array / 2, alpha=0.3)
    ax.xaxis.set_ticks(np.arange(top_composers))
    ax.xaxis.set_ticklabels(composers[sorted_idxes], rotation=90, fontsize=13)
    ax.yaxis.set_ticks([0, 5, 10, 15, 20])
    ax.yaxis.set_ticklabels([0, 5, 10, 15, 20], fontsize=13)
    ax.set_ylabel('Number of notes per second', fontsize=15)
    ax.set_xlim(-1, top_composers)
    ax.set_ylim((0, 20))
    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save to {}'.format(fig_path))


def plot_selected_composers_note_histogram(args):

    # Arugments & parameters
    workspace = args.workspace

    # Paths
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')
    meta_path = os.path.join(workspace, 'statistics.pkl')
    fig_path = 'results/selected_composers_note_histogram.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # Load meta
    statistics_dict = pickle.load(open(meta_path, 'rb'))
    sorted_idxes = np.argsort(statistics_dict['number_of_piano_works'])[::-1]
    composers = ['Bach, Johann Sebastian', 'Beethoven, Ludwig van', 'Liszt, Franz']

    # Load events
    all_music_events = pickle.load(open(all_music_events_path, 'rb'))
    print('Load finish.')

    # All music piece names
    all_names = sorted([name for name in all_music_events.keys()])

    fig, axs = plt.subplots(len(composers), 1, sharex=False, figsize=(20, 6))

    for j, composer in enumerate(composers):
        names = []
        all_piano_notes = []

        # Select the music pieces of a specific composer
        for name in all_names:
            [surname, firstname] = name.split(', ')[0 : 2]
            if surname == composer.split(', ')[0] and firstname == composer.split(', ')[1]:
                names.append(name)

        # Collect all notes of a specific composer
        for name in names:
            note_events = all_music_events[name]['note_events']
            piano_notes = [note_event['midi_note'] - begin_note for note_event in note_events]
            all_piano_notes += piano_notes

        counts = count_notes(all_piano_notes)

        axs[j].set_title(composer)
        axs[j].set_ylabel('Number of notes', fontsize=12)
        axs[j].set_xlim(-1, classes_num)
        axs[j].bar(np.arange(classes_num), counts, align='center')
        axs[j].xaxis.set_ticks(np.arange(classes_num))
        axs[j].xaxis.set_ticklabels(note_names, fontsize=10)
        axs[j].yaxis.set_ticks([0, 1e4, 2e4, 3e4])
        axs[j].yaxis.set_ticklabels(['0', '10,000', '20,000', '30,000'], fontsize=12)

    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save fig to {}'.format(fig_path))


def plot_selected_composers_chroma(args):

    def _count_chroma(piano_notes):
        counts = np.zeros(12)
        for k in range(88):
            counts[(k - 3) % 12] += piano_notes.count(k)
        return counts

    # Arugments & parameters
    workspace = args.workspace

    # Paths
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')
    meta_path = os.path.join(workspace, 'statistics.pkl')
    fig_path = 'results/selected_composers_chroma.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # Load meta
    statistics_dict = pickle.load(open(meta_path, 'rb'))
    sorted_idxes = np.argsort(statistics_dict['number_of_piano_works'])[::-1]
    composers = composers = ['Bach, Johann Sebastian', 'Mozart, Wolfgang Amadeus', 
        'Beethoven, Ludwig van', 'Chopin, Frédéric', 'Liszt, Franz', 'Debussy, Claude']

    # Load events
    all_music_events = pickle.load(open(all_music_events_path, 'rb'))
    print('Load finish.')

    # All music piece names
    all_names = sorted([name for name in all_music_events.keys()])

    fig, axs = plt.subplots(2, 3, sharex=False, figsize=(10, 4))

    for j, composer in enumerate(composers):
        names = []
        all_piano_notes = []

        # Select the music pieces of a specific composer
        for name in all_names:
            [surname, firstname] = name.split(', ')[0 : 2]
            if surname == composer.split(', ')[0] and firstname == composer.split(', ')[1]:
                names.append(name)

        # Collect all notes of a specific composer
        for name in names:
            note_events = all_music_events[name]['note_events']
            piano_notes = [note_event['midi_note'] - begin_note for note_event in note_events]
            all_piano_notes += piano_notes

        chroma_counts = _count_chroma(all_piano_notes)
        chroma_frequency = chroma_counts / np.sum(chroma_counts)

        chroma_num = 12
        axs[j // 3, j % 3].set_title(composer)
        axs[j // 3, j % 3].set_ylabel('Frequency')
        axs[j // 3, j % 3].set_ylim(0, 0.15)
        axs[j // 3, j % 3].yaxis.grid(color='k', linestyle='--', linewidth=0.3)
        axs[j // 3, j % 3].bar(np.arange(chroma_num), chroma_frequency, align='center')
        axs[j // 3, j % 3].xaxis.set_ticks(np.arange(chroma_num))
        axs[j // 3, j % 3].xaxis.set_ticklabels(chroma_names, fontsize=12)

    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save fig to {}'.format(fig_path))


def plot_selected_composers_intervals(args):

    def _count_interval(piano_notes):
        piano_notes = np.array(piano_notes)
        diff = piano_notes[1 :] - piano_notes[0 : -1]
        counts = np.zeros(23)

        for k in range(-11, 12):
            counts[k + 11] = diff.tolist().count(k)

        return counts

    # Arugments & parameters
    workspace = args.workspace

    # Paths
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')
    meta_path = os.path.join(workspace, 'statistics.pkl')
    fig_path = 'results/selected_composers_intervals.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # Load meta
    statistics_dict = pickle.load(open(meta_path, 'rb'))
    sorted_idxes = np.argsort(statistics_dict['number_of_piano_works'])[::-1]
    composers = composers = ['Bach, Johann Sebastian', 'Mozart, Wolfgang Amadeus', 
        'Beethoven, Ludwig van', 'Chopin, Frédéric', 'Liszt, Franz', 'Debussy, Claude']

    # Load events
    all_music_events = pickle.load(open(all_music_events_path, 'rb'))
    print('Load finish.')

    # All music piece names
    all_names = sorted([name for name in all_music_events.keys()])

    fig, axs = plt.subplots(2, 3, sharex=False, figsize=(10, 4))

    for j, composer in enumerate(composers):
        names = []
        all_piano_notes = []

        # Select the music pieces of a specific composer
        for name in all_names:
            [surname, firstname] = name.split(', ')[0 : 2]
            if surname == composer.split(', ')[0] and firstname == composer.split(', ')[1]:
                names.append(name)

        # Collect all notes of a specific composer
        for name in names:
            note_events = all_music_events[name]['note_events']
            piano_notes = [note_event['midi_note'] - begin_note for note_event in note_events]
            all_piano_notes += piano_notes

        interval_counts = _count_interval(all_piano_notes)
        interval_frequency = interval_counts / np.sum(interval_counts)

        intervals_num = 23  # From downward major 7-th to upward major 7-th
        axs[j // 3, j % 3].set_title(composer, fontsize=12)
        axs[j // 3, j % 3].set_xlim(-1, intervals_num)
        axs[j // 3, j % 3].set_ylim(0, 0.15)
        axs[j // 3, j % 3].yaxis.grid(color='k', linestyle='--', linewidth=0.3)
        axs[j // 3, j % 3].bar(np.arange(intervals_num), interval_frequency, align='center')
        axs[j // 3, j % 3].xaxis.set_ticks(np.arange(intervals_num))
        labels = ['-11', '\n-10', '-9', '\n-8', '-7', '\n-6', '-5', '\n-4', '-3', '\n-2', '-1', '\n0', '1', '\n2', '3', '\n4', '5', '\n6', '7', '\n8', '9', '\n10', '11']
        axs[j // 3, j % 3].xaxis.set_ticklabels(labels, fontsize=11, rotation='0')

        if j in [0, 3]:
            axs[j // 3, j % 3].set_ylabel('Frequency', fontsize=12)

        if j in [1, 2, 4, 5]:
            axs[j // 3, j % 3].tick_params(labelleft='off')

    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save fig to {}'.format(fig_path))


def plot_selected_composers_chords(args):

    def _count_chord(note_events):
        """Count chords. If multiple notes are within 50 ms, then we call them
        a chord. The chord is transposed so that its bass note is C. Then, 
        all notes are moved to a same octave.

        Args:
          note_events: list, e.g., [
            {'midi_note': 77, 'onset_time': 177.96, 'offset_time': 180.29, 'velocity': 103},
            {'midi_note': 29, 'onset_time': 178.64, 'offset_time': 180.34, 'velocity': 87},
            ...]

        Returns:
          chord_dict, dict, e.g., 
            {'{0}': 5872, '{0, 7}': 386, '{0, 2, 4, 5, 9, 10}': 2, '{0, 3}': 777, ...}
        """
        chord_dict = {}
        anchor_time = 0
        chord = [0]
        delta_time = 0.05

        for n in range(len(note_events) - 10):
            event_time = note_events[n]['onset_time']
            piano_note = note_events[n]['midi_note'] - 21

            if event_time - anchor_time > delta_time:
                """Collect chord"""
                tmp = np.array(chord)
                tmp -= np.min(tmp)  # Transpose the chord so that its bass note is C
                tmp %= 12   # Move all notes to a same octave
                tmp = str(set(sorted(tmp)))
                if tmp in chord_dict.keys():
                    chord_dict[tmp] += 1
                else:
                    chord_dict[tmp] = 1

                anchor_time = event_time
                chord = [piano_note]
            else:
                """Append notes to a chord"""
                chord.append(piano_note)

        return chord_dict

    def sort_dict(chord_dict):
        """Get chord list by their descending chord numer

        Args:
          chord_dict: dict, e.g., 
            {'{0}': 5872, '{0, 7}': 386, '{0, 2, 4, 5, 9, 10}': 2, '{0, 3}': 777, ...}

        Returns:
          sorted_list: list, e.g.,
            [('{0}', 10497), ('{0, 3}', 1517), ('{0, 4}', 1302), ...]
        """
        sorted_list = [(key, chord_dict[key]) for key in chord_dict.keys()]
        sorted_list.sort(key=lambda e: e[1], reverse=True)
        return sorted_list

    def _merge_inversion(sorted_list, chord_dict):
        """Merge inversions of a chord.

        Args:
          sorted_list: list, e.g.,
            [('{0}', 10497), ('{0, 3}', 1517), ('{0, 4}', 1302), ...]

        Output:
          merged_chord_dict: dict, .e.g.,
            [{'{0}': 10497, '{0, 3}': 2763, '{0, 4}': 2182, ...]
        """
        merged_chord_dict = {}
        for pair in sorted_list:

            inversions = _get_inversion_list(pair[0])
            exist = False
            for inv in inversions:
                if inv in merged_chord_dict.keys():
                    exist = True
                    break

            if not exist:
                merged_chord_dict[pair[0]] = 0
                for inversion in inversions:
                    if inversion in chord_dict.keys():
                        merged_chord_dict[pair[0]] += chord_dict[inversion]

        return merged_chord_dict

    def _get_inversion_list(key):
        chord = eval(key)
        chord = np.array(list(chord))
        inversions = []
        for c in chord:
            inversions.append(str(set(sorted((chord - c) % 12))))
        return inversions

    def _get_chords_with_n_notes(sorted_list, n_chord):
        output_list = []
        for e in sorted_list:
            if e[0].count(',') == n_chord - 1:
                output_list.append(e)
        return output_list

    def _normalize_list(input_list):
        """Histogram to frequency."""
        total = np.sum([e[1] for e in input_list])
        output_list = []
        for e in input_list:
            output_list.append((e[0], e[1] / total))
        return output_list

    # Arugments & parameters
    workspace = args.workspace
    n_chords = args.n_chords
    top_chords = 6  # Number of chords to plot

    # Paths
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')
    meta_path = os.path.join(workspace, 'statistics.pkl')
    fig_path = 'results/selected_composers_chords_{}.pdf'.format(n_chords)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # Load meta
    statistics_dict = pickle.load(open(meta_path, 'rb'))
    sorted_idxes = np.argsort(statistics_dict['number_of_piano_works'])[::-1]
    composers = composers = ['Bach, Johann Sebastian', 'Mozart, Wolfgang Amadeus', 
        'Beethoven, Ludwig van', 'Chopin, Frédéric', 'Liszt, Franz', 'Debussy, Claude']

    # Load events
    all_music_events = pickle.load(open(all_music_events_path, 'rb'))
    print('Load finish.')

    # All music piece names
    all_names = sorted([name for name in all_music_events.keys()])

    if n_chords == 3:
        figsize = (10, 4)
    elif n_chords == 4:
        figsize = (10, 4.3)

    fig, axs = plt.subplots(2, 3, sharex=False, figsize=figsize)

    for j, composer in enumerate(composers):
        names = []
        all_note_events = []

        # Select the music pieces of a specific composer
        for name in all_names:
            [surname, firstname] = name.split(', ')[0 : 2]
            if surname == composer.split(', ')[0] and firstname == composer.split(', ')[1]:
                names.append(name)

        # Collect all note events of a specific composer
        for name in names:
            note_events = all_music_events[name]['note_events']
            all_note_events += note_events
        """E.g., [
            {'midi_note': 77, 'onset_time': 177.96, 'offset_time': 180.29, 'velocity': 103},
            {'midi_note': 29, 'onset_time': 178.64, 'offset_time': 180.34, 'velocity': 87},
            ...]"""

        # Count number of chords
        chord_dict = _count_chord(all_note_events)
        """E.g., {'{0}': 5872, '{0, 7}': 386, '{0, 2, 4, 5, 9, 10}': 2, '{0, 3}': 777, ...}"""

        # Sort the number of chords by descending order
        sorted_list = sort_dict(chord_dict)
        """E.g., [('{0}', 10497), ('{0, 3}', 1517), ('{0, 4}', 1302), ...]"""

        # Merge inversion of chords to the most frequent inversion
        chord_dict = _merge_inversion(sorted_list, chord_dict)
        """E.g., [{'{0}': 10497, '{0, 3}': 2763, '{0, 4}': 2182, ...]"""

        # Sort the number of chords by descending order
        sorted_list = sort_dict(chord_dict)
        """E.g., [('{0}', 10497), ('{0, 3}', 2763), ('{0, 4}', 2182), ...]"""

        # Get chords with n notes
        sorted_list = _get_chords_with_n_notes(sorted_list, n_chords)

        # Normalize to frequency
        sorted_list = _normalize_list(sorted_list)

        # Choose top chords
        sorted_list = sorted_list[0 : top_chords]
        ids = [e[0] for e in sorted_list]
        counts = [e[1] for e in sorted_list] 

        axs[j // 3, j % 3].set_title(composer, fontsize=12)
        axs[j // 3, j % 3].bar(np.arange(len(counts)), counts, align='center')
        axs[j // 3, j % 3].set_ylim(0, 0.4)
        axs[j // 3, j % 3].yaxis.grid(color='k', linestyle='--', linewidth=0.3)
        axs[j // 3, j % 3].xaxis.set_ticks(np.arange(top_chords))
        axs[j // 3, j % 3].xaxis.set_ticklabels(ids, fontsize=8, rotation=90)

        if j in [0, 3]:
            axs[j // 3, j % 3].set_ylabel('Frequency', fontsize=12)

        if j in [1, 2, 4, 5]:
            axs[j // 3, j % 3].tick_params(labelleft='off')

    plt.tight_layout(0, 1, 1)
    plt.savefig(fig_path)
    print('Save fig to {}'.format(fig_path))


def count_notes(piano_notes):
    counts = []
    for k in range(88):
        counts.append(piano_notes.count(k))
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_meta_info = subparsers.add_parser('meta_info')
    parser_meta_info.add_argument('--workspace', type=str, required=True)
    
    parser_plot_composer_works_num = subparsers.add_parser('plot_composer_works_num')
    parser_plot_composer_works_num.add_argument('--workspace', type=str, required=True)

    parser_plot_composer_durations = subparsers.add_parser('plot_composer_durations')
    parser_plot_composer_durations.add_argument('--workspace', type=str, required=True)

    parser_events = subparsers.add_parser('calculate_music_events_from_midi')
    parser_events.add_argument('--workspace', type=str, required=True)

    parser_plot_note_histogram = subparsers.add_parser('plot_note_histogram')
    parser_plot_note_histogram.add_argument('--workspace', type=str, required=True)

    parser_plot_mean_std_notes = subparsers.add_parser('plot_mean_std_notes')
    parser_plot_mean_std_notes.add_argument('--workspace', type=str, required=True)

    parser_plot_notes_per_second_mean_std = subparsers.add_parser('plot_notes_per_second_mean_std')
    parser_plot_notes_per_second_mean_std.add_argument('--workspace', type=str, required=True)

    parser_plot_selected_composers_note_histogram = subparsers.add_parser('plot_selected_composers_note_histogram')
    parser_plot_selected_composers_note_histogram.add_argument('--workspace', type=str, required=True)

    parser_plot_selected_composers_chroma = subparsers.add_parser('plot_selected_composers_chroma')
    parser_plot_selected_composers_chroma.add_argument('--workspace', type=str, required=True)

    parser_plot_selected_composers_intervals = subparsers.add_parser('plot_selected_composers_intervals')
    parser_plot_selected_composers_intervals.add_argument('--workspace', type=str, required=True)

    parser_plot_selected_composers_chords = subparsers.add_parser('plot_selected_composers_chords')
    parser_plot_selected_composers_chords.add_argument('--workspace', type=str, required=True)
    parser_plot_selected_composers_chords.add_argument('--n_chords', type=int, required=True)

    parser_c = subparsers.add_parser('note_intervals')
    parser_c.add_argument('--workspace', type=str, required=True)

    args = parser.parse_args()
    
    if args.mode == 'meta_info':
        meta_info(args)

    elif args.mode == 'plot_composer_works_num':
        plot_composer_works_num(args)

    elif args.mode == 'plot_composer_durations':
        plot_composer_durations(args)

    elif args.mode == 'calculate_music_events_from_midi':
        calculate_music_events_from_midi(args)

    elif args.mode == 'plot_note_histogram':
        plot_note_histogram(args)

    elif args.mode == 'plot_mean_std_notes':
        plot_mean_std_notes(args)

    elif args.mode == 'plot_notes_per_second_mean_std':
        plot_notes_per_second_mean_std(args)

    elif args.mode == 'plot_pedals_per_piece_mean_std':
        plot_pedals_per_piece_mean_std(args)

    elif args.mode == 'plot_selected_composers_note_histogram':
        plot_selected_composers_note_histogram(args)

    elif args.mode == 'plot_selected_composers_chroma':
        plot_selected_composers_chroma(args)

    elif args.mode == 'plot_selected_composers_intervals':
        plot_selected_composers_intervals(args)

    elif args.mode == 'plot_selected_composers_chords':
        plot_selected_composers_chords(args)

    elif args.mode == 'note_intervals':
        note_intervals(args)

    else:
        raise Exception('Incorrect argument!')
