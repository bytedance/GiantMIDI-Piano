import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
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

white_notes = np.array([0, 2] + [3 + e + i * 12 for e in [0, 2, 4, 5, 7, 9, 11] for i in range(7)] + [87])
black_notes = np.array([1] + [3 + e + i * 12 for e in [1, 3, 6, 8, 10] for i in range(7)])
C_notes = [3, 15, 27, 39, 51, 63, 75, 87]

chroma_names = [r'$\bf{C}$', r'$\mathregular{C{\sharp}}$', r'$\mathregular{D}$', 
    r'$\mathregular{D{\sharp}}$', r'$\mathregular{E}$', r'$\mathregular{F}$', 
    r'$\mathregular{F{\sharp}}$', r'$\mathregular{G}$', r'$\mathregular{G{\sharp}}$', 
    r'$\mathregular{A}$', r'$\mathregular{A{\sharp}}$', r'$\mathregular{B}$']

six_composers = ['Bach, Johann Sebastian', 'Mozart, Wolfgang Amadeus', 
        'Beethoven, Ludwig van', 'Chopin, Frédéric', 'Liszt, Franz', 
        'Debussy, Claude']

frames_per_second = 100
begin_note = 21
classes_num = 88


def meta_info(args):
    """Print works number and composers number.

    Args:
        workspace: str
        surname_in_youtube_title: bool

    Returns:
        NoReturn
    """

    # Arugments & parameters
    workspace = args.workspace
    surname_in_youtube_title = args.surname_in_youtube_title

    # Paths
    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob_split.csv')
    statistics_path = os.path.join(workspace, 'statistics.pkl')
    os.makedirs(os.path.dirname(statistics_path), exist_ok=True)

    # Read csv file
    meta_dict = read_csv_to_meta_dict(csv_path)

    audios_num = len(meta_dict['audio_name'])

    solo_piano_works_indexes = []

    for n in range(audios_num):

        if meta_dict['audio_name'][n] != "" and int(meta_dict['giant_midi_piano'][n]):
            if surname_in_youtube_title:
                if int(meta_dict['surname_in_youtube_title'][n]) == 1:
                    solo_piano_works_indexes.append(n)
            else:
                solo_piano_works_indexes.append(n)

    print('Music pieces num: {}'.format(len(solo_piano_works_indexes)))

    solo_piano_composer_names = []

    for index in solo_piano_works_indexes:
        solo_piano_composer_names.append(
            '{}, {}'.format(meta_dict['surname'][index], meta_dict['firstname'][index]))

    unique_solo_piano_composer_names = np.array(list(set(solo_piano_composer_names)))
    print('Composers num: {}'.format(len(unique_solo_piano_composer_names)))

    # Durations of solo piano works.
    durations_dict = {composer: 0 for composer in unique_solo_piano_composer_names}
    
    for index in solo_piano_works_indexes:
        composer = '{}, {}'.format(meta_dict['surname'][index], meta_dict['firstname'][index])
        durations_dict[composer] += float(meta_dict['audio_duration'][index])

    return durations_dict


def plot_composer_works_num(args):
    """Plot works number of composers.

    Args:
        workspace: str
        surname_in_youtube_title: bool

    Returns:
        NoReturn
    """

    def _get_composer_works_num(meta_dict, indexes, composers):
        """Get the number of works of composers.

        Args:
            meta_dict, dict, keys: ['surname', 'firstname', 'music', 'nationality', 
                'birth', 'death', 'youtube_title', 'youtube_id', 'similarity', 
                'piano_solo_prob', 'audio_name']
            indexes: 1darray, e.g., [0, 2, 5, 6, ...]
            composers: List[str]

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
        sorted_indexes = np.argsort(number_of_works, kind='stable')[::-1]

        return number_of_works, sorted_indexes

    # arugments & parameters
    workspace = args.workspace
    surname_in_youtube_title = args.surname_in_youtube_title

    # Paths
    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob_split.csv')
    fig_path = 'results/composer_works_num.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # Read csv file.
    meta_dict = read_csv_to_meta_dict(csv_path)
    
    for key in meta_dict.keys():
        meta_dict[key] = np.array(meta_dict[key])

    complete_works_indexes = []
    solo_piano_works_indexes = []

    for n in range(len(meta_dict['youtube_title'])):

        if meta_dict['audio_name'][n] == '':
            flag = False
        else:
            if surname_in_youtube_title and int(meta_dict['surname_in_youtube_title'][n]) == 0:
                flag = False
            else:
                flag = True

        if flag:
            complete_works_indexes.append(n)

            if int(meta_dict['giant_midi_piano'][n]) == 1:
                solo_piano_works_indexes.append(n)

    # Get solo piano composer names.
    solo_piano_composer_names = []

    for idx in solo_piano_works_indexes:
        solo_piano_composer_names.append(
            '{}, {}'.format(meta_dict['surname'][idx], meta_dict['firstname'][idx]))

    unique_solo_piano_composer_names = sorted(list(set(solo_piano_composer_names)))

    # Get composer works number.
    composer_complete_works_nums, _ = _get_composer_works_num(
        meta_dict, complete_works_indexes, unique_solo_piano_composer_names)

    composer_solo_piano_works_nums, sorted_indexes = _get_composer_works_num(
        meta_dict, solo_piano_works_indexes, unique_solo_piano_composer_names)

    # Plot
    top_composers = 100
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    ax.set_xlim(-1, top_composers)
    ax.set_ylim(0, 200)
    ax.set_ylabel('Number of works', fontsize=15)

    sorted_complete_works = np.array(composer_complete_works_nums)[sorted_indexes[0 : top_composers]]
    sorted_solo_piano_works = np.array(composer_solo_piano_works_nums)[sorted_indexes[0 : top_composers]]
    sorted_piano_composers = np.array(unique_solo_piano_composer_names)[sorted_indexes[0 : top_composers]]

    line1 = ax.bar(np.arange(top_composers), sorted_complete_works, 
        align='center', color='pink', alpha=0.5, label='Complete works')

    line2 = ax.bar(np.arange(top_composers), sorted_solo_piano_works, 
        align='center', color='C0', alpha=1, label='Solo piano works')

    for i, v in enumerate(sorted_solo_piano_works):
        ax.text(i - 0.3, max(v - len(str(v) * 7), 3), str(v), color='yellow', fontweight='bold', rotation=90)

    for i, v in enumerate(sorted_complete_works):
        ax.text(i - 0.3, min(200, v) + 5, str(v), color='red', fontweight='bold', rotation=90)

    ax.xaxis.set_ticks(np.arange(top_composers))
    ax.xaxis.set_ticklabels([composer[0 : 23] for composer in sorted_piano_composers], rotation=90, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.yaxis.grid(color='k', linestyle='--', linewidth=0.3)
    ax.legend(handles=[line1, line2], fontsize=15, loc=1, framealpha=1.)
    
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.savefig(fig_path)

    print('Save fig to {}'.format(fig_path))


def plot_composer_durations(args):
    """Plot works durations of composers.

    Args:
        workspace: str
        surname_in_youtube_title: bool

    Returns:
        NoReturn
    """

    def _get_composer_durations(meta_dict, works_indexes, composers):
        """Get the number of works of composers.

        Args:
          meta_dict, dict, keys: ['surname', 'firstname', 'music', 'nationality', 
              'birth', 'death', 'youtube_title', 'youtube_id', 'similarity', 
              'piano_solo_prob', 'audio_name', 'audio_duration']
          works_indexes: 1darray, e.g., [0, 2, 5, 6, ...]
          composers: list

        Returns:
          durations: (composers_num,)
          sorted_indexes: (composers_num,)
        """
        # Composers
        full_names = []

        # Number of works
        durations_dict = {composer: 0 for composer in composers}
        
        for index in works_indexes:
            composer = '{}, {}'.format(meta_dict['surname'][index], meta_dict['firstname'][index])
            if composer in composers:
                durations_dict[composer] += float(meta_dict['audio_duration'][index]) / 3600

        durations = np.array([durations_dict[composer] for composer in composers])

        # Sort by number of works
        sorted_indexes = np.argsort(durations)[::-1]

        return durations, sorted_indexes

    # arugments & parameters
    workspace = args.workspace
    surname_in_youtube_title = args.surname_in_youtube_title

    # Paths
    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob_split.csv')
    fig_path = 'results/composer_durations.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # Read csv file.
    meta_dict = read_csv_to_meta_dict(csv_path)

    for key in meta_dict.keys():
        meta_dict[key] = np.array(meta_dict[key])

    complete_works_indexes = []
    solo_piano_works_indexes = []

    for n in range(len(meta_dict['youtube_title'])):

        if meta_dict['audio_name'][n] == '':
            flag = False
        else:
            if surname_in_youtube_title and int(meta_dict['surname_in_youtube_title'][n]) == 0:
                flag = False
            else:
                flag = True

        if flag:
            complete_works_indexes.append(n)

            if meta_dict['piano_solo_prob'][n].astype(np.float32) >= 0.5:
                solo_piano_works_indexes.append(n)

    # Get composer names.
    solo_piano_composer_names = []

    for idx in solo_piano_works_indexes:
        solo_piano_composer_names.append('{}, {}'.format(meta_dict['surname'][idx], meta_dict['firstname'][idx]))

    unique_solo_piano_composer_names = list(set(solo_piano_composer_names))

    # Get composer works number.
    composer_durations_complete_works, _ = _get_composer_durations(
        meta_dict, complete_works_indexes, unique_solo_piano_composer_names)

    composer_durations_solo_piano_works, sorted_indexes = _get_composer_durations(
        meta_dict, solo_piano_works_indexes, unique_solo_piano_composer_names)

    # Plot
    N = 100
    sorted_full_durations = np.array(composer_durations_complete_works)[sorted_indexes[0 : N]]
    sorted_piano_durations = np.array(composer_durations_solo_piano_works)[sorted_indexes[0 : N]]
    sorted_piano_composers = np.array(unique_solo_piano_composer_names)[sorted_indexes[0 : N]]

    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    ax.set_xlim(-1, N)
    ax.set_ylim(0, 30)
    ax.set_ylabel('Duration (h)', fontsize=15)

    line1 = ax.bar(np.arange(N), sorted_full_durations, 
        align='center', color='pink', alpha=0.5, label='Complete works')

    line2 = ax.bar(np.arange(N), sorted_piano_durations, 
        align='center', color='C0', alpha=1, label='Solo piano works')

    for i, v in enumerate(sorted_piano_durations):
        round_v = int(np.around(v))
        ax.text(i - 0.3, max(v - len(str(round_v)) * 1.2, 0.5), str(round_v), 
            color='yellow', fontweight='bold', rotation=90)

    for i, v in enumerate(sorted_full_durations):
        v = int(np.around(v))
        ax.text(i - 0.3, min(30, v) + 1, str(v), color='red', fontweight='bold', rotation=90)

    ax.xaxis.set_ticks(np.arange(N))
    ax.xaxis.set_ticklabels([composer[0 : 23] for composer in sorted_piano_composers]
        , rotation=90, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.yaxis.grid(color='k', linestyle='--', linewidth=0.3)
    ax.legend(handles=[line1, line2], fontsize=15, loc=1, framealpha=1.)
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.savefig(fig_path)
    print('Save fig to {}'.format(fig_path))


def plot_nationalities(args):
    """Plot nationalities of composers. All composers nationalities, birhs, and
    deaths are manually checked.
    """

    # paths
    csv_path = 'manual_labels/composers_manually_checked.csv'
    fig_path = 'results/nationalities.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    
    # Read csv file.
    df = pd.read_csv(csv_path, sep='\t')

    nationalities = df['nationality'].values.tolist()
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

    c_unknown = 'pink'
    c_europe = 'C0'
    c_north_america = 'C1'
    c_south_america = 'C2'
    c_asia = 'C3'
    c_africa = 'C4'

    nationality_color_bar = {'unknown': c_unknown, 'German': c_europe, 
        'French': c_europe, 'American': c_north_america, 'British': c_europe, 
        'Italian': c_europe, 'Russian': c_europe, 'Austrian': c_europe, 
        'Polish': c_europe, 'Spanish': c_europe, 'Belgian': c_europe,  
        'Norwegian': c_europe, 'Czech': c_europe, 'Swedish': c_europe, 
        'Hungarian': c_europe, 'Brazilian': c_south_america, 'Dutch': c_europe, 
        'Danish': c_europe, 'Australian': c_europe, 'Canadian': c_north_america, 
        'Ukrainian': c_europe, 'Finnish': c_europe, 'Mexican': c_south_america, 
        'Swiss': c_europe, 'Bohemian': c_europe, 'Argentine': c_south_america, 
        'Irish': c_europe, 'Japanese': c_asia, 'Portuguese': c_europe, 
        'Armenian': c_europe, 'Romanian': c_europe, 'Chinese': c_asia, 
        'Cuban': c_south_america, 'Chilean': c_south_america, 
        'Croatian': c_europe, 'Venezuelan': c_south_america, 'Filipino': c_asia, 
        'Colombian': c_south_america, 'Uruguayan': c_south_america, 
        'Iranian': c_asia, 'Azerbaijani': c_asia, 'Bulgarian': c_europe, 
        'Indian': c_asia, 'Belarusian': c_europe, 'Icelandic': c_europe, 
        'Turkish': c_asia, 'Paraguayan': c_south_america, 'Egyptian': c_africa, 
        'Lithuanian': c_europe, 'Ecuadorean': c_south_america, 
        'Latvian': c_europe, 'Nigerian': c_africa, 'Greek': c_europe, 
        'Peruvian': c_south_america}

    # Plot nationalities
    N = len(nationalities_count)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.set_xlim(-1, N - 1)
    ax.set_ylim(0, 500)
    ax.set_xlabel('Nationalities', fontsize=14)
    ax.set_ylabel('Number of composers', fontsize=14)
    barlist = ax.bar(np.arange(N), nationalities_count, align='center', color='C0', alpha=1)
    for i in range(len(barlist)):
        barlist[i].set_color(nationality_color_bar[unique_nationalities[i]])
    ax.bar(0, 0, color=c_unknown, label='Unknown')
    ax.bar(0, 0, color=c_europe, label='European')
    ax.bar(0, 0, color=c_north_america, label='North American')
    ax.bar(0, 0, color=c_south_america, label='South American')
    ax.bar(0, 0, color=c_asia, label='Asian')
    ax.bar(0, 0, color=c_africa, label='Afican')
    ax.legend()
    ax.xaxis.set_ticks(np.arange(N + 1))
    ax.xaxis.set_ticklabels(['Unknown'] + unique_nationalities[1 : N].tolist() + ['']
        , rotation=90, fontsize=10)
    ax.yaxis.grid(color='k', linestyle='--', linewidth=0.3)

    for i, v in enumerate(nationalities_count):
        ax.text(i - 0.3, min(500, v) + 10, str(v), color='k', fontweight='bold'
            , rotation=90, fontsize=8)

    plt.tight_layout(pad=0, h_pad=1, w_pad=0)
    plt.savefig(fig_path)
    print('Save fig to {}'.format(fig_path))


def calculate_music_events_from_midi(args):
    """Collect all notes of GiantMIDI-Piano and dump into a pickle file.

    Args:
        workspace: str

    Returns:
        NoReturn
    """

    # Arugments & parameters
    workspace = args.workspace

    midis_dir = os.path.join(workspace, "midis")
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')

    midi_names = sorted(os.listdir(midis_dir))

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
    """Plot note historgram of GiantMIDI-Piano.

    Args:
        workspace: str
        surname_in_youtube_title: bool

    Returns:
        NoReturn
    """
    
    # arugments & parameters
    workspace = args.workspace
    surname_in_youtube_title = args.surname_in_youtube_title
    
    # paths
    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob_split.csv')
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')
    fig_path = 'results/note_histogram.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    
    # Read csv file.
    meta_dict = read_csv_to_meta_dict(csv_path)
    audios_num = len(meta_dict['audio_name'])

    # Load events
    all_music_events = pickle.load(open(all_music_events_path, 'rb'))
    print('Load finish.')

    all_names = sorted([name for name in all_music_events.keys()])
    all_piano_notes = []

    for n in range(audios_num):
    
        audio_name = meta_dict['audio_name'][n]

        if audio_name in all_music_events.keys():

            if surname_in_youtube_title:
                if int(meta_dict['surname_in_youtube_title'][n]) == 1:
                    flag = True
                else:
                    flag = False
            else:
                flag = True
        else:
            flag = False

        if flag:
            note_events = all_music_events[audio_name]['note_events']
            piano_notes = [note_event['midi_note'] - begin_note for note_event in note_events]
            all_piano_notes += piano_notes

    counts = count_notes(all_piano_notes)
    counts = np.array(counts)

    fig, ax = plt.subplots(1, 1, figsize=(20, 3))

    ax.bar(np.arange(classes_num)[white_notes], counts[white_notes], 
        align='center', color='k', fill=False, hatch='')

    ax.bar(np.arange(classes_num)[black_notes], counts[black_notes], 
        align='center', color='k')

    ax.xaxis.set_ticks(np.arange(classes_num))
    ax.xaxis.set_ticklabels(note_names, fontsize=10)

    colors = ['k'] * 88
    for i in C_notes:
        colors[i] = 'r'

    for xtick, color in zip(ax.get_xticklabels(), colors):
        xtick.set_color(color)

    if surname_in_youtube_title:
        ax.yaxis.set_ticks([0, 2e5, 4e5, 6e5, 8e5, 1e6])
        ax.yaxis.set_ticklabels(['0', '200,000', '400,000', '600,000', '800,000', 
            '1,000,000'], fontsize=10)
    else:
        ax.yaxis.set_ticks([0, 2e5, 4e5, 6e5, 8e5, 1e6, 1.2e6])
        ax.yaxis.set_ticklabels(['0', '200,000', '400,000', '600,000', '800,000', 
            '1,000,000', '1,200,000'], fontsize=10)
    ax.set_xlim(-1, classes_num)
    ax.set_ylabel('Number of notes', fontsize=15)
    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save to {}'.format(fig_path))
    print('Notes number: {}'.format(counts))
    print('Total notes: {}'.format(np.sum(counts)))


def get_all_names(all_music_events, csv_path, surname_in_youtube_title):

    meta_dict = read_csv_to_meta_dict(csv_path)
    audios_num = len(meta_dict['audio_name'])

    all_names = []

    for n in range(audios_num):
    
        audio_name = meta_dict['audio_name'][n]

        if audio_name in all_music_events.keys():

            if surname_in_youtube_title:
                if int(meta_dict['surname_in_youtube_title'][n]) == 1:
                    flag = True
                else:
                    flag = False
            else:
                flag = True
        else:
            flag = False

        if flag:
            all_names.append(audio_name)

    return all_names


def plot_selected_composers_note_histogram(args):
    """Plot note historgram of GiantMIDI-Piano.

    Args:
        workspace: str
        surname_in_youtube_title: bool

    Returns:
        NoReturn
    """

    # arugments & parameters
    workspace = args.workspace
    surname_in_youtube_title = args.surname_in_youtube_title

    # paths
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')
    fig_path = 'results/selected_composers_note_histogram.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    composers = ['Bach, Johann Sebastian', 'Beethoven, Ludwig van', 'Liszt, Franz']

    # Load note events.
    all_music_events = pickle.load(open(all_music_events_path, 'rb'))
    print('Load finish.')

    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob_split.csv')
    all_names = get_all_names(all_music_events, csv_path, surname_in_youtube_title)

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
        counts = np.array(counts)

        axs[j].set_title(composer)
        axs[j].set_ylabel('Number of notes', fontsize=12)
        axs[j].set_xlim(-1, classes_num)
        axs[j].bar(np.arange(classes_num)[white_notes], counts[white_notes], 
            align='center', fill=False, hatch='')
        axs[j].bar(np.arange(classes_num)[black_notes], counts[black_notes], 
            align='center', color='k')
        axs[j].xaxis.set_ticks(np.arange(classes_num))
        axs[j].xaxis.set_ticklabels(note_names, fontsize=10)

        colors = ['k'] * 88
        for i in C_notes:
            colors[i] = 'r'

        for xtick, color in zip(axs[j].get_xticklabels(), colors):
            xtick.set_color(color)

        axs[j].yaxis.set_ticks([0, 1e4, 2e4, 3e4])
        axs[j].yaxis.set_ticklabels(['0', '10,000', '20,000', '30,000'], fontsize=12)

    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save fig to {}'.format(fig_path))


def plot_mean_std_notes(args):
    """Plot note mean and standard values of composers.

    Args:
        workspace: str
        surname_in_youtube_title: bool

    Returns:
        NoReturn
    """

    # Arugments & parameters
    workspace = args.workspace
    surname_in_youtube_title = args.surname_in_youtube_title
    top_composers = 100

    # Paths
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')
    fig_path = 'results/note_mean_std.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # Load meta
    durations_dict = meta_info(args)
    composers = np.array([key for key in durations_dict.keys()])
    durations = np.array([durations_dict[key] for key in durations_dict.keys()])
    sorted_idxes = np.argsort(durations)[::-1]
    sorted_composers = composers[sorted_idxes][0 : top_composers]

    # Load events
    all_music_events = pickle.load(open(all_music_events_path, 'rb'))
    print('Load finish.')

    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob_split.csv')
    all_names = get_all_names(all_music_events, csv_path, surname_in_youtube_title)

    stat_dict = {}

    # Calculate the average and standard values of pitches of all composers.
    for composer in sorted_composers:
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
    mean_array = np.array([stat_dict[composer]['mean'] for composer in sorted_composers])
    std_array = np.array([stat_dict[composer]['std'] for composer in sorted_composers])
    sorted_idxes = np.argsort(mean_array)
    
    mean_array = mean_array[sorted_idxes]
    std_array = std_array[sorted_idxes]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    ax.plot(std_array[sorted_idxes], c='r')
    (markerline, stemlines, baseline) = ax.stem(mean_array)
    ax.fill_between(np.arange(top_composers), mean_array - std_array / 2, 
        mean_array + std_array / 2, alpha=0.3)
    ax.plot(39 * np.ones(top_composers), linestyle='--', c='k')

    for i, v in enumerate(mean_array):
        ax.text(i - 0.3, min(v + 9, 53), '{:.2f}'.format(v + 21), color='red', 
            fontweight='bold', rotation=90)

    ax.xaxis.set_ticks(np.arange(top_composers))

    txt_list = []
    for composer in sorted_composers[sorted_idxes]:
        if composer in six_composers:
            txt_list.append(r'$\bf{' + composer[0 : 23] + r'}$')
        else:
            txt_list.append(composer[0 : 23])

    ax.xaxis.set_ticklabels(txt_list, rotation=90, fontsize=13)
    
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
    surname_in_youtube_title = args.surname_in_youtube_title
    top_composers = 100

    # Paths
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')
    fig_path = 'results/notes_per_second_mean_std.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # Load meta
    durations_dict = meta_info(args)
    composers = np.array([key for key in durations_dict.keys()])
    durations = np.array([durations_dict[key] for key in durations_dict.keys()])
    sorted_idxes = np.argsort(durations)[::-1]
    sorted_composers = composers[sorted_idxes][0 : top_composers]

    # Load events
    all_music_events = pickle.load(open(all_music_events_path, 'rb'))
    print('Load finish.')

    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob_split.csv')
    all_names = get_all_names(all_music_events, csv_path, surname_in_youtube_title)

    stat_dict = {}

    # Calculate the average and standard values of notes per second of all composers.
    for composer in sorted_composers:
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
    mean_array = np.array([stat_dict[composer]['mean'] for composer in sorted_composers])
    std_array = np.array([stat_dict[composer]['std'] for composer in sorted_composers])
    sorted_idxes = np.argsort(mean_array)
    
    mean_array = mean_array[sorted_idxes]
    std_array = std_array[sorted_idxes]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    (markerline, stemlines, baseline) = ax.stem(mean_array)
    ax.fill_between(np.arange(top_composers), mean_array - std_array / 2, 
        mean_array + std_array / 2, alpha=0.3)
    
    for i, v in enumerate(mean_array):
        ax.text(i - 0.3, v + 3.5, '{:.2f}'.format(v), color='red', 
            fontweight='bold', rotation=90)

    ax.xaxis.set_ticks(np.arange(top_composers))

    txt_list = []
    for composer in sorted_composers[sorted_idxes]:
        if composer in six_composers:
            txt_list.append(r'$\bf{' + composer[0 : 23] + r'}$')
        else:
            txt_list.append(composer[0 : 23])

    ax.xaxis.set_ticklabels(txt_list, rotation=90, fontsize=13)

    ax.yaxis.set_ticks([0, 5, 10, 15, 20])
    ax.yaxis.set_ticklabels([0, 5, 10, 15, 20], fontsize=13)
    ax.set_ylabel('Number of notes per second', fontsize=15)
    ax.set_xlim(-1, top_composers)
    ax.set_ylim((0, 20))
    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save to {}'.format(fig_path))


def plot_selected_composers_chroma(args):

    def _count_chroma(piano_notes):
        counts = np.zeros(12)
        for k in range(88):
            counts[(k - 3) % 12] += piano_notes.count(k)
        return counts

    # Arugments & parameters
    workspace = args.workspace
    surname_in_youtube_title = args.surname_in_youtube_title

    # Paths
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')
    meta_path = os.path.join(workspace, 'statistics.pkl')
    fig_path = 'results/selected_composers_chroma.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # Load meta
    statistics_dict = pickle.load(open(meta_path, 'rb'))
    sorted_idxes = np.argsort(statistics_dict['number_of_piano_works'])[::-1]

    # Load events
    all_music_events = pickle.load(open(all_music_events_path, 'rb'))
    print('Load finish.')

    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob_split.csv')
    all_names = get_all_names(all_music_events, csv_path, surname_in_youtube_title)

    fig, axs = plt.subplots(2, 3, sharex=False, figsize=(10, 4))

    for j, composer in enumerate(six_composers):
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
        white_notes_octave = np.array([0, 2, 4, 5, 7, 9, 11])
        black_notes_octave = np.array([1, 3, 6, 8, 10])
        axs[j // 3, j % 3].bar(white_notes_octave, chroma_frequency[white_notes_octave],
            align='center', color='k', fill=False, hatch='')
        axs[j // 3, j % 3].bar(black_notes_octave, chroma_frequency[black_notes_octave],
            align='center', color='k')
        axs[j // 3, j % 3].xaxis.set_ticks(np.arange(chroma_num))
        axs[j // 3, j % 3].xaxis.set_ticklabels(chroma_names, fontsize=12)

        colors = ['r'] + ['k'] * 11
        for xtick, color in zip(axs[j // 3, j % 3].get_xticklabels(), colors):
            xtick.set_color(color)

        print(composer, chroma_counts)

    plt.tight_layout(pad=0, h_pad=1, w_pad=1)
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
    surname_in_youtube_title = args.surname_in_youtube_title

    # Paths
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')
    meta_path = os.path.join(workspace, 'statistics.pkl')
    fig_path = 'results/selected_composers_intervals.pdf'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # Load meta
    statistics_dict = pickle.load(open(meta_path, 'rb'))
    sorted_idxes = np.argsort(statistics_dict['number_of_piano_works'])[::-1]

    # Load events
    all_music_events = pickle.load(open(all_music_events_path, 'rb'))
    print('Load finish.')

    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob_split.csv')
    all_names = get_all_names(all_music_events, csv_path, surname_in_youtube_title)

    fig, axs = plt.subplots(2, 3, sharex=False, figsize=(10, 4))

    for j, composer in enumerate(six_composers):
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
        white_notes_inverval = np.array([0, 2, 4, 6, 7, 9, 11, 13, 15, 16, 18, 20, 22])
        black_notes_inverval = np.array([1, 3, 5, 8, 10, 12, 14, 17, 19, 21])
        axs[j // 3, j % 3].bar(white_notes_inverval, interval_frequency[white_notes_inverval],
            align='center', color='k', fill=False, hatch='')
        axs[j // 3, j % 3].bar(black_notes_inverval, interval_frequency[black_notes_inverval],
            align='center', color='k')
        axs[j // 3, j % 3].xaxis.set_ticks(np.arange(intervals_num))
        labels = ['-11', '\n-10', '-9', '\n-8', '-7', '\n-6', '-5', '\n-4', 
            '-3', '\n-2', '-1', '\n' + r'$\bf{0}$', '1', '\n2', '3', '\n4', '5', '\n6', '7', 
            '\n8', '9', '\n10', '11']
        axs[j // 3, j % 3].xaxis.set_ticklabels(labels, fontsize=11, rotation='0')
        colors = ['k'] * 11 + ['r'] + ['k'] * 11
        for xtick, color in zip(axs[j // 3, j % 3].get_xticklabels(), colors):
            xtick.set_color(color)

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
                # Collect chord.
                tmp = np.array(chord)
                tmp -= np.min(tmp)  # Transpose the chord so that its bass note is C.
                tmp %= 12   # Move all notes to a same octave.
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
        """Get chord list by their descending chord number.

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
    surname_in_youtube_title = args.surname_in_youtube_title
    top_chords = 6  # Number of chords to plot

    # Paths
    all_music_events_path = os.path.join(workspace, 'all_music_events.pkl')
    meta_path = os.path.join(workspace, 'statistics.pkl')
    fig_path = 'results/selected_composers_chords_{}.pdf'.format(n_chords)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # Load meta
    statistics_dict = pickle.load(open(meta_path, 'rb'))
    sorted_idxes = np.argsort(statistics_dict['number_of_piano_works'])[::-1]

    # Load events
    all_music_events = pickle.load(open(all_music_events_path, 'rb'))
    print('Load finish.')

    csv_path = os.path.join(workspace, 'full_music_pieces_youtube_similarity_pianosoloprob_split.csv')
    all_names = get_all_names(all_music_events, csv_path, surname_in_youtube_title)

    if n_chords == 3:
        figsize = (10, 4)
    elif n_chords == 4:
        figsize = (10, 4.3)

    fig, axs = plt.subplots(2, 3, sharex=False, figsize=figsize)

    for j, composer in enumerate(six_composers):
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

    plt.tight_layout(pad=0, h_pad=1, w_pad=1)
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
    parser_meta_info.add_argument('--surname_in_youtube_title', action='store_true', default=False)
    
    parser_plot_composer_works_num = subparsers.add_parser('plot_composer_works_num')
    parser_plot_composer_works_num.add_argument('--workspace', type=str, required=True)
    parser_plot_composer_works_num.add_argument('--surname_in_youtube_title', action='store_true', default=False)

    parser_plot_composer_durations = subparsers.add_parser('plot_composer_durations')
    parser_plot_composer_durations.add_argument('--workspace', type=str, required=True)
    parser_plot_composer_durations.add_argument('--surname_in_youtube_title', action='store_true', default=False)

    parser_plot_nationalities = subparsers.add_parser('plot_nationalities')

    parser_events = subparsers.add_parser('calculate_music_events_from_midi')
    parser_events.add_argument('--workspace', type=str, required=True)

    parser_plot_note_histogram = subparsers.add_parser('plot_note_histogram')
    parser_plot_note_histogram.add_argument('--workspace', type=str, required=True)
    parser_plot_note_histogram.add_argument('--surname_in_youtube_title', action='store_true', default=False)

    parser_plot_mean_std_notes = subparsers.add_parser('plot_mean_std_notes')
    parser_plot_mean_std_notes.add_argument('--workspace', type=str, required=True)
    parser_plot_mean_std_notes.add_argument('--surname_in_youtube_title', action='store_true', default=False)

    parser_plot_notes_per_second_mean_std = subparsers.add_parser('plot_notes_per_second_mean_std')
    parser_plot_notes_per_second_mean_std.add_argument('--workspace', type=str, required=True)
    parser_plot_notes_per_second_mean_std.add_argument('--surname_in_youtube_title', action='store_true', default=False)

    parser_plot_selected_composers_note_histogram = subparsers.add_parser('plot_selected_composers_note_histogram')
    parser_plot_selected_composers_note_histogram.add_argument('--workspace', type=str, required=True)
    parser_plot_selected_composers_note_histogram.add_argument('--surname_in_youtube_title', action='store_true', default=False)

    parser_plot_selected_composers_chroma = subparsers.add_parser('plot_selected_composers_chroma')
    parser_plot_selected_composers_chroma.add_argument('--workspace', type=str, required=True)
    parser_plot_selected_composers_chroma.add_argument('--surname_in_youtube_title', action='store_true', default=False)

    parser_plot_selected_composers_intervals = subparsers.add_parser('plot_selected_composers_intervals')
    parser_plot_selected_composers_intervals.add_argument('--workspace', type=str, required=True)
    parser_plot_selected_composers_intervals.add_argument('--surname_in_youtube_title', action='store_true', default=False)

    parser_plot_selected_composers_chords = subparsers.add_parser('plot_selected_composers_chords')
    parser_plot_selected_composers_chords.add_argument('--workspace', type=str, required=True)
    parser_plot_selected_composers_chords.add_argument('--n_chords', type=int, required=True)
    parser_plot_selected_composers_chords.add_argument('--surname_in_youtube_title', action='store_true', default=False)

    parser_c = subparsers.add_parser('note_intervals')
    parser_c.add_argument('--workspace', type=str, required=True)

    args = parser.parse_args()
    
    if args.mode == 'meta_info':
        meta_info(args)

    elif args.mode == 'plot_composer_works_num':
        plot_composer_works_num(args)

    elif args.mode == 'plot_composer_durations':
        plot_composer_durations(args)

    elif args.mode == 'plot_nationalities':
        plot_nationalities(args)

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
