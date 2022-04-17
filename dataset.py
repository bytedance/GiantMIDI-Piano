import os
import sys
import numpy as np
import argparse
import re
import string
import time
import csv
import glob
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from config import nationalities


def space_to_underscore(s):
    return s.replace(' ', '_')


def underscore_to_space(s):
    return s.replace('_', ' ')


def download_imslp_htmls(args):
    """Download html pages of all composers on IMSLP. In total 18,399 html 
    pages have been downloaded.
    """

    # Arguments & parameters
    workspace = args.workspace

    # Paths
    htmls_dir = os.path.join(workspace, 'htmls')
    os.makedirs(htmls_dir, exist_ok=True)

    # Download composer page
    html_path = os.path.join(workspace, 'Category:Composers.html')
    os.system('wget --quiet -O {} https://imslp.org/wiki/Category:Composers'.format(html_path))

    # Load html text
    with open(html_path, 'r') as fr:
        text = fr.read()

    # Get all composer names. Will looks like: 
    # ['A., Jag', 'Aadler, C. A.', 'Aagesen, Truid', ...]
    names = []

    for ch in string.ascii_uppercase[0 : 26]:
        """Search from A to Z. Get all composers by their surnames."""
        substring = text[re.search(f'"{ch}":\[', text).end() :]
        substring = substring[: re.search('\]', substring).start()]
        substring = substring.encode('utf8').decode('unicode_escape')
        names += substring[1 : -1].split('","')

    bgn_time = time.time()

    # Download html pages of all composers
    for n, name in enumerate(names):
        surname_firstname = name.split(', ')
        """E.g., ['A.', 'Jag']"""

        if len(surname_firstname) == 1:
            surname = surname_firstname[0]
            composer_link = 'https://imslp.org/wiki/Category:{}'.format(space_to_underscore(surname))
            html_path = os.path.join(htmls_dir, '{}.html'.format(surname))

        elif len(surname_firstname) == 2:
            [surname, firstname] = surname_firstname
            composer_link = 'https://imslp.org/wiki/Category:{}%2C_{}'.format(
                space_to_underscore(surname), space_to_underscore(firstname))
            html_path = os.path.join(htmls_dir, '{}, {}.html'.format(surname, firstname))

        os.system('wget --quiet -O "{}" "{}"'.format(html_path, composer_link))
        print(n, html_path, os.path.isfile(html_path))

    print('Finish! {:.3f} s'.format(time.time() - bgn_time))


def download_wikipedia_htmls(args):
    """Download wikipedia pages of composers if exist. In total 6,831 wikipedia
    pages are downloaded."""

    # Arguments & parameters
    workspace = args.workspace

    # Paths
    htmls_dir = os.path.join(workspace, 'htmls')
    html_names = sorted(os.listdir(htmls_dir))

    wikipedias_dir = os.path.join(workspace, 'wikipedias')
    os.makedirs(wikipedias_dir, exist_ok=True)

    # Download wikipedia of composers
    for n, html_name in enumerate(html_names):
        print(n, html_name)     # E.g., 'A., Jag.html'
        html_path = os.path.join(htmls_dir, html_name)

        surname_firstname = html_name[0 : -5].split(', ')
        if len(surname_firstname) == 2:
            [surname, firstname] = surname_firstname

        with open(html_path, 'r') as fr:
            text = fr.read()

        tmp = re.search('Detailed biography: <a href="', text)
        if tmp: # Only part of composer has wikipedia page
            text = text[tmp.end() : tmp.end() + 500]
            wikipedia_link = text[: re.search('"', text).start()]
            print(wikipedia_link)

            out_path = os.path.join(wikipedias_dir, '{}, {}.html'.format(surname, firstname))
            os.system('wget --quiet -O "{}" "{}"'.format(out_path, wikipedia_link))


def create_meta_csv(args):
    """Create GiantMIDI-Piano meta csv. This csv collects 144,079 music pieces 
    from all composers."""

    # Arguments & parameters
    workspace = args.workspace

    # Paths
    htmls_dir = os.path.join(workspace, 'htmls')
    wikipedias_dir = os.path.join(workspace, 'wikipedias')
    out_csv_path = os.path.join(workspace, 'full_music_pieces.csv')

    html_names = sorted(os.listdir(htmls_dir))

    meta_dict = {'surname': [], 'firstname': [], 'music': [], 'nationality': [], 
        'birth': [], 'death': []}

    for n, html_name in enumerate(html_names):
        print(n, html_name)     # E.g., 'A., Jag.html'

        surname_firstname = html_name[0 : -5].split(', ')
        if len(surname_firstname) == 2:
            [surname, firstname] = surname_firstname

        # Parse nationality, birth and death from Wikipedia
        wikipedia_path = os.path.join(wikipedias_dir, '{}, {}.html'.format(surname, firstname))
        (nationality, birth, death) = get_composer_info_from_wikipedia(wikipedia_path)
            
        # Parse music pieces from IMSLP html
        html_path = os.path.join(htmls_dir, html_name)
        music_names = get_music_names_from_imslp(html_path)
        music_names = [remove_suffix(music_name, firstname, surname) for music_name in music_names]

        for music_name in music_names:
            meta_dict['surname'].append(surname)
            meta_dict['firstname'].append(firstname)
            meta_dict['music'].append(music_name)
            meta_dict['nationality'].append(nationality)
            meta_dict['birth'].append(birth)
            meta_dict['death'].append(death)

    write_meta_dict_to_csv(meta_dict, out_csv_path)
    print('Write out to {}'.format(out_csv_path))


def get_composer_info_from_wikipedia(wikipedia_path):
    """Get nationality, birth and death from wikipedia."""

    nationality = None
    years = []

    if os.path.isfile(wikipedia_path):
        with open(wikipedia_path, 'r') as fr:
            text = fr.read()

        text = text.replace(': ', ':')
        text = text.replace('", "', '","')
        bgn = re.search('wgCategories":\[', text)

        if not bgn:
            return 'unknown', 'unknown', 'unknown'

        bgn = bgn.end()
        text = text[bgn + 1 :]
        fin = re.search('\]', text).start()
        text = text[0 : fin - 1]
        text = text.split('","')
        sentence = ' '.join(text)
        words = nltk.word_tokenize(sentence)
        pairs = nltk.pos_tag(words)

        for pair in pairs:
            if pair[1] == 'JJ': # Nationality
                if not nationality and pair[0] in nationalities:
                    nationality = pair[0]
            elif pair[1] == 'CD':   # Birth or death year
                try:
                    year = int(pair[0][0:4])
                    if year >= 1000 and year <= 9999:
                        years.append(year)
                except:
                    pass

        years = sorted(years)

    if len(years) >= 2:
        birth = str(years[0])
        death = str(years[1])
    else:
        birth = 'unknown'
        death = 'unknown'

    if not nationality:
        nationality = 'unknown'

    return nationality, birth, death


def get_music_names_from_imslp(ismlp_path):
    """Get all music names of a composer by parsing his / her IMSLP html page."""

    with open(ismlp_path, 'r') as fr:
        text = fr.read()

    # All music pieces information are before catpagejs
    obj = re.search("</div><script>if\(typeof catpagejs=='undefined'\)", text)
    if obj:
        text = text[: obj.start()]

    soup = BeautifulSoup(text, 'html.parser')
    links = soup.find_all('a')

    music_names = []

    for link in links:
        link = str(link)
        if 'categorypagelink' in link:
            """link looks like: '<a class="categorypagelink" href="/wiki/Je_t%27aime_Juliette_(A.,_Jag)" title="Je t\'aime Juliette (A., Jag)">Je t\'aime Juliette (A., Jag)</a>'
            """
            bgn = re.search('title=', link).end()
            link = link[bgn + 1 :]
            fin = re.search('>', link).start()
            music_name = link[0 : fin - 1]  # "Je t'aime Juliette (A., Jag)"
            music_names.append(music_name)

    for link in links:
        link = str(link)
        if 'next 200' in link:
            """link looks like: '<a class="categorypaginglink" href="/index.php?title=Category:Mozart,_Wolfgang_Amadeus&amp;pagefrom=Fantasia+in+f+minor%2C+k.0608%7E%7Emozart%2C+wolfgang+amadeus%0AFantasia+in+F+minor%2C+K.608+%28Mozart%2C+Wolfgang+Amadeus%29#mw-pages" title="Category:Mozart, Wolfgang Amadeus">next 200</a>'
            """
            bgn = re.search('href="', link).end()
            link = link[bgn :]
            fin = re.search('"', link).start()
            link = link[: fin]
            link = 'https://imslp.org{}'.format(link)
            link = link.replace('&amp;', '&')
            print(link)
            os.system('wget --quiet -O _tmp.html "{}"'.format(link))
            music_names += get_music_names_from_imslp('_tmp.html')
            break

    return music_names


def remove_suffix(music_name, firstname, surname):
    loct = re.search(f' \({surname}, {firstname}\)', music_name)
    if loct:
        music_name = music_name[0 : loct.start()]
    return music_name


def write_meta_dict_to_csv(meta_dict, out_csv_path):
    """Write meta dict to csv path."""
    with open(out_csv_path, 'w') as fw:
        line = '\t'.join([key for key in meta_dict.keys()])
        fw.write('{}\n'.format(line))

        for n in range(len(meta_dict['firstname'])):
            line = '\t'.join([str(meta_dict[key][n]) for key in meta_dict.keys()])
            fw.write('{}\n'.format(line))


def read_csv_to_meta_dict(csv_path):
    """Read csv file to meta_dict."""
    lines = []
    with open(csv_path, 'r') as fr:
        for line in fr.readlines():
            line = line.split('\n')[0].split('\t')
            lines.append(line)

    meta_dict = {key: [] for key in lines[0]}

    lines = lines[1 :]
    for m, line in enumerate(lines):
        for k, key in enumerate(meta_dict.keys()):
            meta_dict[key].append(line[k])

    return meta_dict

def _read_title_id(path):
    with open(path, 'r') as fr:
        lines = fr.readlines()

    if len(lines) == 2:
        title = lines[0].split('\n')[0]
        id = lines[1].split('\n')[0]
        return title, id
    else:
        return 'none', 'none'
    

def _too_many_requests(path):
    with open(path, 'r') as fr:
        lines = fr.readlines()

    for line in lines:
        print(line)
        if 'HTTP Error 429: Too Many Requests' in line:
            return True

    return False


def search_youtube(args):
    """Search music names on YouTube, and append searched YouTube titles and 
    IDs to meta csv."""

    # Arguments & parameters
    workspace = args.workspace
    mini_data = args.mini_data

    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''

    # Paths
    csv_path = os.path.join(workspace, 'full_music_pieces.csv')

    stdout_path = os.path.join(workspace, '_tmp', 'stdout.txt')
    error_path = os.path.join(workspace, '_tmp', 'error.txt')
    os.makedirs(os.path.dirname(stdout_path), exist_ok=True)

    youtube_csv_path = os.path.join(workspace, '{}full_music_pieces_youtube.csv'.format(prefix))

    meta_dict = read_csv_to_meta_dict(csv_path)

    youtube_meta_dict = {key: [] for key in meta_dict.keys()}
    youtube_meta_dict['youtube_title'] = []
    youtube_meta_dict['youtube_id'] = []

    n = 0
    while n < len(meta_dict['surname']):
        print(n, meta_dict['surname'][n])
        search_str = '{} {}, {}'.format(meta_dict['firstname'][n], 
            meta_dict['surname'][n], meta_dict['music'][n])
         
        youtube_simulate_str = 'youtube-dl --get-id --get-title ytsearch$1:"{}" 1>"{}" 2>"{}"'.\
            format(search_str, stdout_path, error_path)

        os.system(youtube_simulate_str)

        if _too_many_requests(error_path):
            sleep_seconds = 3600
            print('Too many requests! Sleep for {} s ...'.format(sleep_seconds))
            time.sleep(sleep_seconds)
            continue

        (title, id) = _read_title_id(stdout_path)
        youtube_meta_dict['youtube_title'].append(title)
        youtube_meta_dict['youtube_id'].append(id)

        for key in meta_dict.keys():
            youtube_meta_dict[key].append(meta_dict[key][n])
        print(', '.join([youtube_meta_dict[key][n] for key in youtube_meta_dict.keys()]))

        n += 1
        if mini_data and n == 10:
            break

    write_meta_dict_to_csv(youtube_meta_dict, youtube_csv_path)
    print('Write out to {}'.format(youtube_csv_path))


def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3


def jaccard_similarity(x, y):
    intersect = intersection(x, y)
    similarity = len(intersect) / max(float(len(x)), 1e-8)
    return similarity


def calculate_similarity(args):
    """Calculate and append the similarity between YouTube titles and IMSLP 
    music names to meta csv."""

    # Arguments & parameters
    workspace = args.workspace
    mini_data = args.mini_data

    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''

    # Paths
    youtube_csv_path = os.path.join(workspace, 
        '{}full_music_pieces_youtube.csv'.format(prefix))

    similarity_csv_path = os.path.join(workspace, 
        '{}full_music_pieces_youtube_similarity.csv'.format(prefix))

    # Meta info to be downloaded
    meta_dict = read_csv_to_meta_dict(youtube_csv_path)
    meta_dict['similarity'] = []
    meta_dict['surname_in_youtube_title'] = []

    tokenizer = RegexpTokenizer('[A-Za-z0-9ÇéâêîôûàèùäëïöüÄß]+')

    count = 0
    download_time = time.time()

    for n in range(len(meta_dict['surname'])):
        target_str = '{} {}, {}'.format(meta_dict['firstname'][n], 
            meta_dict['surname'][n], meta_dict['music'][n])

        target_str_without_firstname = '{}, {}'.format(
            meta_dict['surname'][n], meta_dict['music'][n])

        searched_str = meta_dict['youtube_title'][n]

        target_words = tokenizer.tokenize(target_str_without_firstname.lower())
        searched_words = tokenizer.tokenize(searched_str.lower())

        similarity = jaccard_similarity(target_words, searched_words)
        meta_dict['similarity'].append(str(similarity))

        if meta_dict['surname'][n] in meta_dict['youtube_title'][n]:
            meta_dict['surname_in_youtube_title'].append(1)
        else:
            meta_dict['surname_in_youtube_title'].append(0)

        if meta_dict['surname'][n] in meta_dict['youtube_title'][n]:
            meta_dict['surname_in_youtube_title'].append(1)
        else:
            meta_dict['surname_in_youtube_title'].append(0)

    write_meta_dict_to_csv(meta_dict, similarity_csv_path)
    print('Write out to {}'.format(similarity_csv_path))


def download_youtube(args):
    """Download IMSLP music pieces from YouTube. 59,969 files are downloaded 
    in Jan. 2020.
    """
    # Arguments & parameters
    workspace = args.workspace
    begin_index = args.begin_index
    end_index = args.end_index
    mini_data = args.mini_data

    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''

    # Paths
    similarity_csv_path = os.path.join(workspace, 
        '{}full_music_pieces_youtube_similarity.csv'.format(prefix))

    mp3s_dir = os.path.join(workspace, 'mp3s')
    os.makedirs(mp3s_dir, exist_ok=True)

    stdout_path = os.path.join(workspace, '_tmp', 'stdout.txt')
    error_path = os.path.join(workspace, '_tmp', 'error.txt')
    os.makedirs(os.path.dirname(stdout_path), exist_ok=True)

    # Meta info to be downloaded
    meta_dict = read_csv_to_meta_dict(similarity_csv_path)

    count = 0
    download_time = time.time()

    n = begin_index
    while n < min(end_index, len(meta_dict['surname'])):
        print('{}; {} {}; {}; {}'.format(n, meta_dict['firstname'][n], 
            meta_dict['surname'][n], meta_dict['music'][n], meta_dict['youtube_title'][n]))

        if float(meta_dict['similarity'][n]) > 0.6:
            count += 1
            
            bare_name = os.path.join('{}, {}, {}, {}'.format(
                meta_dict['surname'][n], meta_dict['firstname'][n], 
                meta_dict['music'][n], meta_dict['youtube_id'][n]).replace('/', '_'))
            
            youtube_str = 'youtube-dl -f bestaudio -o "{}/{}.%(ext)s" https://www.youtube.com/watch?v={} 1>"{}" 2>"{}"' \
                .format(mp3s_dir, bare_name, meta_dict['youtube_id'][n], stdout_path, error_path)
            
            os.system(youtube_str)
            
            if _too_many_requests(error_path):
                sleep_seconds = 3600
                print('Too many requests! Sleep for {} s ...'.format(sleep_seconds))
                time.sleep(sleep_seconds)
                continue

            # Convert to MP3
            audio_paths = glob.glob(os.path.join(mp3s_dir, '{}*'.format(bare_name)))
            print(audio_paths)

            if len(audio_paths) > 0:
                audio_path = audio_paths[0]

                mp3_path = os.path.join(mp3s_dir, '{}.mp3'.format(bare_name))
                
                os.system('ffmpeg -i "{}" -loglevel panic -y -ac 1 -ar 32000 "{}" '\
                    .format(audio_path, mp3_path))

                if os.path.splitext(audio_path)[-1] != '.mp3':
                    os.system('rm "{}"'.format(audio_path))
            
        n += 1
            
    print('{} out of {} audios are downloaded!'.format(count, end_index - begin_index))
    print('Time: {:.3f}'.format(time.time() - download_time))


def download_youtube_piano_solo(args):
    """Download piano solo of GiantMIDI-Piano. 10,848 files can be downloaded in 
    Jan. 2020.
    """
    # Arguments & parameters
    workspace = args.workspace
    begin_index = args.begin_index
    end_index = args.end_index
    mini_data = args.mini_data

    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''

    # Paths
    similarity_csv_path = os.path.join(workspace, 
        '{}full_music_pieces_youtube_similarity_pianosoloprob.csv'.format(prefix))

    mp3s_dir = os.path.join(workspace, 'mp3s_piano_solo')
    os.makedirs(mp3s_dir, exist_ok=True)

    stdout_path = os.path.join(workspace, '_tmp', 'stdout.txt')
    error_path = os.path.join(workspace, '_tmp', 'error.txt')
    os.makedirs(os.path.dirname(stdout_path), exist_ok=True)

    # Meta info to be downloaded
    meta_dict = read_csv_to_meta_dict(similarity_csv_path)

    count = 0
    download_time = time.time()

    n = begin_index
    while n < min(end_index, len(meta_dict['surname'])):
        print('{}; {} {}; {}; {}'.format(n, meta_dict['firstname'][n], 
            meta_dict['surname'][n], meta_dict['music'][n], meta_dict['youtube_title'][n]))

        try:
            prob = float(meta_dict['piano_solo_prob'][n])
        except ValueError as ve:
            print(f"{n}/{len(meta_dict['piano_solo_prob'])}", "SKIPPING ENTRY:", ve)
            n += 1
            continue

        if prob >= 0.5:

            bare_name = os.path.join('{}, {}, {}, {}'.format(
                meta_dict['surname'][n], meta_dict['firstname'][n], 
                meta_dict['music'][n], meta_dict['youtube_id'][n]).replace('/', '_'))
            
            youtube_str = 'youtube-dl -f bestaudio -o "{}/{}.%(ext)s" https://www.youtube.com/watch?v={} 1>"{}" 2>"{}"' \
                .format(mp3s_dir, bare_name, meta_dict['youtube_id'][n], stdout_path, error_path)
            
            os.system(youtube_str)
            
            if _too_many_requests(error_path):
                sleep_seconds = 3600
                print('Too many requests! Sleep for {} s ...'.format(sleep_seconds))
                time.sleep(sleep_seconds)
                continue

            # Convert to MP3
            audio_paths = glob.glob(os.path.join(mp3s_dir, '{}*'.format(bare_name)))
            print(audio_paths)

            if len(audio_paths) > 0:
                audio_path = audio_paths[0]

                mp3_path = os.path.join(mp3s_dir, '{}.mp3'.format(bare_name))
                
                os.system('ffmpeg -i "{}" -loglevel panic -y -ac 1 -ar 32000 "{}" '\
                    .format(audio_path, mp3_path))

                if os.path.splitext(audio_path)[-1] != '.mp3':
                    os.system('rm "{}"'.format(audio_path))
            count += 1
        n += 1
            
    print('{} out of {} audios are downloaded!'.format(count, end_index - begin_index))
    print('Time: {:.3f}'.format(time.time() - download_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Plot statistics
    parser_imslp_htmls = subparsers.add_parser('download_imslp_htmls')
    parser_imslp_htmls.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    parser_wikipedia = subparsers.add_parser('download_wikipedia_htmls')
    parser_wikipedia.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    parser_create_meta_csv = subparsers.add_parser('create_meta_csv')
    parser_create_meta_csv.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    parser_search = subparsers.add_parser('search_youtube')
    parser_search.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_search.add_argument('--mini_data', action='store_true', default=False)

    parser_similarity = subparsers.add_parser('calculate_similarity')
    parser_similarity.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_similarity.add_argument('--mini_data', action='store_true', default=False)

    parser_download_youtube = subparsers.add_parser('download_youtube')
    parser_download_youtube.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_download_youtube.add_argument('--begin_index', type=int, default=0)
    parser_download_youtube.add_argument('--end_index', type=int, required=True)
    parser_download_youtube.add_argument('--mini_data', action='store_true', default=False)

    parser_download_youtube_piano_solo = subparsers.add_parser('download_youtube_piano_solo')
    parser_download_youtube_piano_solo.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_download_youtube_piano_solo.add_argument('--begin_index', type=int, default=0)
    parser_download_youtube_piano_solo.add_argument('--end_index', type=int, required=True)
    parser_download_youtube_piano_solo.add_argument('--mini_data', action='store_true', default=False)

    # Parse arguments
    args = parser.parse_args()

    if args.mode == 'download_imslp_htmls':
        download_imslp_htmls(args)

    elif args.mode == 'download_wikipedia_htmls':
        download_wikipedia_htmls(args)

    elif args.mode == 'create_meta_csv':
        create_meta_csv(args)

    elif args.mode == 'search_youtube':
        search_youtube(args)

    elif args.mode == 'calculate_similarity':
        calculate_similarity(args)

    elif args.mode == 'download_youtube':
        download_youtube(args)

    elif args.mode == 'download_youtube_piano_solo':
        download_youtube_piano_solo(args)

    else:
        raise Exception('Error argument!')
