# Sorts = popularity release_date title
import sys
import os
import pandas as pd
import requests
from datetime import datetime

import lyricsgenius as lg

DB_NAME = 'lyrics-{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))
IMAGES_FMT = './images/{}.png'
ARTIST_FILE = 'artists.txt'


def get_lyrics(genius, artist, song_limit=None):
    print('Finding for %s' % artist)
    results = []
    artist = genius.search_artist(artist, max_songs=song_limit)
    download_image(artist)
    print('Found %d songs' % len(artist.songs))
    c = 0
    for song in artist.songs:

        date = song._body['release_date']
        if not date:
            date = song._body['release_date_for_display']
        if not song.lyrics:
            continue
        try:
            results.append({'artist': artist.name, 'title': song.title, 'date': date,
                            'lyrics': song.lyrics[song.lyrics.index('\n') + 1:].replace('\n', '\\n')})
            c += 1
        except Exception as e:
            print(e)
    print('Saved %d songs' % c)
    print()
    return results


def download_image(artist):
    img_data = requests.get(artist.image_url).content
    with open(IMAGES_FMT.format(artist.name), 'wb') as handler:
        handler.write(img_data)


def get_db_kwargs(db_path):
    # Find out if first write or not
    if os.path.exists(db_path):
        return {'mode': 'a', 'index': False, 'header': False}
    else:
        return {'index': False}


def main():
    api_key = sys.argv[1]

    print('DB name is: %s' % DB_NAME)

    with open(ARTIST_FILE, 'r', encoding='utf-8') as f:
        artists = f.read().splitlines()
    print('Reading artists: %s' % ', '.join(artists))

    genius = lg.Genius(api_key, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True,
                       verbose=True)

    for artist in artists:
        lyrics = get_lyrics(genius, artist)
        lyrics_df = pd.json_normalize(lyrics)

        write_kwargs = get_db_kwargs(DB_NAME)
        lyrics_df.to_csv(DB_NAME, **write_kwargs)


if __name__ == '__main__':
    main()
