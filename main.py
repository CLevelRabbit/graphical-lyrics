# Sorts = popularity release_date title
import os
import pandas as pd
import requests
import lyricsgenius as lg

API_KEY = ""
LYRICS_PATH = 'lyrics4.csv'
IMAGES_FMT = './images/{}.png'


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


def main():
    API_KEY = sys.argv[1]

    # Find out if first write or not
    if os.path.exists(LYRICS_PATH):
        write_kwargs = {'mode': 'a', 'index': False, 'header': False}
    else:
        write_kwargs = {'index': False}

    genius = lg.Genius(API_KEY, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True,
                       verbose=False)
    artists = ["רביד פלוטניק", "טונה", "טדי נגוסה", "ג'ימבו ג'יי", "איזי", "הצל", "זי קיי", "מיכאל כהן", "נורוז",
               "סאבלינימל", "מיכאל סוויסה", "סטטיק", "שאנן סטריט", "דודו פארוק", "שקל", "פלד", "לוקץ'", "טל טירנגל",
               "כליפי", "הדג נחש", "שאזאמאט", "אקו", "סימה נון", "קפה שחור חזק", "Shabak Samech"]
    artists = ['Eminem']
    for artist in artists:
        lyrics = get_lyrics(genius, artist)
        skills_pd = pd.json_normalize(lyrics)
        skills_pd.to_csv(LYRICS_PATH, **write_kwargs)


if __name__ == '__main__':
    main()
