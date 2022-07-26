import re
import string
import numerizer
import pandas as pd
from google_trans_new import google_translator
from spacy import load

ME_WORDS = ['אני', 'שלי', 'לי', 'עצמי', 'אותי', 'אנחנו', 'שלנו', 'לנו', 'עצמנו', 'אותנו']

def get_clean_df(db_name):
    """ Get the df """
    df = pd.read_csv(db_name)
    df = _get_clean_df(df)
    _add_metrics_to_df(df)
    return df


def _get_clean_df(df):
    """ Clean the df to get rid of irrelevant rows and add manageable columns """

    # Drop double songs
    df = df.drop_duplicates(subset=['title', 'artist'])

    # Only valid songs with valid date
    df = df[df.date.notna()]

    # Hebrew only name for convenience
    df['name'] = df.artist.apply(lambda x: x.split('-')[1].strip()).apply(lambda x: x.replace('Z (Israel)', 'אי-זי'))

    # Readable release year
    df['year'] = pd.to_datetime(df.date, infer_datetime_format=True).dt.year

    # Clean version of lyrics without "noise"
    df['lyrics_clean'] = df['lyrics'] \
        .str.replace('\\n', ' ', regex=False) \
        .str.replace('Embed', '', regex=False) \
        .str.replace('[{}״׳]'.format(string.punctuation), '', regex=True) \
        .str.replace('  ', ' ', regex=False)

    # List version of words, count words, and count uniq words
    df['words'] = df['lyrics_clean'].str.split(' ')
    df['total_words'] = df.words.apply(lambda x: len(x))
    df['uniq_words'] = df.words.apply(lambda x: len(set(x)))

    return df


def _add_metrics_to_df(df):
    """ Add general metrics of the song: reference to self, number of questions """

    def get_mes(row):
        """ lambda func to count references to self in function, using the artist name and general "self" words"""
        names = row.artist.replace('-', '').replace('  ', ' ').split(' ')
        names += ME_WORDS
        cnt = 0
        for name in names:
            cnt += row.words.count(name)
        return cnt

    # Count reference to self
    df['me'] = df.apply(lambda row: get_mes(row), axis=1)

    # Count number of questions
    df['questions'] = df['lyrics'].str.count('\?')


def _uniq_words(df):
    """ agg func to count number of unique words """
    return df.to_frame().words.explode().unique().shape[0]


def get_totals_df(df):
    """ Get df that summarizes metrics for artist """

    # Initial aggregation
    totals_df = df.groupby(['artist', 'name']).agg({'title': pd.Series.nunique,
                                                    'questions': 'sum',
                                                    'me': 'sum',
                                                    'total_words': 'sum',
                                                    'words': _uniq_words})

    # Fix layout and columns
    totals_df.columns = ['title', 'questions', 'me', 'total_words', 'uniq_words']
    totals_df.reset_index(inplace=True)

    # Add average metrics per song
    for col in ['questions', 'uniq_words', 'total_words', 'me']:
        totals_df[col + '_avg'] = totals_df[col] / totals_df.title

    for col in ['questions_avg', 'me_avg', 'uniq_words_avg']:
        totals_df[col + '_normalized'] = totals_df[col] / totals_df['total_words_avg']

    return totals_df


def get_helper_dfs(df):
    """
    Return the helper "words dfs"
    """
    words_df = df.explode('words')[['artist', 'title', 'date', 'words']].reset_index()
    words_counts_df = words_df.groupby(['artist'])['words'].value_counts().to_frame('cnt').reset_index()

    return words_df, words_counts_df


def get_year_df(df):
    """
    Create a df that aggregates each year
    """

    # Perform the agg
    year_df = df.groupby(['artist', 'name', 'year']).agg({'title': pd.Series.nunique,
                                                          'questions': 'sum',
                                                          'me': 'sum',
                                                          'total_words': 'sum',
                                                          'words': _uniq_words})
    year_df.columns = ['title', 'questions', 'me', 'total_words', 'uniq_words']

    # Add averages by somg
    for col in ['questions', 'uniq_words', 'total_words', 'me']:
        year_df[col + '_avg'] = year_df[col] / year_df.title
    year_df.reset_index(inplace=True)

    return year_df


def get_counter_df(df, column_name, phrases, prefixes, replacements=None):
    """
    Create a df the counts for each artist the appearance of each phrase (convoluted but works, hopefully)
    @param df: the song df
    @param column_name: the column name of the counted data in the output df
    @param phrases: a list (or list of lists) of the phrases to locate
    @param prefixes: prefixes to check before the words, like "lamed"\"bet" or other prefix hebrew letters
    @param replacements: a list of 2-tuples of src and dst to replace in the src lyrics
    @return: a df the counts the appearance of each phrase for each artist
    """
    def replace_and_locate(lyrics):
        """
        an apply func that makes the replacements and locates the regex
        """
        if replacements:
            for src, dst in replacements:
                lyrics = lyrics.replace(src, dst)
        return len(
            re.findall('|'.join([r'\b[{}]?{}\b'.format(prefixes, sub_phrase) for sub_phrase in phrase]), lyrics))

    tdf = df[['name', 'title', 'lyrics_clean']].copy()
    for phrase in phrases:
        if type(phrase) is not list:
            phrase = [phrase]
        tdf[phrase[0]] = df.lyrics_clean.apply(replace_and_locate)

    tdf = tdf.groupby('name').sum().stack().reset_index()
    tdf.columns = ['name', column_name, 'cnt']
    return tdf


def find_numbers_nlp(df, max_num=1000000):
    """
    Translate songs to english, run NLP model and locate numbers.
    @param df: the song df
    @param max_num: the highest number to find
    @return: returns a list of (number, count) tuples
    """
    totals = {}
    # Load AI modules
    translator = google_translator()
    nlp = load('en_core_web_sm')  # or any other model
    for song in df.lyrics_clean:
        # Translate song
        translate_text = translator.translate(song, lang_src='he', lang_tgt='en')
        # Parse NLP of the song
        doc = nlp(translate_text)
        # Find all numbers
        for text, option in doc._.numerize().items():
            for match in re.findall(r'\d+', option):
                match = int(match)
                if match > max_num:
                    continue
                cnt = totals.get(match, 0)
                totals[match] = cnt + 1

    results = [(number, count) for number, count in totals.items()]
    return results
