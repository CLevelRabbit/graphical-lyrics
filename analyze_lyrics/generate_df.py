import string
import pandas as pd

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
