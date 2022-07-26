import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw
from plotly.subplots import make_subplots
import circlify

from main import IMAGES_FMT
from analyze_lyrics.generate_df import get_year_df, get_counter_df, find_numbers_nlp

EXCLUDED_CITIES = ['רחובות', 'אלעד']
ISRAELI_CITIES = ['אום אל פחם', 'אופקים', 'אור יהודה', 'אור עקיבא', 'אילת', 'אלעד', 'אריאל', 'אשדוד', 'אשקלון',
                  'באקה אל גרביה', 'באר שבע', 'בית שאן', 'בית שמש', 'ביתר עילית', 'בני ברק', 'בת ים', 'גבעת שמואל',
                  'גבעתיים', 'דימונה', 'הוד השרון', 'הרצלייה', 'חדרה', 'חולון', 'חיפה', 'טבריה', 'טייבה', 'טירה',
                  'טירת כרמל', 'טמרה', 'יבנה', 'יהוד', 'יקנעם עילית', 'ירושלים', 'כפר יונה', 'כפר סבא', 'כפר קאסם',
                  'כרמיאל', 'לוד', 'מגדל העמק', 'מודיעין מכבים רעות', 'מודיעין עילית', 'מעלה אדומים', 'מעלות תרשיחא',
                  'נהרייה', 'נס ציונה', 'נצרת', 'נצרת עילית', 'נשר', 'נתיבות', 'נתניה', 'סחנין', 'עכו', 'עפולה',
                  'עראבה',
                  'ערד', 'פתח תקווה', 'צפת', 'קלנסווה', 'קריית אונו', 'קריית אתא', 'קריית ביאליק', 'קריית גת',
                  'קריית ים',
                  'קריית מוצקין', 'קריית מלאכי', 'קריית שמונה', 'ראש העין', 'ראשון לציון', 'רהט', 'רחובות', 'רמלה',
                  'רמת גן', 'רמת השרון', 'רעננה', 'שדרות', 'שפרעם', 'תל אביב', 'יפו']
CITY_REPLACEMENTS = [('ת"א', 'תל אביב'), ('קרית', 'קריית')]

NUMBERS = [
    ['0', 'אפס'],
    ['1', 'אחת', 'אחד'],
    ['2', 'שתיים', 'שניים'],
    ['3', 'שלוש', 'שלושה'],
    ['4', 'ארבע', 'ארבעה'],
    ['5', 'חמש', 'חמישה'],
    ['6', 'שש', 'שישה'],
    ['7', 'שבע', 'שבעה'],
    ['8', 'שמונה', 'שמונה'],
    ['9', 'תשע', 'תשעה']
]

CITY_PREFIXES = 'ולשבמ'
NUMBER_PREFIXES = 'הולשבמ'


def get_img(artist, is_round=True):
    """ Get image of artist as downloaded from Genius """
    img = Image.open(IMAGES_FMT.format(artist))
    if not is_round:
        return img
    h, w = img.size

    # creating luminous image
    lum_img = Image.new('L', [h, w], 0)
    draw = ImageDraw.Draw(lum_img)
    draw.pieslice([(0, 0), (h, w)], 0, 360, fill=255)
    img_arr = np.array(img)
    lum_img_arr = np.array(lum_img)
    final_img_arr = np.dstack((img_arr, lum_img_arr))
    return Image.fromarray(final_img_arr)


def draw_artists(df, xaxis, yaxis, xtitle=None, ytitle=None, title=None, upside_artists=[], is_table=False,
                 size_factor=None):
    """

    @param df: dataframe to draw from
    @param xaxis: column name for the X axis
    @param yaxis: column name for the Y axis
    @param xtitle: Optional - text title for X axis
    @param ytitle: Optional - text title for Y axis
    @param title: Optional - title for the graph
    @param upside_artists: Optional - names of artists to put the names of on top
    @param is_table: True to add table of the data
    @param size_factor: column name to act as a size factor for the image
    @return:
    """

    # Fixated graph size
    height = 800
    width = 1000

    # Names for axis'
    xtitle = xtitle if xtitle else xaxis
    ytitle = ytitle if ytitle else yaxis

    # Calc range for axis'
    xdiff = df[xaxis].max() - df[xaxis].min()
    ydiff = df[yaxis].max() - df[yaxis].min()

    # Calc range of visible parts of axis'
    xaxis_range = [df[xaxis].min() - xdiff * 0.05, df[xaxis].max() + xdiff * 0.05]
    yaxis_range = [df[yaxis].min() - ydiff * 0.1, df[yaxis].max() + ydiff * 0.1]

    # Calc size of images
    ratio = 16
    xsize = (xaxis_range[1] - xaxis_range[0]) / ratio
    ysize = width / ratio / height * (yaxis_range[1] - yaxis_range[0])

    # Add table
    if is_table:
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter"}, {"type": "table"}]],
                            column_widths=[0.7, 0.3], horizontal_spacing=0.01)
        fig.add_trace(
            go.Table(
                header=dict(values=[ytitle, xtitle, 'שם'],
                            font=dict(size=10),
                            align="right"),
                cells=dict(values=[df[yaxis].apply(lambda x: "{:.2f}".format(x)), df[xaxis], df.name],
                           align="right")),
            row=1, col=2)
    else:
        fig = go.Figure()

    # Create graph
    fig.add_trace(go.Scatter(x=df[xaxis], y=df[yaxis], mode="markers"))

    # Add names of artists (uncomment below part to add arrows)
    # Also add images
    if size_factor:
        maxi = df[size_factor].max()[0]
    for i, row in df.iterrows():
        if row['name'] in upside_artists:
            yshift = (ysize / 2) * 1.4
        else:
            yshift = (-ysize / 2) * 1.3
        fig.add_annotation(x=row[xaxis], y=row[yaxis] + yshift,
                           text=row['name'],
                           showarrow=False
                           # showarrow=True, arrowhead=2,
                           # xref='x', yref='y',
                           # ax=-40, ay=-40,

                           )

        # Add the artis's images
        fig.add_layout_image(dict(
            source=get_img(row['artist']),
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="middle",
            x=row[xaxis],
            y=row[yaxis],
            sizex=xsize,
            sizey=ysize,
            sizing="contain",
            opacity=1,
            layer="above"
        )
        )

    # Add aestetic metadata
    fig.update_layout(title=title,
                      xaxis_title=xtitle, yaxis_title=ytitle,
                      height=height, width=width,
                      yaxis_range=yaxis_range, xaxis_range=xaxis_range,
                      plot_bgcolor="#dfdfdf",
                      font=dict(
                          family="Alef",
                          size=18,
                          color="RebeccaPurple"
                      )
                      )

    return fig


def draw_years_by_trait(year_df, trait, trait_title=None, size_trait=None, title=''):
    # Fixated graph size
    height = 600
    width = 1000

    # Trait title
    if not trait_title:
        trait_title = trait

    # Calc for each axis: min/max, diff and range of visible axis
    xmax = year_df.year.max()
    xmin = year_df.year.min()
    ymax = year_df[trait].max()
    ymin = year_df[trait].min()

    xdiff = xmax - xmin
    ydiff = ymax - ymin

    xaxis_range = [xmin - xdiff * 0.05, xmax + xdiff * 0.1]
    yaxis_range = [ymin - ydiff * 0.1, ymax + ydiff * 0.1]

    # Calc size of each image
    ratio = 16
    xsize = (xaxis_range[1] - xaxis_range[0]) / ratio
    ysize = width / ratio / height * (yaxis_range[1] - yaxis_range[0])

    # Create totals+year df
    tdf = year_df.sort_values(by='year').groupby(['artist']).tail(1).sort_values(by=trait,
                                                                                 ascending=False).reset_index()

    # Add trace for each artist
    fig = go.Figure()
    for artist in tdf.artist:
        curr_df = year_df[year_df.artist == artist]
        if size_trait:
            sizes = curr_df.title * 2
        else:
            sizes = [1] * curr_df.shape[0]
        fig.add_trace(go.Scatter(
            x=curr_df.year, y=curr_df[trait],
            marker=dict(size=sizes),
            name=curr_df.name.iloc[0],
            legendgroup=curr_df.name.iloc[0],
        ))

    # Update aesthetics
    fig.update_layout(
        title=title,
        title_font_size=30,
        title_x=0.5,
        title_y=0.9,
        xaxis_title='שנה',
        yaxis_title=trait_title,
        height=height, width=width,
        legend_title=' ',
        yaxis_range=yaxis_range, xaxis_range=xaxis_range,
    )

    # Add images next to legend (this part is VERY flaky)
    for i, row in tdf.iterrows():
        x = xaxis_range[1] - xsize
        y = yaxis_range[1] - i / len(tdf) * ydiff * 1.2 - ysize * 1.3
        fig.add_layout_image(
            dict(
                source=get_img(row.artist),
                xref="x", yref="y",
                x=x, y=y,
                xanchor="left", yanchor="bottom",
                layer="above",
                sizing="contain",
                sizex=xsize, sizey=ysize
            )
        )
    fig.layout.legend.tracegroupgap = height / (len(tdf) * 1.9)

    return fig


def draw_years_by_trait_binned(df, trait, trait_title, size_trait='title', number_of_bins=4):
    figs = []

    # Seperate to bins
    starting_years_df = df.groupby(['name']).agg({'year': 'min'}).reset_index()
    skills_bins = pd.qcut(starting_years_df['year'], number_of_bins)
    starting_years_df['start_year_group'] = skills_bins.apply(lambda x: round(x.left))

    # Create the year agg df
    year_df = get_year_df(df)

    # Iterate bins
    bins = starting_years_df.start_year_group.unique().sort_values()
    for i, group in enumerate(bins):
        names = starting_years_df[starting_years_df.start_year_group == group].name

        # Fix namings (if 'number_of_bins' != 4, should be revisited)
        if i == len(bins) - 1:
            end_year = 2022
        else:
            end_year = bins[i + 1]
        if group == 2015:
            group = 2016
        if end_year == 2015:
            end_year = 2016

        # Draw actual graph
        fig = draw_years_by_trait(year_df[year_df.name.isin(names)], trait,
                                  trait_title=trait_title,
                                  title='ראפרים שהתחילו החל משנת %d ולפני שנת %d' % (group, end_year),
                                  size_trait=size_trait)
        figs.append(fig)

    return figs


def draw_top_words(words_counts_df, cnt, default_color='seashell', colors_dict={}, title=''):
    """
    Print circle graph of the top words (or combinations) in the words_df
    @param words_counts_df: df of all words
    @param cnt: how many to take from the top
    @param default_color: Optional - default color for each circle
    @param colors_dict: Optional - customized dict matching a word to a color
    @param title: Optional - Title for the graph
    """

    # Get the requested number of top words
    top_words_df = words_counts_df.groupby(['words']).sum().sort_values(by=['cnt'], ascending=False).reset_index().head(
        cnt)

    # Generate the coordinations and radiuses of the circles
    # (Using the "circlify" lib)
    circles = circlify.circlify(
        [{'word': row.words, 'datum': row.cnt} for _, row in top_words_df.iterrows()],
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1.05)
    )

    # Create the graph
    fig = go.Figure()
    fig.update_xaxes(
        range=[-1.05, 1.05],  # making slightly wider axes than -1 to 1 so no edge of circles cut-off
        showticklabels=False,
        showgrid=False,
        zeroline=False
    )
    fig.update_yaxes(
        range=[-1.05, 1.05],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    # Add all circles
    for circle in circles:
        x, y, r = circle
        word = circle.ex['word']

        # Draw the circle
        fig.add_shape(type="circle",
                      xref="x", yref="y",
                      x0=x - r, y0=y - r, x1=x + r, y1=y + r,
                      fillcolor=colors_dict.get(word, default_color),
                      line_color="black",
                      line_width=2,
                      )

        # Add the word(s)
        fig.add_annotation(x=x, y=y,
                           text='{}<br>{}'.format(word, circle.ex['datum']),
                           showarrow=False

                           )

    # For aesthetics
    fig.update_layout(width=800, height=800, plot_bgcolor="white", title=title)

    return fig


def print_top_words_combo(df, size_of_combo, cnt, title=None, colors_dict={}):
    """
    Print circle graph of combinations of words, e.g. 2\3 words in a row
    @param df: song df
    @param size_of_combo: how many words in a row to count
    @param cnt: how many to take from the top
    @param title: Optional - title for the graph
    @param colors_dict: Optional - customized dict matching a word to a color
    @return: the figure to show
    """
    results = []
    for j, row in df.iterrows():
        for i in range(len(row.words) - (size_of_combo - 1)):
            combo = ' '.join(row.words[i:i + size_of_combo])
            results.append([row.artist, row.name, row.title, combo])
    combo_words_df = pd.DataFrame(results, columns=['artist', 'name', 'title', 'words'])
    combo_words_df = combo_words_df.groupby(['artist'])['words'].value_counts().to_frame('cnt').reset_index()
    return draw_top_words(combo_words_df, cnt, title=title, colors_dict=colors_dict)


def print_metric_bar_graph(totals_df, trait, title=None):
    """
    a generic function the prints a bar graph of a totals column
    @param totals_df: the totals_df (agg df)
    @param trait: the column name to count
    @param title: optional - title for the graph
    @return: figure
    """
    # Sort by the trait
    tdf = totals_df.sort_values(by=trait).iloc[::-1].reset_index()

    # Create the graph
    fig = go.Figure(data=[go.Bar(x=tdf.name, y=tdf[trait])])

    # Calc image size
    size = tdf[trait].max() / 14
    cnt = tdf.shape[0]

    # Add the images
    for i, row in tdf.iterrows():
        fig.add_layout_image(
            dict(
                source=get_img(row.artist),
                xref="paper", yref="y",
                x=(i + 0.1) / cnt, y=row[trait] + size,
                xanchor="left", yanchor="top",
                layer="above",
                sizing="contain",
                sizex=1000000, sizey=size)
        )
    fig.update_layout(height=600, width=1000, title=title, yaxis_range=[0, tdf[trait].max() + size * 1.2])
    return fig


def print_top_cities(df, cities=ISRAELI_CITIES, num_of_cities=10, is_stacked=False):
    """
    Print the most mentioned israeli cities
    @param df: the song df
    @param cities: the list of cities
    @param num_of_cities: number of cities to put in graph
    @param is_stacked: if True - stacks each artist's part in the sum
    @return: figure
    """
    city_df = get_counter_df(df, 'city', cities, CITY_PREFIXES)
    city_df = city_df.groupby(['city']).sum().sort_values(by='cnt', ascending=False).head(
        num_of_cities).reset_index()
    if is_stacked:
        # Recalculate with only the top cities so it will be sorted. Should fix sometime
        city_df = get_counter_df(df, 'city', city_df.city, CITY_PREFIXES)
        fig = px.bar(city_df, x="city", y="cnt", color="name", color_discrete_sequence=px.colors.qualitative.Alphabet)
        fig.update_layout(height=650, width=1000)
    else:
        fig = go.Figure(data=[go.Bar(x=city_df.city, y=city_df.cnt)])
        fig.update_layout(height=450, width=550)
    return fig


def print_cities_artists(df, cities=ISRAELI_CITIES, cities_to_exclude=EXCLUDED_CITIES, num_of_artists=5):
    """
    Print the artists that mentiones the most cities
    @param df: the song df
    @param cities: a list of cities
    @param cities_to_exclude: cities not to count
    @param num_of_artists: how many artists to count
    @return: figure
    """
    city_df = get_counter_df(df, 'city', cities, CITY_PREFIXES, replacements=CITY_REPLACEMENTS)
    city_df = city_df[~city_df.city.isin(cities_to_exclude)].copy()
    city_df = city_df[city_df.cnt > 0]

    # Get only top artists
    top_city_artists = list(
        city_df.groupby(['name']).sum().sort_values(by='cnt', ascending=False).head(num_of_artists).index)
    city_df = city_df[city_df.name.isin(top_city_artists)]

    fig = px.bar(city_df, x="name", y="cnt", color="city")
    fig.update_xaxes(categoryorder='sum descending')
    fig.update_layout(height=850)
    return fig


def print_numbers(df, numbers=NUMBERS, is_ai=False):
    """
    Print the most mentioned numbers
    @param df: the song df
    @param numbers: a list of numbers
    @return: figure
    """
    if not is_ai:
        numbers_df = get_counter_df(df, 'number', numbers, NUMBER_PREFIXES)
        numbers_df = numbers_df.groupby(['number']).sum().sort_values(by='number', ascending=True).reset_index()
        fig = px.pie(numbers_df, values='cnt', names='number', labels='number')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    else:
        results = find_numbers_nlp(df)
        numbers = [i[0] for i in results]
        counts = [i[1] for i in results]
        tdf = pd.DataFrame.from_dict({'number': numbers, 'cnt': counts})
        fig = px.pie(tdf, values='cnt', names='number', labels='number')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

