import json
import csv

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir


# similarity_score.json
def main():
    name_list = [f.removesuffix('.txt') for f in listdir('../TRANSCRIPTS/TEXT')]
    # print(len(name_list))
    with open('../TRANSCRIPTS/similarity_score.json') as similarity_file:
        sim_list = json.loads(similarity_file.read())
        create_heatmap(sim_list, name_list)

    create_bar_graph()


def create_heatmap(similarity_list, name_list):
    all_list = []
    for name in name_list:
        value_list = [{'name': x['b'].removesuffix('.txt'), 'similarity': round(x['similarity'], 3)}
                      for x in similarity_list if x['a'].removesuffix('.txt') == name]
        value_list += [{'name': x['a'].removesuffix('.txt'), 'similarity': round(x['similarity'], 3)}
                       for x in similarity_list if x['b'].removesuffix('.txt') == name]
        value_list.append({'name': name, 'similarity': 0.00})

        value_list = list(sorted(value_list, key=lambda k: k['name']))
        print(value_list)
        value_list = [sim['similarity'] for sim in value_list]
        if len(value_list) > 1:
            all_list.append(value_list)
        else:
            print(name)

    print(all_list)
    print(len(value_list))
    fig, ax = plt.subplots()
    hm_values = np.array(all_list)
    print(hm_values)
    im, c_bar = heatmap(hm_values, name_list, name_list, ax=ax, cmap="YlGn", cbarlabel="Similarity Score")
    fig.tight_layout()
    plt.savefig("../TRANSCRIPTS/GRAPHS/similarity_graph.png")


# ratio_list.json
def axis_name(available_names):

    def if_name_available(item):
        if item['name'] in available_names:
            return False
        else:
            return True

    with open('../TRANSCRIPTS/ratio_list.json') as ratio_file:
        ratio_list = list(filter(if_name_available, json.loads(ratio_file.read())))
        with_ratio = ['{name} ({ratio:.2f})'.format(name=ratio['name'], ratio=ratio['ratio']) for ratio in ratio_list]

        print(with_ratio)
        return with_ratio


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


# ratios.csv
def create_bar_graph():
    with open('../TRANSCRIPTS/ratios.csv') as ratio_file:
        df = pd.read_csv(ratio_file, index_col=0)
        df.sort_values(by='ratio')
        fig = plt.figure()
        ax = fig.add_subplot(111)  # Create matplotlib axes
        ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.

        width = 0.4
        df.likes.plot(kind='bar', ax=ax, color='green', width=width, position=1, label='Likes')
        df.dislikes.plot(kind='bar', ax=ax, color="yellow", width=width, position=2, label='Dislikes')
        df.ratio.plot(kind='bar', ax=ax2, color="blue", width=width, position=3, label='Ratio')

        ax.set_ylabel('Likes and Dislikes Count')
        ax2.set_ylabel('Ratio')
        ax.legend(loc=0)
        ax2.legend(loc=2)
        plt.tight_layout()
        plt.savefig("../TRANSCRIPTS/GRAPHS/ratio_graph.png")


if __name__ == "__main__":  # run the script
    main()
