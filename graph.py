import json
import csv

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir


def main():
    # with open('youtube_vids.json') as f:
    # video_dict = json.loads(f.read())
    # objects = {video['name']: video['id'] for video in video_dict}
    name_list = [f.removesuffix('.txt') for f in listdir('../TRANSCRIPTS/TEXT')]
    # name_list, filtered_ratio = axis_name()
    # name_row = ["names"] + filtered_ratio
    name_row = ["names"] + name_list

    with open('../TRANSCRIPTS/similarity_score.json') as similarity_file:
        sim_dict = json.loads(similarity_file.read())
        # print(sim_dict)

    csv_array = []
    value_list = []
    for name in name_list:
        list_a = [x for x in sim_dict if x['a'].removesuffix('.txt') == name]
        obj_a = {compare['b'].removesuffix('.txt'): compare['similarity'] for compare in list_a}
        list_b = [x for x in sim_dict if x['b'].removesuffix('.txt') == name]
        obj_b = {compare['a'].removesuffix('.txt'): compare['similarity'] for compare in list_b}

        obj_a.update(obj_b)
        obj_a[name] = 0.000

        # print(obj_a)
        # print(obj_b)
        row = []
        for name_col in name_list:
            value = round(obj_a[name_col], 3)
            row.append(value)
        row_with_name = [name] + row
        print(row_with_name)
        csv_array.append(row_with_name)
        value_list.append(row)

    with open('../TRANSCRIPTS/heatmap_values.csv', 'w') as h_values:
        write = csv.writer(h_values)
        write.writerow(name_row)
        write.writerows(csv_array)

    fig, ax = plt.subplots()
    hm_values = np.array(value_list)
    im, cbar = heatmap(hm_values, name_list, name_list, ax=ax,
                       cmap="YlGn", cbarlabel="Similarity Score")
    fig.tight_layout()
    plt.savefig("../TRANSCRIPTS/similarity_score.png")


def axis_name():

    def if_tana(item):
        if "Tana Mongeau" in item:
            return False
        else:
            return True

    with open('../TRANSCRIPTS/ratio_list.json') as ratio_file:
        ratio_list = json.loads(ratio_file.read())
        from_ratio = list(filter(if_tana, sorted([ratio['name'] for ratio in ratio_list])))
        print(from_ratio)

        with_ratio = []
        for ratio in ratio_list:
            if ratio['ratio'] is not None:
                with_ratio.append('{name} ({ratio:.2f})'.format(name=ratio['name'],
                                                                ratio=ratio['ratio']))
            else:
                with_ratio.append('{name} (None)'.format(name=ratio['name'], ratio=ratio['ratio']))

        sorted_ratio = sorted(with_ratio)

        filtered_ratio = list(filter(if_tana, sorted_ratio))

        print(with_ratio)
        print(sorted_ratio)
        print(filtered_ratio)
        return [from_ratio, filtered_ratio]


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


def create_ratio_csv():
    with open('../TRANSCRIPTS/ratio_list.json') as ratio_file:
        ratio_list = json.loads(ratio_file.read())
        ratio_list = sorted(ratio_list, key=lambda k: k['ratio'])
        labels = ["names", "likes", "dislikes", "ratio"]
        data = [[ratio['name'], ratio['like'], ratio['dislikes'], ratio['ratio']] for ratio in ratio_list]

    with open('../TRANSCRIPTS/ratios.csv', 'w') as h_values:
        write = csv.writer(h_values)
        write.writerow(labels)
        write.writerows(data)


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
        plt.savefig("../TRANSCRIPTS/ratio_graph.png")


if __name__ == "__main__":  # run the script
    main()
