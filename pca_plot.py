
from pca_cumulative_explained import pca_cumulative_explained
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from sklearn.decomposition import PCA


def pca_plot(data, n_comp=0, threshold=0.85):
    from pca_cumulative_explained import pca_cumulative_explained
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    X = data
    n_features = X.shape[1]
    if (threshold > 1.00) or (threshold < 0.00):
        raise ThresholdValueError(
            "Threshold value must be between 0.00 and 1.00")
    else:
        thresh = threshold

    title_msg = "Explained variances of Principal components"
    if (n_comp > 20) or (n_features > 20):
        n_comp = 20
        title_msg = "Explained variances of Principal components (only top 20 shown)"
    elif n_comp == 0:
        n_comp = n_features

    myPCA = PCA(n_components=n_comp)
    pcs = myPCA.fit_transform(X)

    evr = myPCA.explained_variance_ratio_
    cumul = pca_cumulative_explained(myPCA)

    x_axis = ["PC{0}".format(i+1) for i in range(n_comp)]

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar(x_axis, evr)
    plt.xlabel("Components")
    ax.set_ylabel("PC explained variance (bar)")
    ax.grid(False)

    for x, y in zip(x_axis, evr):

        label = "{:.3f}".format(y)

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 5),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

    ax2 = ax.twinx()
    ax2.plot(x_axis, cumul, "ro--")
    ax2.grid(False)
    ax2.axhline(y=thresh, color='k', linestyle='--')

    plt.annotate("Threshold = {0}".format(thresh),  # this is the text
                 # these are the coordinates to position the label
                 (x_axis[-1], thresh),
                 textcoords="offset points",  # how to position the text
                 xytext=(2, 10),  # distance from text to points (x,y)
                 ha='center',
                 color="k")  # horizontal alignment can be left, right or center
    ax2.set_ylabel("Cumulative Explained variance (Line)")

    for x, y in zip(x_axis, cumul):

        label = "{:.2f}".format(y)

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(2, 10),  # distance from text to points (x,y)
                     ha='center',
                     color="red")  # horizontal alignment can be left, right or center

    plt.title(title_msg)
    plt.show()


class ThresholdValueError(Exception):
    pass
