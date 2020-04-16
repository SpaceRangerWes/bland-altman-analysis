from collections import namedtuple

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pathlib import Path
from pyspark.sql.functions import monotonically_increasing_id
from scipy import stats

import numpy as np


spark = (SparkSession.builder.config("spark.sql.session.timeZone",
                                     "UTC").master("local").getOrCreate())

LIMIT_OF_AGREEMENT = 1.96


def _bland_altman_data_eval(data1, data2):
    if data1.size != data2.size:
        raise Exception("Data arrays need to be equal in length")
    else:
        return 0


BlandAltmanStat = namedtuple('BlandAltmanStat',
                             ['diff', 'mean', 'mean_diff', 'std_dev', 'confidence_intervals', 'mean_diff_perc', 'std_dev_perc'])


def calculate_confidence_intervals(md, sd, n, confidence_interval):

    confidence_intervals = dict()

    confidence_interval = confidence_interval / 100.

    confidence_intervals['mean'] = stats.norm.interval(confidence_interval, loc=md,
                                                      scale=sd / np.sqrt(n))

    seLoA = ((1 / n) + (LIMIT_OF_AGREEMENT ** 2 / (2 * (n - 1)))) * (sd ** 2)
    loARange = np.sqrt(seLoA) * stats.t._ppf((1 - confidence_interval) / 2., n - 1)

    confidence_intervals['upperLoA'] = ((md + LIMIT_OF_AGREEMENT * sd) + loARange,
                                       (md + LIMIT_OF_AGREEMENT * sd) - loARange)

    confidence_intervals['lowerLoA'] = ((md - LIMIT_OF_AGREEMENT * sd) + loARange,
                                       (md - LIMIT_OF_AGREEMENT * sd) - loARange)


    return confidence_intervals


def bland_altman(data1, data2):
    _bland_altman_data_eval(data1, data2)

    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    # perc_diff = (diff / mean) * 100
    perc_diff = None

    mean_diff = np.mean(diff)
    # mean_diff_perc = np.mean(perc_diff)
    mean_diff_perc = None

    std_dev = np.std(diff, axis=0)
    # std_dev_perc = np.std(perc_diff, axis=0)
    std_dev_perc = None

    ci = calculate_confidence_intervals(md=mean_diff, sd=std_dev, n=len(diff), confidence_interval=90)

    return BlandAltmanStat(diff=diff, mean=mean, mean_diff=mean_diff, std_dev=std_dev, confidence_intervals=ci,
                           mean_diff_perc=mean_diff_perc, std_dev_perc=std_dev_perc)


def rangeFrameLocator(tickLocs, axisRange):
    """
    Convert axis tick positions for a Tufte style range frame. Takes existing tick locations, places a tick at the min and max of the data, and drops existing ticks that fall outside of this range or too close to the margins.
    TODO: Convert to a true axis artist that also sets spines
    :param list tickLocs: List of current tick locations on the axis
    :param tuple axisRange: Tuple of (min, max) value on the axis
    :returns: List of tick locations
    :rtype: list
    """
    newTicks = [axisRange[0]]

    for tick in tickLocs:
        if tick <= axisRange[0]:
            pass
        elif tick >= axisRange[1]:
            pass
        else:
            newTicks.append(tick)

    newTicks.append(axisRange[1])

    return newTicks


def rangeFrameLabler(tickLocs, tickLabels, cadence):
    """
    Takes lists of tick positions and labels and drops the marginal text label where the gap between ticks is less than half the cadence value

    :param list tickLocs: List of current tick locations on the axis
    :param list tickLabels: List of tick labels
    :param float cadence: Gap between major tick positions
    :returns: List of tick labels
    :rtype: list
    """
    labels = []

    for i, tick in enumerate(tickLocs):
        if tick == tickLocs[0]:
            labels.append(tickLabels[i])

        elif tick == tickLocs[-1]:
            labels.append(tickLabels[i])

        elif (tick < (tickLocs[0] + (cadence / 2.0))) & (tick < (tickLocs[0] + cadence)):
            labels.append('')

        elif (tick > (tickLocs[-1] - (cadence / 2.0))) & (tick > (tickLocs[-1] - cadence)):
            labels.append('')

        else:
            labels.append(tickLabels[i])

    return labels

def _drawBlandAltman(ba_stats: BlandAltmanStat):
    import matplotlib
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
    from matplotlib import transforms
    from matplotlib import ticker

    figureSize = (10, 7)
    dpi = 72
    # savePath = None
    # figureFormat = 'png'
    meanColour = '#6495ED'
    loaColour = 'coral'
    pointColour = '#6495ED'
    title = None

    fig, ax = plt.subplots(figsize=figureSize, dpi=dpi)

    if 'mean' in ba_stats.confidence_intervals.keys():
        ax.axhspan(ba_stats.confidence_intervals['mean'][0],
                   ba_stats.confidence_intervals['mean'][1],
                   facecolor=meanColour, alpha=0.2)

    if 'upperLoA' in ba_stats.confidence_intervals.keys():
        ax.axhspan(ba_stats.confidence_intervals['upperLoA'][0],
                   ba_stats.confidence_intervals['upperLoA'][1],
                   facecolor=loaColour, alpha=0.2)

    if 'lowerLoA' in ba_stats.confidence_intervals.keys():
        ax.axhspan(ba_stats.confidence_intervals['lowerLoA'][0],
                   ba_stats.confidence_intervals['lowerLoA'][1],
                   facecolor=loaColour, alpha=0.2)

    ##
    # Plot the mean diff and LoA
    ##
    ax.axhline(ba_stats.mean_diff, color=meanColour, linestyle='--')
    ax.axhline(ba_stats.mean_diff + LIMIT_OF_AGREEMENT*ba_stats.std_dev, color=loaColour, linestyle='--')
    ax.axhline(ba_stats.mean_diff - LIMIT_OF_AGREEMENT*ba_stats.std_dev, color=loaColour, linestyle='--')

    ##
    # Plot the data points
    ##
    ax.scatter(ba_stats.mean, ba_stats.diff, alpha=0.5, c=pointColour)

    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    limit_of_agreement_range = (ba_stats.mean_diff + (LIMIT_OF_AGREEMENT * ba_stats.std_dev)) - (ba_stats.mean_diff - LIMIT_OF_AGREEMENT*ba_stats.std_dev)
    offset = (limit_of_agreement_range / 100.0) * 1.5

    # ax.text(0.98, ba_stats.mean_diff + offset, 'Mean', ha="right", va="bottom", transform=trans)
    # ax.text(0.98, ba_stats.mean_diff - offset, f'{ba_stats.mean_diff:.2f}', ha="right", va="top", transform=trans)
    #
    # ax.text(0.98, ba_stats.mean_diff + (LIMIT_OF_AGREEMENT * ba_stats.std_dev) + offset,
    #         '+{LIMIT_OF_AGREEMENT:.2f} SD', ha="right", va="bottom", transform=trans)
    # ax.text(0.98, ba_stats.mean_diff + (LIMIT_OF_AGREEMENT * ba_stats.std_dev) - offset,
    #         '{ba_stats.mean_diff + LIMIT_OF_AGREEMENT*ba_stats.std_dev:.2f}', ha="right", va="top", transform=trans)
    #
    # ax.text(0.98, ba_stats.mean_diff - (LIMIT_OF_AGREEMENT * ba_stats.std_dev) - offset,
    #         '-{LIMIT_OF_AGREEMENT:.2f} SD', ha="right", va="top", transform=trans)
    # ax.text(0.98, ba_stats.mean_diff - (LIMIT_OF_AGREEMENT * ba_stats.std_dev) + offset,
    #         '{ba_stats.mean_diff - LIMIT_OF_AGREEMENT*ba_stats.std_dev:.2f}', ha="right", va="bottom", transform=trans)

    # Only draw spine between extent of the data
    ax.spines['left'].set_bounds(min(ba_stats.diff), max(ba_stats.diff))
    ax.spines['bottom'].set_bounds(min(ba_stats.mean), max(ba_stats.mean))

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylabel('Difference between methods')
    ax.set_xlabel('Mean of methods')

    tickLocs = ax.xaxis.get_ticklocs()
    cadenceX = tickLocs[2] - tickLocs[1]
    tickLocs = rangeFrameLocator(tickLocs, (min(ba_stats.mean), max(ba_stats.mean)))
    ax.xaxis.set_major_locator(ticker.FixedLocator(tickLocs))

    tickLocs = ax.yaxis.get_ticklocs()
    cadenceY = tickLocs[2] - tickLocs[1]
    tickLocs = rangeFrameLocator(tickLocs, (min(ba_stats.diff), max(ba_stats.diff)))
    ax.yaxis.set_major_locator(ticker.FixedLocator(tickLocs))

    plt.draw() # Force drawing to populate tick labels

    labels = rangeFrameLabler(ax.xaxis.get_ticklocs(), [item.get_text() for item in ax.get_xticklabels()], cadenceX)
    ax.set_xticklabels(labels)

    labels = rangeFrameLabler(ax.yaxis.get_ticklocs(), [item.get_text() for item in ax.get_yticklabels()], cadenceY)
    ax.set_yticklabels(labels)

    ax.patch.set_alpha(0)

    if title:
        ax.set_title(title)

    # plt.show()
    plt.savefig("mygraph4.png")


if __name__ == '__main__':
    dir_path = Path(__file__).parent.parent
    resources_path = Path(dir_path, "resources")
    test_data_path = str(Path(resources_path, "input_test4.csv"))

    df = (spark.read.csv(path=test_data_path, inferSchema=True, header=True)
          .withColumn("id", monotonically_increasing_id())
          .drop("StoreNumber", "BusinessDate")
          )
    df.show()

    def data_transform(inner_df: DataFrame):
        dtype = [("id", int), ("value", float)]
        data = np.array(inner_df.collect(), dtype=dtype)
        np.sort(data, order='id')
        return data['value']

    data1 = data_transform(df.select("id", "LeftTotalItemSaleDiscountAmount"))
    data2 = data_transform(df.select("id", "RightTotalItemSaleDiscountAmount"))

    correlation = stats.pearsonr(data1, data2)
    print(correlation)

    ba_stat = bland_altman(data1, data2)
    _drawBlandAltman(ba_stat)


