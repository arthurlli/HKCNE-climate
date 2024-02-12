########################################################################################################################
# This file is for Chapter 3.8 Climate Change.
# Note:
# Prepared by Arthur, RA. 20211103
########################################################################################################################

# Chapter 3.8.3.10 Extreme weather
########################################################################################################################
from modules import *  # functions and global variables


def run_hot_nights():
    plt.rcParams['font.family'] = cn_font

    # read data
    data = pd.read_csv('Data\\HKO_full.csv')
    hns = data[['Year', 'hot_nights']]
    # select after 1946
    hns = hns[hns['Year'] >= 1947]

    # plot figure
    fig = plt.figure()
    fig.set_size_inches(fwidth * 0.8, fheight * 0.8)
    plt.plot(hns['Year'], hns['hot_nights'], c='k')
    plt.ylabel('日數')
    fig.savefig('plots\\1947-2020天文台每年熱夜天數.jpg', dpi=300)


def run_hot_days():
    plt.rcParams['font.family'] = cn_font

    # read data
    data = pd.read_csv('Data\\HKO_full.csv')
    hds = data[['Year', 'hot_days']]
    # select from 1947
    hds = hds[hds['Year'] >= 1947]

    # plot figure
    fig = plt.figure()
    fig.set_size_inches(fwidth * 0.8, fheight * 0.8)
    plt.plot(hds['Year'], hds['hot_days'], c='k')
    plt.ylabel('日數')
    fig.savefig('plots\\1947-2020天文台每年酷熱天氣天數.jpg', dpi=300)


def run_cold_days():
    plt.rcParams['font.family'] = cn_font

    # read data
    data = pd.read_csv('Data\\HKO_full.csv')
    cds = data[['Year', 'cold_days']]
    # select from 1947
    cds = cds[cds['Year'] >= 1947]

    # plot figure
    fig = plt.figure()
    fig.set_size_inches(fwidth * 0.8, fheight * 0.8)
    plt.plot(cds['Year'], cds['cold_days'], c='k')
    plt.ylabel('日數')
    fig.savefig('plots\\1947-2020天文台每年寒冷天氣天數.jpg', dpi=300)


if __name__ == "__main__":
    start_time = time.time()
    run_hot_nights()
    run_hot_days()
    run_cold_days()
    end_of_code(start_time, show_plots=True)
