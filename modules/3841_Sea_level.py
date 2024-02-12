########################################################################################################################
# This file is for Chapter 3.8 Climate Change.
# Note:
# Prepared by Arthur, RA. 20211028
########################################################################################################################

# Chapter 3.8.4.1 Sea level
########################################################################################################################
from modules import *  # functions and global variables

def run_main():
    plt.rcParams['font.family'] = cn_font

    # read data
    data = pd.read_csv('Data\\HKO_20211016.csv')
    sl = data[['Year', 'corrected_sl']]
    year = sl['Year']

    # selection: 1954-2020
    start_yr = 1954
    sl = sl[sl['Year'] >= start_yr]
    year = year[year>= start_yr]

    # plot figure
    fig = plt.figure()
    fig.set_size_inches(fwidth *0.7, fheight*0.8)
    plt.grid()
    plt.plot(year, sl['corrected_sl'], c='k', marker=marker)
    lm_ = lm(lmx=year, lmy=sl['corrected_sl'])
    # plt.plot(year, lm_[0], c='b')  # 20220411: remove regression line
    plt.ylim((1.0, 1.8))
    plt.ylabel('海圖基準面以上的潮水高度 [米]')
    fig.tight_layout()
    fig.savefig('plots\\1954至2020維多利亞港平均海平面高度（海圖基準面以上）.jpg', dpi=300)


if __name__ == "__main__":
    start_time = time.time()
    run_main()
    end_of_code(start_time, show_plots=True)

