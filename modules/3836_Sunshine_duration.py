########################################################################################################################
# This file is for Chapter 3.8 Climate Change.
# Note:
# Prepared by Arthur, RA. 20211028
########################################################################################################################

# Chapter 3.8.3.3 Section 7 - Sunshine duration
########################################################################################################################
from modules import *  # functions and global variables

def run_main():
    plt.rcParams['font.family'] = cn_font

    # read data
    sdu = pd.read_csv('Data\\Sdu_KP.csv')
    year = sdu['Year']

    ## Uncomment for more details
    # ls = ['-', '-', '--', '-.']
    # deg, m = 3, 5
    # tp, lm_mtx, lm_coef, pl_mtx, pl_coef, pl_cov, ma_mtx = data_smoothing(sdu['Sunshine_hr'], year, deg, m, True)
    # plot_smoothed_data(year, sdu['Sunshine_hr'], 'Measurement in KP', 'black', ls, lm_mtx,
    #                    pl_mtx, ma_mtx, deg, m, 'Sdu_KP.jpg', 'Year', 'Total bright sunshine [hr]', outside_legend=False)

    # plot data: only data
    fig = plt.figure()
    fig.set_size_inches(fwidth * 0.8, fheight * 0.8)
    plt.plot(year, sdu['Sunshine_hr'], c='k', label='京士柏')
    plt.ylabel('年總日照時數 [小時]')
    plt.legend()
    fig.tight_layout()
    fig.savefig('plots\\1961至2020京士柏年總日照時數.jpg', dpi=300)


if __name__ == "__main__":
    start_time = time.time()
    run_main()
    end_of_code(start_time, show_plots=True)
