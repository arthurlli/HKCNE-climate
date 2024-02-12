########################################################################################################################
# This file is for Chapter 3.8 Climate Change.
# Note:
# Prepared by Arthur, RA. 20211028
########################################################################################################################

# Chapter 3.8.3.3 Section 8 - Evaporation
########################################################################################################################
from modules import *  # functions and global variables

def run_main():
    plt.rcParams['font.family'] = cn_font

    # read data
    eva = pd.read_csv('Data\\EP_KP.csv')
    year = eva['Year']

    # plot only data
    fig = plt.figure()
    fig.set_size_inches(fwidth * 0.8, fheight * 0.8)
    plt.plot(year, eva['Evaporation'] / 10, c='k', label='京士柏')
    plt.ylabel('年蒸發量 [毫米]')
    plt.legend()
    fig.tight_layout()
    fig.savefig('plots\\1968至2020京士柏年蒸發量.jpg', dpi=300)

    ## uncomment for plot with smoothed lines
    # ls = ['-', '-', '--', '-.']
    # deg, m = 3, 5
    # tp, lm_mtx, lm_coef, pl_mtx, pl_coef, pl_cov, ma_mtx = data_smoothing(eva['Evaporation'], year, deg, m, True)
    # plot_smoothed_data(year, eva['Evaporation'], 'Measurement in KP', 'black', ls, lm_mtx,
    #                    pl_mtx, ma_mtx, deg, m, 'EP_KP.jpg', 'Year', 'Evaporation [0.1 mm]')


if __name__ == "__main__":
    start_time = time.time()
    run_main()
    end_of_code(start_time, show_plots=True)

