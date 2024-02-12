########################################################################################################################
# This file is for Chapter 3.8 Climate Change.
# Prepared by Arthur, RA. 20211028
########################################################################################################################

# Chapter 3.8.3.3 Wetbulb & Dewpoint temperature and relative humidity
########################################################################################################################
from modules import *  # functions and global variables

def run_main():
    plt.rcParams['font.family'] = cn_font

    # prepare data
    HKO = pd.read_csv('Data\\HKO_20211016.csv')
    list = ['Year', 'avg_wet_b', 'avg_dew', 'avg_rh']
    label_list = ['年份', '濕球溫度', '露點溫度', '相對濕度']
    wet_dew_rh = HKO[list]
    year = HKO['Year']
    # select data from 1951-2020
    start_ = 1951
    wet_dew_rh = wet_dew_rh[wet_dew_rh['Year'] >= start_]
    year = year[year >= start_]

    # plot data

    # plot data: 1st y is wetb and dp, 2nd y is RH
    fig, ax = plt.subplots()
    fig.set_size_inches(fwidth * 0.8, fheight * 0.8)
    for i in range(2):
        y = wet_dew_rh[list[i + 1]]
        c = next(ax._get_lines.prop_cycler)['color']
        ds, lm_mtx, lm_coef, pl_mtx, pl_coef, pl_cov, ma_mtx = data_smoothing(y, year, 3, 5, True)
        ax.plot(year, y, ls='-', label=label_list[i + 1], c=c)
        # ax.plot(year[-len(pl_mtx):], pl_mtx, c=c)
    ax.set_ylabel(r'溫度 [$\degree$C]')
    ax.set_ylim((15.5, 22))
    # start second y-axis
    ax2 = ax.twinx()
    y = wet_dew_rh['avg_rh']
    ds, lm_mtx, lm_coef, pl_mtx, pl_coef, pl_cov, ma_mtx = data_smoothing(y, year, 3, 5, True)
    ax2.plot(year, y, ls='-', label=label_list[3], c='green')
    ax2.set_ylabel('相對濕度 [%]', c='green')
    ax2.tick_params(axis='y', colors='green')
    ax2.set_ylim((72.5, 100))
    ax.legend(loc='upper left')
    ax2.legend(loc='lower right')
    fig.tight_layout()
    # fig.savefig('1951至2020年平均濕球溫度、露點溫度和相對濕度趨勢.jpg',dpi=300)

    # compute 1951-80 & 1991-2020 rate
    ref_period = (1951, 1980)
    com_period = (1991, 2020)
    ref_data = wet_dew_rh[(wet_dew_rh['Year'] >= ref_period[0]) & (wet_dew_rh['Year'] <= ref_period[1])]
    ref_year = year[(wet_dew_rh['Year'] >= ref_period[0]) & (wet_dew_rh['Year'] <= ref_period[1])]
    print(f"For period {ref_period}")
    print_lm_coef(np.array(ref_data['avg_wet_b']).reshape(1, -1), ref_year, ['avg_wet_b'], ignore_na=True, scale=100, data_len=True)
    print_lm_coef(np.array(ref_data['avg_dew']).reshape(1, -1), ref_year, ['avg_dew'], ignore_na=True, scale=100, data_len=True)
    print_lm_coef(np.array(ref_data['avg_rh']).reshape(1, -1), ref_year, ['avg_rh'], ignore_na=True, scale=100, data_len=True)

    com_data = wet_dew_rh[(wet_dew_rh['Year'] >= com_period[0]) & (wet_dew_rh['Year'] <= com_period[1])]
    com_year = year[(wet_dew_rh['Year'] >= com_period[0]) & (wet_dew_rh['Year'] <= com_period[1])]
    print(f"For period {com_period}")
    print_lm_coef(np.array(com_data['avg_wet_b']).reshape(1, -1), com_year, ['avg_wet_b'], ignore_na=True, scale=100, data_len=True)
    print_lm_coef(np.array(com_data['avg_dew']).reshape(1, -1), com_year, ['avg_dew'], ignore_na=True, scale=100, data_len=True)
    print_lm_coef(np.array(com_data['avg_rh']).reshape(1, -1), com_year, ['avg_rh'], ignore_na=True, scale=100, data_len=True)


if __name__ == "__main__":
    start_time = time.time()
    run_main()
    end_of_code(start_time, show_plots=True)
