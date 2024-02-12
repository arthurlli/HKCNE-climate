########################################################################################################################
# This file is for Chapter 3.8 Climate Change.
# Note: GT are measured at 0700, TKL 2006 removed due to part of monthly data missing.
# Prepared by Arthur, RA. 20211028
########################################################################################################################

# Chapter 3.8.3.4 Grass temperature (7 a.m.)
########################################################################################################################
from modules import *  # functions and global variables

def run_main():
    plt.rcParams['font.family'] = cn_font

    # read data:
    gt = pd.read_csv('Data\\grass_T_yr.csv')
    year = gt['yyyy']

    # remove TKL 2006 data: extreme value due to missing data
    gt['TKL'][year==2006] = np.nan
    # select 1971-2020
    gt = gt[year >= 1971]
    year = year[year >= 1971]

    names = ['HKO', 'KP', 'KSC', 'TKL', 'TMS']
    data = gt[names].T
    cn_names = ['天文台', '京士柏', '滘西洲', '打鼓嶺', '大帽山']

    # plot figure
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    deg, m = 3, 5
    ls = ['-', '-', '--', '-.']
    for i, dt in enumerate(np.array(data)):
        c = next(ax._get_lines.prop_cycler)['color']
        ax.plot(year, dt, c=c, label=cn_names[i])
        tp, lm_mtx, lm_coef, pl_mtx, pl_coef, pl_cov, ma_mtx = data_smoothing(dt, year, deg, m, True)
        start = len(lm_mtx)  # start point if nan exists
        ax.plot(year[-start:], lm_mtx, c=c, ls=ls[1])
    # set params
    ax.set_ylabel(r'溫度 [$\degree$C]')
    ax.legend()
    fig.tight_layout()
    # fig.savefig('plots\\1971至2020年香港各地草地最低溫度趨勢.jpg', dpi=300)

    # compute and show rates
    print_lm_coef(gt[names].T, year, names, scale=100, data_len=True, ignore_na=True)


if __name__ == "__main__":
    start_time = time.time()
    run_main()
    end_of_code(start_time, show_plots=True)


