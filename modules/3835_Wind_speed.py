########################################################################################################################
# This file is for Chapter 3.8 Climate Change.
# Note:
# Prepared by Arthur, RA. 20211028
########################################################################################################################

# Chapter 3.8.3.3 Section 5 - Wind speed
########################################################################################################################
from modules import *  # functions and global variables

def run_main():
    plt.rcParams['font.family'] = cn_font

    # read data
    ws = pd.read_csv('Data\\WGL_KP_wind_1968-2020.csv')
    year = ws['Year']
    wkp = ws['KP']
    wwgl = ws['WGL']

    # select only 1971-2020
    start_yr = 1971
    wkp, wwgl = wkp[year >= start_yr], wwgl[year >= start_yr]
    year = year[year >= start_yr]

    # plot data
    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(fwidth * 0.8, fheight)
    c = 'black'
    # plot before 1999
    ind = np.where(year == 1999)[0][0] + 1
    ax[0].plot(year[:ind], wwgl[:ind], ls='-', label='橫瀾島', c=c)
    lm_ = lm(year[:ind], wwgl[:ind])[0]
    ax[0].plot(year[:ind], lm_, c=c, ls='-')
    # plot after 1999
    ax[0].plot(year[ind:], wwgl[ind:], ls='-', c=c)
    lm_ = lm(year[ind:], wwgl[ind:])[0]
    ax[0].plot(year[ind:], lm_, c=c, ls='-')
    ax[0].set_ylabel('風速 [米/秒]')
    ax[0].set_ylim(get_limit(wwgl, 0.85))
    ax[0].legend(loc='upper left')

    # plt KP
    stop = 1996
    ind = np.where(year == stop)[0][0]
    c = 'b'  # color for KP
    ax[1].plot(year[:ind], wkp[:ind], ls='-', label='京士柏', c=c)
    tp, lm_mtx, lm_coef, pl_mtx, pl_coef, pl_cov, ma_mtx = data_smoothing(wkp[:ind], year[:ind], deg, m, True)
    ax[1].plot(year[:ind], lm_mtx, c=c, ls='-')
    # second part
    ax[1].plot(year[ind:], wkp[ind:], ls='-', c=c)
    tp, lm_mtx, lm_coef, pl_mtx, pl_coef, pl_cov, ma_mtx = data_smoothing(wkp[ind:], year[ind:], deg, m, True)
    ax[1].plot(year[ind:], lm_mtx, c=c, ls='-')
    ax[1].set_ylabel('風速 [米/秒]')
    ax[1].set_ylim(get_limit(wkp, 0.85))
    ax[1].legend(loc='upper left')
    fig.tight_layout()
    fig.savefig('plots\\1971-2020京士柏和橫瀾島風速的變化.jpg', dpi=300)

    # compute and show rates: 1971-1993 & 2000*-2020
    period = (1971, 1993)
    year_p1 = year[(year>=period[0]) & (year<=period[1])]
    print('For 1971-1993 WGL:')
    print_lm_coef(wwgl[(year>=period[0]) & (year<=period[1])], year_p1, 'WGL 1st part', data_len=True, ignore_na=True, scale=10)
    print('For 1971-1993 KP:')
    print_lm_coef(wkp[(year>=period[0]) & (year<=period[1])], year_p1, 'KP 1st part', data_len=True, ignore_na=True, scale=10)
    print('###########################################################################################################')
    period = (2000, 2020)
    year_p2 = year[(year>=period[0]) & (year<=period[1])]
    print('For 2000-2020 WGL:')
    print_lm_coef(wwgl[(year>=period[0]) & (year<=period[1])], year_p2, 'WGL 2nd part', data_len=True, ignore_na=True, scale=10)
    print('For 2000-2020 KP:')
    print_lm_coef(wkp[(year>=period[0]) & (year<=period[1])], year_p2, 'KP 2nd part', data_len=True, ignore_na=True, scale=10)


if __name__ == "__main__":
    start_time = time.time()
    run_main()
    end_of_code(start_time, show_plots=True)

