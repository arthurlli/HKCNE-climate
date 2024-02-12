########################################################################################################################
# This file is for Chapter 3.8 Climate Change.
# Prepared by Arthur, RA. 20211027
########################################################################################################################

# Chapter 3.8.2.1 Temperature
########################################################################################################################
from modules import *  # functions and global variables


def run_avg_temp(ignore_84=True):
    # For reading data: 1940-1946 == NA
    plt.rcParams['font.family'] = cn_font  # set Chinese font
    HKO = pd.read_csv('Data\\HKO.csv')
    # choose starting from 1884 or 1885
    if ignore_84:
        t_start_ind = 1
        t_start_year = 1885
    else:
        t_start_ind = 0
        s_start_year = 1884

    # read data: only use 1885-2020
    year = np.array(HKO['Year'])[t_start_ind:]
    avg_T = HKO['avg_T']
    avg_min_T = HKO['avg_min_T']
    avg_max_T = HKO['avg_max_T']
    temp_mtx = np.vstack((avg_max_T, avg_T, avg_min_T))[:, t_start_ind:]

    # locate start and end of NA: no other NA except 1940-46
    ind1, ind2 = get_nas_ind(temp_mtx[0, :])
    assert year[ind1] == 1940 and year[ind2] == 1946
    # print coefs and LM plots:
    fig, ax = plt.subplots()
    fig.set_size_inches(10, fheight)
    for i, x in enumerate(temp_mtx):
        lm_coefs = lm(lmx=year[:ind1], lmy=x[:ind1])
        ax.plot(year, x, c='k', label=t_name[i], ls=t_ls[i])
        ax.plot(year[:ind1], lm_coefs[0], c='blue')
        lm_coefs = lm(lmx=year[ind2 + 1:], lmy=x[ind2 + 1:])
        ax.plot(year[ind2 + 1:], lm_coefs[0], c='blue', label='Linear regression line' if i == 2 else None)
    # add vertical lines
    ax.axvline(year[ind1], c='k')
    ax.axvline(year[ind2], c='k')
    ax.set_ylabel(r'溫度 [$\degree$C]')
    plt.show()
    fig.tight_layout()
    # fig.savefig('plots\\T_LMs.jpg',dpi=300)
    plt.rcParams['font.family'] = en_font

    # compute and show 1885-2020: except NAs
    print("Ignore 1940-46...")
    print_lm_coef(temp_mtx, year, t_name[:-1], ignore_na=True, scale=100, data_len=True)

    # compute and show 1885-1939 & 1947-2020:
    print("For 1885-1939: ")
    print_lm_coef(temp_mtx[:, :ind1], year[:ind1], t_name[:-1], ignore_na=True, scale=100, data_len=True)
    print("For 1947-2020: ")
    print_lm_coef(temp_mtx[:, ind2 + 1:], year[ind2 + 1:], t_name[:-1], ignore_na=True, scale=100, data_len=True)


def run_avg_temp_new(ignore_84=True):
    # For reading data: 1940-1946 == NA
    plt.rcParams['font.family'] = cn_font  # set Chinese font
    HKO = pd.read_csv('Data\\HKO.csv')
    # choose starting from 1884 or 1885
    if ignore_84:
        t_start_ind = 1
        t_start_year = 1885
    else:
        t_start_ind = 0
        s_start_year = 1884

    # read data: only use 1885-2020
    year = np.array(HKO['Year'])[t_start_ind:]
    avg_T = HKO['avg_T']
    avg_min_T = HKO['avg_min_T']
    avg_max_T = HKO['avg_max_T']
    temp_mtx = np.vstack((avg_max_T, avg_T, avg_min_T))[:, t_start_ind:]
    # ignore NAs
    year_noNA = year[~np.isnan(temp_mtx[0, :])]

    # print coefs and LM plots:
    fig, ax = plt.subplots()
    fig.set_size_inches(10, fheight)
    for i, x in enumerate(temp_mtx):
        lm_coefs = lm(lmx=year_noNA, lmy=x[~np.isnan(x)])
        insert_arr = lm_coefs[1][1] + lm_coefs[1][2] * year[np.isnan(x)]
        # insert NA back to the lm arrays
        lm_values = np.insert(lm_coefs[0], np.where(year == 1940)[0][0], insert_arr)
        # plot lm
        ax.plot(year, lm_values, c='blue')
        ax.plot(year, temp_mtx[i, :], c='k', label=t_name[i], ls=t_ls[i])
    ax.set_ylabel(r'溫度 [$\degree$C]')
    plt.show()
    fig.tight_layout()
    # fig.savefig('plots\\T_LMs_20220409.jpg',dpi=300)
    plt.rcParams['font.family'] = en_font

    # compute and show 1885-2020: except NAs
    print("Ignore 1940-46...")
    print_lm_coef(temp_mtx, year, t_name[:-1], ignore_na=True, scale=100, data_len=True)


def run_seasonal(ignore_84=True):
    plt.rcParams['font.family'] = cn_font  # set Chinese font
    if ignore_84:
        t_start_ind = 1
        t_start_year = 1885
    else:
        t_start_ind = 0
        s_start_year = 1884
    # next, plot seasonal temperature
    tp_name = ['Tmax', 'Tavg', 'Tmin']
    year = np.linspace(1884, 2020, 137, dtype=int)
    # 4 seasons; 3 types: max, avg, min; each year
    t_mo_mtx = np.zeros((4, 3, len(year)))
    for i, sea in enumerate(t_mo_mtx):
        for j, type in enumerate(sea):
            file_name = tp_name[j] + '_mo.csv'
            dt = pd.read_csv('Data\\' + file_name)
            for k, ele in enumerate(type):
                temp = dt[seasons[i]].iloc[k]
                if np.sum(np.isnan(temp)) >= 2:
                    type[k] = np.nan
                else:
                    type[k] = np.nanmean(temp)
    # select data: 1885-2020
    year = year[t_start_ind:]
    t_mo_mtx = t_mo_mtx[:, :, t_start_ind:]

    # plot figures and print rates
    plt.rcParams['font.size'] = 40
    for i, dt in enumerate(t_mo_mtx):
        # do data smoothing
        tp, lm_mtx, lm_coef, pl_mtx, pl_coef, pl_cov, ma_mtx = data_smoothing(dt, year, deg, m, True)
        # plot data and smoothing lines: change param "show_params" to see other lines
        plot_smoothed_data(x=year, y=dt, data_label=t_name, col=t_col, ls=t_ls, lm_mtx=lm_mtx, pl_mtx=pl_mtx,
                           ma_mtx=ma_mtx, deg=deg, m=m, fig_name=f'annual_T_' + seasons_name[i],
                           title=f'{seasons_name[i]}', xlabel='Year',
                           ylabel=r'溫度 [$\degree$C]', show_params=['ylabel', 'lm'], fwidth=9, fheight=6, savefig=False)
        plt.annotate(f'{seasons_name_cn[i]}', xy=(0.05, 0.9), xycoords='axes fraction', c='k', size=30)
    plt.rcParams['font.size'] = 20

    # show coefficients
    for i, dt in enumerate(t_mo_mtx):
        print(f"For {seasons_name[i]}:")
        print_lm_coef(dt, year, t_name, ignore_na=True, scale=100, data_len=True)
        print("##################################################################################")


if __name__ == "__main__":
    start_time = time.time()
    run_avg_temp()
    run_seasonal()
    end_of_code(start_time, show_plots=True)
