########################################################################################################################
# This file is for Chapter 3.8 Climate Change.
# Note: KP 1992 has modified, see "KP1992_7Oct"
# Prepared by Arthur, RA. 20211027
########################################################################################################################
# TODO change data address
# Chapter 3.8.3.2 Rainfall
########################################################################################################################
from modules import *  # functions and global variables

def run_rf():
    def select_data(data, period):
        start_yr, end_yr = period
        _ = data.columns[0]
        # 2 layer selections
        s1_data = data[np.array(data[_], dtype=int) >= start_yr]
        f_data = s1_data[np.array(s1_data[_], dtype=int) <= end_yr]
        return f_data

    # CN font
    plt.rcParams['font.family'] = cn_font
    p = ['1951-1980', '1991-2020']
    x_l = ['一月', '二月', '三月', '四月', '五月', '六月', '七月', '八月', '九月', '十月', '十一月', '十二月']

    # read data
    data = pd.read_csv('Data\\rf_mo.csv')
    cols = data.columns

    # compute mean values for two periods
    period = (1951, 1980)
    df1 = select_data(data, period)
    period = (1991, 2020)
    df2 = select_data(data, period)
    col_ = cols[np.logical_not(cols == 'Year')]
    m1 = df1[col_].mean(axis=0)
    print(f'Yearly mean of 1951-1980: {m1.sum():.1f} mm')
    m2 = df2[col_].mean(axis=0)
    print(f'Yearly mean of 1991-2020: {m2.sum():.1f} mm')
    mean_ = pd.concat([m1, m2], axis=1)
    mean_.columns = p
    mean_.index = x_l
    # prepare difference for plotting
    diff_ = pd.DataFrame(np.array(m2) - np.array(m1), index=mean_.index, columns=['兩段時期之降雨差'])

    # bar plot
    fig, ax = plt.subplots()
    mean_.plot.bar(y=p, ax=ax, xlabel='月份', ylabel=f'月總降雨 [毫米]', figsize=(fwidth, fheight))
    ax2 = ax.twinx()
    diff_.plot(ax=ax2, legend=False, c='k', marker=marker, ylabel='差距 [毫米]')
    ax2.legend(loc='upper left')
    fig.tight_layout()

    # compute and show yearly rate and seasonal rate
    yrl_rf = pd.read_csv('Data\\HKO_full.csv')['rf']
    year = pd.read_csv('Data\\HKO_full.csv')['Year']
    print('For RF in 1884-2020: ')
    print_lm_coef(yrl_rf, year, 'RF', ignore_na=False, scale=1, data_len=True)
    print('For RF in 1951-2020: ')
    print_lm_coef(yrl_rf[-70:], year[-70:], 'RF', ignore_na=False, scale=1, data_len=True)
    # seasonal: 1951-2020
    data = pd.read_csv('Data\\rf_mo.csv')
    data = data[data['Year'] >= 1951]
    year = data['Year'][data['Year'] >= 1951]
    for i,sea in enumerate(seasons):
        sea_rf = data[sea].sum(axis=1)
        print_lm_coef(sea_rf, year, seasons_name[i], ignore_na=True, scale=1, data_len=True, plot_data=True)


def run_computation_rf():
    def compute_periodic_mean(data, ref_period, comp_period, col):
        dt_ref = data[(data['Year'] >= ref_period[0]) & (data['Year'] <= ref_period[1])]
        dt_cop = data[(data['Year'] >= comp_period[0]) & (data['Year'] <= comp_period[1])]
        size = (ref_period[1]-ref_period[0]+1, comp_period[1]-comp_period[0]+1)
        print(f"Avg yearly {col} in {ref_period} is: {np.mean(dt_ref[col])}, SD={np.std(dt_ref[col])}， SE={np.std(dt_ref[col])/np.sqrt(size[0])}")
        print(f"Avg yearly {col} in {comp_period} is: {np.mean(dt_cop[col])}, SD={np.std(dt_cop[col])}, SE={np.std(dt_cop[col])/np.sqrt(size[1])}")

    def compute_consecutive_rf(n):
        start, end = 1884, 2020
        year = np.linspace(start, end, end - start + 1, dtype=int)
        months = month
        h3sd = np.zeros(len(year))
        h3sd[:] = np.nan
        for i in tqdm(range(len(year))):
            if 1940 <= year[i] <= 1946:
                pass
            else:
                # all year data subtracted from HKO use simple web scrapper
                dt_ = pd.read_csv(f'Data\\rainfall\\{year[i]}.csv')
                assert np.all(months == dt_.columns[2:])
                select_dt = dt_[months]
                arr_1d = select_dt.values.ravel('F')
                # drop na
                arr_1d = arr_1d[~np.isnan(arr_1d)]
                # use convolution: n ones
                arr_consec = np.convolve(arr_1d, np.ones(n), mode='valid')
                # find largest one
                h3sd[i] = np.max(arr_consec)
        return h3sd

    def compute_highest_days():
        # prepare required data
        start, end = 1884, 2020
        year = np.linspace(start, end, end - start + 1, dtype=int)
        months = month
        h3_arr = np.zeros(len(year))  # highest 3 days
        h1_arr = np.zeros(len(year))  # highest 1 day
        n = [0.1, 1, 10, 30]  # rf threshold
        larger_n_d = np.zeros((len(n), len(year)))  # means "larger than n (mm) in days"

        # below loops all daily rf data, obtained by using simple web scrapper
        for i in tqdm(range(len(year))):
            # all year data subtracted from HKO use simple web scrapper
            dt_ = pd.read_csv(f'Data\\rainfall\\{year[i]}.csv')
            assert np.all(months == dt_.columns[2:])
            select_dt = dt_[months]
            # find largest 3 days: array
            h3 = find_largest(data=select_dt, n=3)
            # find largest 1 day:
            h1 = find_largest(data=select_dt, n=1)
            # calculate
            if np.sum(h3) == 0:
                h3_arr[i] = np.nan
                h1_arr[i] = np.nan
            else:
                h3_arr[i] = np.sum(h3)
                h1_arr[i] = h1
            # count larger than n days
            for j, y in enumerate(larger_n_d):
                thres = n[j]
                # col==year
                if count_larger(data=select_dt, n=thres) == 0 and 1940 <= year[i] <= 1946:
                    larger_n_d[j, i] = np.nan
                else:
                    larger_n_d[j, i] = count_larger(data=select_dt, n=thres)
        # prepare hightest 1 hour:
        dt_ = pd.read_csv('Data\\rf_max1hr.csv')  # TODO
        assert np.all(year == dt_['Year'])
        h1h_arr = dt_['Total']
        # stack
        h_ = np.vstack((h3_arr, h1_arr, h1h_arr))

        ## Store data:
        # new_dt = np.vstack((year,h_))
        # data = pd.DataFrame(new_dt.T, columns=['Year', 'max_3_day', 'max_1_day','max_1_hr'])
        # data.to_csv('Data\\rf_max_3-1day_1hr.csv',index=False)

        return larger_n_d, h_

    # 1. avg days with rf>=10mm
    # 2. avg days with rf>=30mm
    # 3. avg max daily rf
    # 4. avg max 3 consecutive rf
    yr_rf = pd.read_csv('Data\\HKO.csv')[['Year', 'rf']]
    mo_rf = pd.read_csv('Data\\rf_mo.csv')
    year = np.array(mo_rf['Year'], dtype=int)

    # prepare data: rf>=n mm and 3 consecutive days
    larger_n_d, h_ = compute_highest_days()
    h3sd = compute_consecutive_rf(n=3)

    # for rf>=n mm, stored in one dataframe
    new_dt = np.vstack((year, larger_n_d))
    data = pd.DataFrame(new_dt.T, columns=['Year', '>=0.1mm', '>=1mm', '>=10mm', '>=30mm'])

    # set periods
    ref_period = (1951, 1980)
    comp_period = (1991, 2020)
    # periodic mean for rf yrly total rf, rf>=10mm, rf>=30mm
    compute_periodic_mean(yr_rf, ref_period, comp_period, 'rf')
    compute_periodic_mean(data, ref_period, comp_period, '>=10mm')
    compute_periodic_mean(data, ref_period, comp_period, '>=30mm')
    # periodic mean for max 1 day rf:
    h1_arr = h_[1]  # see function compute_highest_day definition of h_
    new_dt = np.vstack((year, h1_arr))
    data = pd.DataFrame(new_dt.T, columns=['Year', 'max_1_day'])
    compute_periodic_mean(data, ref_period, comp_period, 'max_1_day')

    # for 3 consecutive days
    new_dt = np.vstack((year, h3sd))
    data = pd.DataFrame(new_dt.T, columns=['Year', 'max_3_consec_day'])
    # compute
    compute_periodic_mean(data, ref_period, comp_period, 'max_3_consec_day')


if __name__ == "__main__":
    start_time = time.time()
    run_rf()
    run_computation_rf()
    end_of_code(start_time, show_plots=True)

