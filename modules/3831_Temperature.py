########################################################################################################################
# This file is for Chapter 3.8 Climate Change.
# Note: KP 1992 has modified, see "KP1992_7Oct"
# Prepared by Arthur, RA. 20211027
########################################################################################################################

# Chapter 3.8.3.1 Temperature
########################################################################################################################
from modules import *  # functions and global variables


def run_temp():
    def select_data(data, period):
        start_yr, end_yr = period
        _ = data.columns[0]
        # 2 layer selections
        s1_data = data[np.array(data[_], dtype=int) >= start_yr]
        f_data = s1_data[np.array(s1_data[_], dtype=int) <= end_yr]
        return f_data

    def print_avgT_rates(period):
        data = pd.read_csv('Data\\HKO_full.csv')
        avgT = data['avg_T']
        year = data['Year']
        # show rates
        print_lm_coef(avgT[(year>=period[0]) & (year<=period[1])], year[(year>=period[0]) & (year<=period[1])], 'Avg T', ignore_na=True, scale=100, data_len=True)

    # CN font
    plt.rcParams['font.family'] = cn_font
    # variables
    p = ['1951-1980', '1991-2020']
    x_l = ['一月', '二月', '三月', '四月', '五月', '六月', '七月', '八月', '九月', '十月', '十一月', '十二月']

    # show avg T rates in 1947-2020, 1971-2020, and 1991-2020
    print_avgT_rates((1947, 2020))
    print_avgT_rates((1971, 2020))
    print_avgT_rates((1991, 2020))

    # read data
    data = pd.read_csv('Data\\Tavg_mo.csv')
    cols = data.columns
    # select 1951-1980
    period = (1951, 1980)
    df1 = select_data(data, period)
    # select 1991-2020
    period = (1991, 2020)
    df2 = select_data(data, period)

    # bar plot
    col_ = cols[np.logical_not(df1.columns == 'Year')]
    m1 = df1[col_].astype(float).mean(0)
    print(f'Yearly mean of 1951-1980: {m1.mean():.1f} C')
    m2 = df2[col_].astype(float).mean(0)
    print(f'Yearly mean of 1991-2020: {m2.mean():.1f} C')
    mean_ = pd.concat([m1, m2], axis=1)
    # rename mean_ index to CN
    mean_.index = x_l
    diff_ = pd.DataFrame(np.array(m2) - np.array(m1), index=mean_.index, columns=['兩段時期之溫度差'])
    mean_.columns = p
    # bar plot
    fig, ax = plt.subplots()
    mean_.plot.bar(y=p, ax=ax, xlabel='月份', ylabel=f'溫度 [{t_unit}]', figsize=(fwidth, fheight))
    ax.set_ylim((10, 30))
    ax2 = ax.twinx()
    diff_.plot(ax=ax2, c='k', ylabel=f'差距 [{t_unit}]', legend=False, marker=marker)
    ax2.legend(loc='upper left')
    ax2.set_ylim((0.2, 1.4))
    fig.tight_layout()

def run_temperature_byStation():
    # CN font
    plt.rcParams['font.family'] = cn_font
    # first, HKO vs KP
    # read data
    datafiles = 'Temperature.csv'
    data = pd.read_csv(f'Data\\{datafiles}')
    year = data['Year']

    # select from 1951
    start_yr = 1951
    ind_ = np.where(year == start_yr)[0][0]
    # line prop
    lw = 1.5

    # plot: HKO==black, KP=blue
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    ax2 = ax.twinx()
    lm_, lm_coef = lm(lmx=year[ind_:], lmy=data['HKO'][ind_:])[0:2]
    l1 = ax.plot(year[ind_:], data['HKO'][ind_:], label=f'天文台', c='k', lw=lw)
    ax.plot(year[ind_:], lm_, c='k')
    index = get_nas_ind(data['KP'][ind_:])
    lm_, lm_coef = lm(lmx=year[ind_ + index[1]:], lmy=data['KP'][ind_ + index[1]:])[0:2]
    l2 = ax2.plot(year[ind_:], data['KP'][ind_:], label=f'京士柏', c='b', lw=lw)
    ax2.plot(year[ind_ + index[1]:], lm_, c='b')
    # two lines in one legend
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)
    ax.set_ylabel(f'天文台平均溫度 [{t_unit}]', labelpad=labelpad)
    yliu, ylil = 25, 20
    ax.set_ylim((ylil, yliu))
    # ax.set_xlabel('年份')
    ax2.set_ylim((ylil + 1, yliu + 1))
    ax2.set_ylabel(f'京士柏平均溫度 [{t_unit}]', labelpad=labelpad, c='b')
    ax2.tick_params(axis='y', colors='blue')
    fig.tight_layout()

    # next, plot LFS, TKL, HKO, SHA, HKS, WGL
    station_ls = np.array([['LFS', 'TKL', 'SHA'], ['HKO', 'HKS', 'WGL']], dtype=object)
    station_ls_cn = np.array([['流浮山', '打鼓嶺', '沙田'], ['天文台', '黃竹坑', '橫瀾島']], dtype=object)

    # plot: 2x3 figure
    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches(14, 8)
    deg = 3
    for i in range(2):
        for j in range(3):
            station = station_ls[i, j]
            index = get_nas_ind(data[station][ind_:])
            lm_, lm_coef = lm(lmx=year[ind_ + index[1]:], lmy=data[station][ind_ + index[1]:])[0:2]
            # pl_ = pl(plx=year[ind_+index[1]:], ply=data[station][ind_+index[1]:], deg=deg)[0]
            ax[i, j].plot(year[ind_ + index[1]:], lm_, c='b')
            # ax[i,j].plot(year[ind_+index[1]:], pl_, c='r', label=f'多項式 (deg.={deg})')
            ax[i, j].plot(year[ind_:], data[station][ind_:], c='k', lw=lw, marker=marker)
            ax[i, j].set_title(station_ls_cn[i, j])
            # ax[i,j].legend(loc='upper left')
            ax[i, j].set_xlabel('年份')
            ax[i, j].set_ylabel(f'溫度[{t_unit}]')
            ax[i, j].set_xlim(1980, 2020)
            ax[i, j].set_ylim(get_limit(data[station][ind_:], scale=0.98))
    fig.tight_layout()

    # and calculate rates
    files = ['Temperature_max.csv', 'Temperature.csv', 'Temperature_min.csv']
    for i, x in enumerate(files):
        data = pd.read_csv(f'Data\\{x}')
        data = data[data['Year'] >= 1991]
        year = data['Year'][data['Year'] >= 1991]
        station = data.columns
        print("######################################################################################")
        print(f'For {t_name[i]}')  # maxT, avgT, minT
        # show coefficients
        print_lm_coef(np.array(data)[:, 1:].T, year, station[1:], ignore_na=True, scale=100, data_len=True)


if __name__ == "__main__":
    start_time = time.time()
    run_temp()
    run_temperature_byStation()
    end_of_code(start_time, show_plots=True)
