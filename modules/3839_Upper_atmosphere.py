########################################################################################################################
# This file is for Chapter 3.8 Climate Change.
# Note:
# Prepared by Arthur, RA. 20211028
########################################################################################################################

# Chapter 3.8.3.9 Upper atmosphere
########################################################################################################################
from modules import *  # functions and global variables


def run_main():
    def to_anomaly(data, ref_period_ind=False):
        data = np.array(data)
        if ref_period_ind:
            ref_data = data[ref_period_ind[0]:ref_period_ind[1]]
            m = np.nanmean(ref_data)
        else:
            m = np.nanmean(data)
        print(f"Mean temperature is {m}")
        new_d = data - m
        return new_d

    plt.rcParams['font.family'] = cn_font

    data = pd.read_csv('Data\\ua_temp_yrmean_1956-2020.csv')
    # select 850 hpa
    ua850 = data[data['pressure'] == 850]
    # remove first data
    ua850 = ua850[1:]
    year = ua850['year']
    pls_ = [850, 700, 500, 400, 300, 250, 200, 150, 100, 50]

    # select data after 1956, in one figure
    data = data[data['year'] > 1956]
    fig, ax = plt.subplots(5, 2)
    fig.set_size_inches(fwidth * 0.8 * 1.5, 4 * 3)
    for i in range(len(pls_)):
        if i < 5:
            row = i
            col = 0
        else:
            row = i - 5
            col = 1
        selection = data[data['pressure'] == pls_[i]]
        dt = selection['temp']
        # take 1957-1980 as reference
        dt = to_anomaly(dt, ref_period_ind=(0, np.where(selection['year'] == 1980)[0][0]))
        ax[row, col].plot(selection['year'], dt, c='k', label=f'{pls_[i]} 百帕斯卡')
        ax[row, col].axhline(y=0, c='grey')
        ax[row, col].set_ylabel(f'溫度距平 [{t_unit}]')
        # ax[row, col].legend(loc='lower right', fontsize=17.5)
        ax[row, col].annotate(f'{pls_[i]} 百帕斯卡', xy=(0.05, 0.77), xycoords='axes fraction', size=20)
        ax[row, col].set_ylim(get_limit(dt, scale=2.2, equal=True))
        # modify y sticks to be integer
        ax[row, col].yaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    fig.savefig('plots\\1957 – 2020京士柏探空儀指定氣壓面氣溫距平（相對於1957-1980平均值）.jpg', dpi=300)


if __name__ == "__main__":
    start_time = time.time()
    run_main()
    end_of_code(start_time, show_plots=True)
