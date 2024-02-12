########################################################################################################################
# This file is for Chapter 3.8 Climate Change.
# Note: data column of "Soil_mX" means 7 am, "Soil_aX" means 7 pm
# Note: raw data states unit in 0.1C but it is actually in C.
# Prepared by Arthur, RA. 20211028
########################################################################################################################

# Chapter 3.8.3.4 Soil temperature
########################################################################################################################
from modules import *  # functions and global variables


def run_main():
    plt.rcParams['font.family'] = cn_font

    # read data
    st_hko = pd.read_csv('Data\\soil_T_yr_HKO.csv')
    st_kp = pd.read_csv('Data\\soil_T_yr_KP.csv')

    # select 1981-2020
    start_yr = 1981
    st_hko = st_hko[st_hko['yyyy'] >= start_yr]
    st_kp = st_kp[st_kp['yyyy'] >= start_yr]
    year = st_hko['yyyy']

    # data already in deg. C
    cols = st_hko.columns[1:]
    st_hko = st_hko[cols]
    st_kp = st_kp[cols]

    # select only am 7: soil_m
    # print(cols)
    st_hko = st_hko[cols[:7]]
    st_kp = st_kp[cols[:7]]
    # check
    assert np.all(st_hko.columns == st_kp.columns)
    # print(st_kp.columns)

    # readable column names: soil_m1-7 
    label_names = ['5 cm', '10 cm', '20 cm', '50 cm', '100 cm', '150 cm', '300 cm']

    # plot figure
    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(fwidth * 0.8, fheight * 1.1)
    # row1 == HKO, row2 == KP
    for j in range(7):
        ax[0].plot(year, st_hko.iloc[:, j], label=label_names[j], marker=marker)
        ax[1].plot(year, st_kp.iloc[:, j], label=label_names[j], marker=marker)
    ax[0].set_ylabel(r'溫度 [$\degree$C]')
    ax[0].text(x=1981, y=26.3, s='天文台')
    ax[1].set_ylabel(r'溫度 [$\degree$C]')
    ax[1].text(x=1981, y=26.3, s='京士柏')
    ax[0].set_ylim(get_limit(st_hko, scale=0.99))
    ax[1].set_ylim(get_limit(st_hko, scale=0.99))
    # inverse legend for readibility
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.01, 1.01))
    fig.tight_layout()
    # fig.savefig('1981-2020天文台和京士柏土壤溫度（上午7時）.jpg', dpi=300)

    # show only HKO
    fig, ax = plt.subplots()
    fig.set_size_inches(fwidth * 0.8, (fheight * 1.1) / 2)
    for i in range(7):
        ax.plot(year, st_hko.iloc[:, i], label=label_names[i], marker=marker)
    ax.set_ylabel(r'溫度 [$\degree$C]')
    ax.text(x=1981, y=26.3, s='天文台')
    ax.set_ylim(get_limit(st_hko, scale=0.99))
    # inverse legend for readibility
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.01, 1.01))
    fig.tight_layout()
    fig.savefig('plots\\1981-2020天文台土壤溫度（上午7時）.jpg', dpi=300)

    # select 1991-2020: re-read data
    st_hko = pd.read_csv('Data\\soil_T_yr_HKO.csv')
    st_kp = pd.read_csv('Data\\soil_T_yr_KP.csv')
    # selection
    start_yr = 1991
    year_91 = year[year >= start_yr]
    st_hko = st_hko[st_hko['yyyy'] >= start_yr]
    st_kp = st_kp[st_kp['yyyy'] >= start_yr]
    cols = st_hko.columns[1:8]
    st_hko, st_kp = st_hko[cols], st_kp[cols]

    # check
    assert np.all(st_hko.columns == st_kp.columns)
    # print(st_kp.columns)

    # compute and show rates
    print('For HKO:')
    print_lm_coef(np.array(st_hko).T, year_91, label_names, True, 100, data_len=True)
    print('######################################################################################')
    print('For KP:')
    print_lm_coef(np.array(st_kp).T, year_91, label_names, True, 100, data_len=True)
    # Note: only 5 cm, 20 cm, 1 m, 3 m are coorperated in the writing.


if __name__ == "__main__":
    start_time = time.time()
    run_main()
    end_of_code(start_time, show_plots=True)
