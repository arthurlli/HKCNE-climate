########################################################################################################################
# This file is for Chapter 3.8 Climate Change.
# Note: In paper, the rate of hourly rf>=30mm was used, cited from literature. The data
#       is not available here.
# Prepared by Arthur, RA. 20211110
########################################################################################################################

# Chapter 3.8.2.2 Rainfall
########################################################################################################################
from modules import *  # functions and global variables

def run_rf():
    plt.rcParams['font.family'] = cn_font  # set Chinese font

    # read data
    data = pd.read_csv('Data\\HKO_full.csv')
    data = data[['Year', 'rf']]

    # plot data and lm
    fig = plt.figure()
    fig.set_size_inches(fwidth, fheight)
    plt.plot(data['Year'], data['rf'], c='b', marker='o')
    plt.grid()
    lm_ = lm(lmx=data['Year'], lmy=data['rf'], print_method=False)
    plt.plot(data['Year'], lm_[0], c='b')
    plt.ylabel('雨量 [毫米]')
    fig.tight_layout()
    fig.savefig('plots\\1884-2020香港天文台年雨量紀錄.jpg', dpi=300)

    # compute and show rate
    print_lm_coef(data['rf'], data['Year'], 'Rainfall', ignore_na=True, scale=1, data_len=True)
    print_lm_coef(data['rf'], data['Year'], 'Rainfall', ignore_na=False, scale=1, data_len=True)
    print('####################################################################')
    # plot max 1 hr rf
    data = pd.read_csv('Data\\rf_max1hr.csv')
    # new array to keep max
    arr_ = np.zeros(len(data['Year']))
    dt_ = data['Total']
    for i, x in enumerate(arr_):
        if i == 0:
            arr_[i] = dt_[i]
            print("Record-breaking values are shown below: ")
        else:
            if dt_[i] > arr_[i - 1]:
                arr_[i] = dt_[i]
                print(f"{data['Year'][i]}: {dt_[i]} mm")
            else:
                arr_[i] = arr_[i - 1]

    # plot
    fig = plt.figure()
    fig.set_size_inches(fwidth, fheight)
    plt.plot(data['Year'][1:], arr_[1:], c='b')
    plt.grid()
    plt.ylabel('一小時雨量最高記錄 [毫米]')
    plt.locator_params(axis='x', nbins=10)
    fig.tight_layout()
    fig.savefig('plots\\1885-2020香港天文台總部一小時雨量最高紀錄.jpg', dpi=300)


if __name__ == "__main__":
    start_time = time.time()
    run_rf()
    end_of_code(start_time, show_plots=True)
