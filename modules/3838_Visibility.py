########################################################################################################################
# This file is for Chapter 3.8 Climate Change.
# Prepared by Arthur, RA. 20211103
########################################################################################################################

# Chapter 3.8.3.8 Visibility
########################################################################################################################
from modules import *  # functions and global variables

def run_main():
    plt.rcParams['font.family'] = cn_font

    HKO_vis = pd.read_csv('Data\\HKO_reduced_visibility_hr.csv')
    HKIA_vis = pd.read_csv('Data\\HKIA_reduced_visibility_hr.csv')
    year = HKO_vis['Year']
    # select after 1970
    HKO_vis = HKO_vis[HKO_vis['Year'] >= 1971]
    # no need to handle HKIA: 1997-2020

    # plot figure
    fig = plt.figure()
    fig.set_size_inches(fwidth * 0.8, fheight * 0.8)
    plt.plot(HKO_vis['Year'], HKO_vis['Total'], label='天文台', c='k', marker=marker)
    plt.plot(HKIA_vis['Year'], HKIA_vis['Total'], label='香港國際機場', c='b', marker=marker)
    plt.legend(loc='upper left')
    plt.grid(zorder=0)
    plt.ylabel('低能見度時數 [小時]')
    fig.tight_layout()
    fig.savefig('plots\\天文台 (1971– 2020) 和香港國際機場 (1997 – 2020)全年低能見度時數.jpg',dpi=300)


if __name__ == "__main__":
    start_time = time.time()
    run_main()
    end_of_code(start_time, show_plots=True)


