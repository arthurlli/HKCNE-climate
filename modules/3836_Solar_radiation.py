########################################################################################################################
# This file is for Chapter 3.8 Climate Change.
# Note:
# Prepared by Arthur, RA. 20211028
########################################################################################################################

# Chapter 3.8.3.3 Section 6 - Solar radiation
########################################################################################################################
from modules import *  # functions and global variables

def run_main():
    plt.rcParams['font.family'] = cn_font

    # read data
    mSR_ = pd.read_csv('Data\\SR_KP.csv', skiprows=1)
    mSR_KSC = pd.read_csv('Data\\SR_KSC.csv')

    # plot data: KP & KSC
    fig = plt.figure()
    fig.set_size_inches(fwidth * 0.8, fheight * 0.8)
    plt.plot(mSR_['Year'], mSR_['Mean'], label='京士柏', c='k')
    plt.plot(mSR_KSC['Year'][1:], mSR_KSC['Mean'][1:], label='滘西洲', c='b')
    plt.ylabel('平均日太陽總輻射量 [兆焦耳/平方米]')
    plt.legend()
    fig.tight_layout()
    fig.savefig('plots\\1968至2020京士柏及滘西洲年平均日太陽總輻射量.jpg', dpi=300)

    # only direct and diffuse
    fig = plt.figure()
    fig.set_size_inches(fwidth * 0.8, fheight * 0.8)
    plt.plot(mSR_['Year'], mSR_['Direct'], label='直接輻射量', c='r')
    plt.plot(mSR_['Year'], mSR_['Diffuse'], label='漫射輻射量', c='b')
    plt.ylabel('平均日直接/漫射輻射量 [兆焦耳/平方米]')
    plt.legend()
    fig.tight_layout()
    fig.savefig('plots\\2009-2020京士柏年平均日太陽直接輻射量和太陽漫射輻射量.jpg', dpi=300)

    # compute and show rates for direct and diffuse
    print_lm_coef(mSR_['Direct'], mSR_['Year'], 'KP Direct SR', ignore_na=True, scale=1, data_len=True)
    print_lm_coef(mSR_['Diffuse'], mSR_['Year'], 'KP Diffuse SR', ignore_na=True, scale=1, data_len=True)


if __name__ == "__main__":
    start_time = time.time()
    run_main()
    end_of_code(start_time, show_plots=True)


