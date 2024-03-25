# This .py contains configurations, functions, and constants
########################################################################################################################
# For necessary packages
import numpy as np
import pandas as pd
from math import exp, pi, sin, sqrt, floor, ceil
import matplotlib.pyplot as plt
import matplotlib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm  # Statistical Models
import scipy.optimize
from sklearn.decomposition import PCA
from matplotlib import cm
import seaborn as sns
from tqdm import tqdm
import time

########################################################################################################################
# global variables:
# temperature data:
t_name = ['Max. T', 'Avg. T', 'Min. T', 'T diff.']
t_ls = ['-.', '-', '--']
t_col = ['red', 'k', 'blue', 'orange']
# solar radiation:
month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
seasons = [['Mar', 'Apr', 'May'], ['Jun', 'Jul', 'Aug'],
           ['Sep', 'Oct', 'Nov'], ['Dec', 'Jan', 'Feb']]  # defined as MAM, JJA, SON, DJF
seasons_col = ['green', 'orange', 'red', 'blue']
seasons_name = ['Spr', 'Sum', 'Aut', 'Win']
seasons_name_cn = ['春', '夏', '秋', '冬']
# set plot fonts:
cn_font = 'Taipei Sans TC Beta'
en_font = 'DejaVu Sans'
larger_s, medium_s, small_s = 22, 20, 15
font = {'family': en_font,
        'weight': 'normal',
        'size': medium_s}
axes = {'titlesize': larger_s,
        'labelsize': medium_s}
ticks = {'labelsize': medium_s}
legend = {'fontsize': small_s}
matplotlib.rc('font', **font)
matplotlib.rc('axes', **axes)
matplotlib.rc('xtick', **ticks)
matplotlib.rc('ytick', **ticks)
matplotlib.rc('legend', **legend)
# set plotting params:
marker = '.'
subp_ls = '--'
subp_a = 0.5
fwidth, fheight = 12, 7
labelpad = 10
t_unit = f'$\degree$C'

# default stat params
deg = 3  # polynomial
m = 10  # moving average


# Functions:
########################################################################################################################
# Do fft for input data:
def fft_spectral(data):
    # should return power spectral density
    def time_series(N, dt, tmin):
        """ Create auxillary variables t, f and w
            :param N: even integer
                No. of samples.
            :param dt: float
                Sampling spacing of the data
            :param tmin: float
                Start time of the series

            :return: t (np.array,(N,1))
                time series
            :return: f (np.array,(N/2,1))
                frequency series
            :return: w (np.array, (N/2,1))
                angular frequency sereis
            :return: Nf (int)
                floor(N/2+1)
        """
        # making the auxillary variable t
        tmax = tmin + dt * (N - 1)
        t = np.zeros((N, 1))
        t[:, 0] = np.linspace(tmin, tmax, N)

        # Nyquist frequencies and frequency
        Nf = floor(N / 2 + 1)
        fmax = 1 / (2 * dt)
        df = fmax / (N / 2)
        f = np.zeros((Nf, 1))
        f[:, 0] = np.linspace(0, fmax, Nf)
        w = f * 2 * np.pi
        return t, f, w, Nf

    def Fouier_kernal(N, M, t, w):
        G = np.zeros((N, M))
        # What is the value of zero frequency?
        G[:, 0] = np.ones((N, 1)).ravel()  # fisrt col is ones
        Mo2 = floor(M / 2)
        for i in range(1, Mo2):
            j = 2 * i - 1
            k = j + 1
            G[:, j] = np.cos(w[i, 0] * t).ravel()  # refer to the png
            G[:, k] = np.sin(w[i, 0] * t).ravel()  # also...
        # nyquist column
        G[:, M - 1] = np.cos(w[-1, 0] * t).ravel()  # the 1 and -1
        return G

    time = np.linspace(1, len(data) + 1, endpoint=False, dtype=int, num=len(data))
    data = np.vstack((time, data))
    assert not np.any(np.isnan(data)), "Has NA in input data"  # if not pass, raise error
    D = data.T
    [Nraw, K] = D.shape
    traw = np.zeros((Nraw, 1))
    traw[:, 0] = D[:, 0]
    Dt = traw[1, 0] - traw[0, 0]
    draw = np.zeros((Nraw, 1))
    draw[:, 0] = D[:, 1]

    # round off to even number of points
    No2 = floor(Nraw / 2)
    N = 2 * No2
    d = np.zeros((N, 1))
    d[:, 0] = draw[0:N, 0]
    t = np.zeros((N, 1))
    t[:, 0] = traw[0:N, 0]

    # obtain the parameters from the dataset
    Dt = t[1, 0] - t[0, 0]
    tmin = t[0, 0]
    t, f, w, Nf = time_series(N, Dt, tmin)
    Nw = Nf

    # solve G
    Mo2 = No2
    M = N
    G = Fouier_kernal(N, M, t, w)

    # analytic formula for inv(GT*G)
    gtgi = 2 * np.ones((M, 1)) / N
    gtgi[0, 0] = 1 / N
    gtgi[M - 1, 0] = 1 / N

    mest = np.multiply(gtgi, np.matmul(G.T, d))

    # compute spectral density
    s = np.zeros((Nw, 1))

    # zero frequency
    s[0, 0] = mest[0, 0]
    # s2 = np.sqrt(mest[1,0]**2 + mest[2,0]**2)

    for i in range(1, Mo2):
        j = 2 * i - 1
        k = j + 1
        s[i, 0] = np.sqrt(mest[j, 0] ** 2 + mest[k, 0] ** 2)  # see amplitude spectral density

    # Nyquist frequency
    s[Nw - 1, 0] = mest[-1, 0]

    # return spectral density and frequency
    return s, f


########################################################################################################################
# handle missing data
def handle_missingData(dt, d, AutoFill=False, print_method=True):
    method = 'mean'
    if not AutoFill:
        while True:
            check = input(f'Use {method} value to fill in, OK? [Y/N]')
            if check in ['Y', 'y', 'yes', 'ok', 'OK', 'Ok']:
                break
            else:
                method = input(f'Choose a method: ')
    else:
        if print_method:
            print(f'Missing data observed and handled: NA filled by {method}')
    imp = SimpleImputer()
    dt = np.array(dt)
    if d == 1:  # 1 dimension
        dt = imp.fit_transform(dt.reshape(-1, 1)).T
        if len(dt) == 1:
            dt = dt[0]
    else:
        dt = imp.fit_transform(dt.T).T
    return dt


########################################################################################################################
# Linear regression
def lm(lmx, lmy, get_sum=False, showp=False, print_method=True):
    # handle missing data:
    if np.any(np.isnan(lmy)):
        lmy = handle_missingData(dt=lmy, d=1, AutoFill=True, print_method=print_method)

    lmx = sm.add_constant(lmx)
    model = sm.OLS(lmy, lmx).fit()
    lmr = model.predict(lmx)
    if get_sum:
        print(model.summary())
    # R2, intercept, coef, sum of squared residuals, standard errors, rate p-value ([0] is for const, [1] for 1st coef)
    r2 = model.rsquared if not np.isinf(model.rsquared) else 0
    lmrc = [r2, model.params[0], model.params[1], np.sum(np.square(model.resid)), model.bse, model.pvalues[1]]
    if showp:
        print(f'P-value for coef. is {model.pvalues[1]}')
    return lmr, lmrc


# a function to show trends
# Note: turn on plot_data & plot_lm can see more info
def print_lm_coef(data, year, label_names, ignore_na=False, scale=1, data_len=False,
                  plot_lm=False, plot_data=False, range=False, title=None):
    def handle_data(x, data_len, ignore_na):
        if ignore_na:
            new_x = x[np.logical_not(np.isnan(x))]
            new_year = year[np.logical_not(np.isnan(x))]
            no_nas = len(x) - len(new_x)  # count NAs ignored
            txt = f"NA ignored (data length={len(new_x)}; number of NAs observed={no_nas})" if data_len else "NA ignored."
        else:
            new_x = x
            new_year = year
            no_nas = np.sum(np.isnan(new_x))  # count NAs in dataset
            txt = f"NA filled by mean (data length={len(new_x)}; number of NAs observed={no_nas})" if data_len else "NA filled by mean."
        return new_x, new_year, txt

    if len(data.shape) == 1:
        new_x, new_year, extra_ = handle_data(data, data_len, ignore_na)
        lm_, coef = lm(lmx=new_year, lmy=new_x)
        print(f"{label_names}: {scale}*rate={coef[2] * scale}, SE={coef[4][1] * scale}, p-value={coef[5]}, " + extra_)

    if len(data.shape) == 2:
        for i, x in enumerate(np.array(data)):
            new_x, new_year, extra_ = handle_data(x, data_len, ignore_na)
            lm_, coef = lm(lmx=new_year, lmy=new_x)
            print(
                f"{label_names[i]}: {scale}*rate={coef[2] * scale}, SE={coef[4][1] * scale}, p-value={coef[5]}, " + extra_)
            if plot_lm:
                if i == 0:
                    fig = plt.figure()
                plt.plot(lm_, label=label_names[i])
            if plot_data:
                plt.plot(x)
        if plot_lm:
            plt.legend(bbox_to_anchor=(1.01, 1.01))
            plt.title(title)
            if range:
                plt.ylim(range)
            fig.tight_layout()


########################################################################################################################
# Polynomials
def pl(plx, ply, deg, cov=True):
    plc = np.polyfit(plx, ply, deg, cov=cov)
    pl = np.poly1d(plc[0])
    plr = pl(plx)
    plcoef, plcov = plc[0], plc[1]  # 1st is coefficients, 2nd is covariances: degree+1 dimensionals
    return plr, plcoef, plcov


########################################################################################################################
# Moving average
def ma(may, n, mode='valid'):
    mar = np.convolve(may, np.ones(n) / n, mode=mode)
    return mar


########################################################################################################################
# Exponential fit
def epf(x, m, t):
    # exponential function
    return m * np.exp(t * x)


########################################################################################################################
def get_nas_ind(data):
    """
    :param data: 1D data
    :return: 1st NA and last NA positions (only non-fillable NAs)
    """
    if np.any(np.isnan(data)):
        array = np.array(data)
        nas = np.where(np.isnan(array))[0]
        start = 0
        end = 0
        if nas[0] == 0:
            for i, x in enumerate(nas):
                if nas[i] == i:
                    end += 1
                    pass
                else:
                    return start, end
        else:
            start = nas[0]
            end = nas[-1]
        return start, end
    else:
        return 0, 0


########################################################################################################################
# contains LM, PL, MA for data smoothing:
def data_smoothing(data, x, poly_degree, ma_num, AutoFillNAN=False):
    """
    :key: This function will process the input data and output a corrected data, if Na observed.
          Results will show linear regression, polynomial, moving average. Additions are pending.
    :param data: data
    :param x: years, or 1D array
    :param poly_degree: degree for polynomial
    :param ma_num: moving average number n
    :param AutoFillNAN: BOOL, ask user or not when filling NA
    :return:
        Output data: after missing data handling, if any;
        Linear regression: 1st is predictions; 2nd is [R2, Intercept];
        Polynomials: 1st is predictions; 2nd is coefficients; 3rd is covariance matrix;
        Moving average: only predictions;
    """
    # dimensions of array
    dim = len(data.shape)

    # handle if NAs at beginning:
    if dim == 1 and np.any(np.isnan(data)):
        ind1, ind2 = get_nas_ind(data)
        if ind1 == 0:
            # if array starts with NA, ignore them
            data = data[ind2:]
            x = x[ind2:]

    # deal with missing data
    if np.any(np.isnan(data)):
        print(f'Missing data observed.')
        data = handle_missingData(dt=data, d=dim, AutoFill=AutoFillNAN)
    # update output data
    output_dt = data

    # main dish
    if dim == 1:
        # linear regression
        lm_result, lm_coef = lm(lmx=x, lmy=data)
        # polynomials
        pl_result, pl_coef, pl_cov = pl(plx=x, ply=data, deg=poly_degree, cov=True)
        # moving average
        ma_result = ma(may=data, n=ma_num, mode='valid')
    else:
        # prepare "results matrix"
        nrow, ncol = len(data[:, 0]), len(data[0, :])
        lm_result, pl_result = [np.zeros((nrow, ncol)) for loop in range(2)]
        ma_result = np.zeros((nrow, ncol - ma_num + 1))
        # prepare matrices
        lm_coef = np.zeros((nrow, 4))
        pl_cov = np.zeros((nrow, poly_degree + 1, poly_degree + 1))
        pl_coef = np.zeros((nrow, poly_degree + 1))
        # smoothing
        for i, dt in enumerate(data):
            # linear regression
            lm_result[i, :] = lm(lmx=x, lmy=dt)[0]
            lm_coef[i, :] = lm(lmx=x, lmy=dt)[1][:-2]
            # polynomials
            pl_result[i, :], pl_coef[i, :], pl_cov[i, :] = pl(plx=x, ply=dt, deg=poly_degree, cov=True)
            # moving average
            ma_result[i, :] = ma(may=dt, n=ma_num, mode='valid')
    return output_dt, lm_result, lm_coef, pl_result, pl_coef, pl_cov, ma_result


# a function to plot LM PL MA with infos
def plot_trends(smoothed_results, x, deg, m, c='k'):
    lm_, lm_coef = smoothed_results[1], smoothed_results[2]
    pl_ = smoothed_results[3]
    ma_ = smoothed_results[6]
    plt.plot(x, lm_, c=c, ls=t_ls[1], label=f'LM, y={lm_coef[1]:.2f}+{lm_coef[2]:.2f}x')
    plt.plot(x, pl_, c=c, ls=t_ls[0], label=f'PL, deg.={deg}')
    plt.plot(x[m // 2:-(m // 2)], ma_, c=c, ls=t_ls[2], label=f'MA, m={m}')


########################################################################################################################
# Plot function, plot data as well as smoothed lines:
def plot_smoothed_data(x, y, data_label, col, ls, lm_mtx, pl_mtx, ma_mtx, deg, m, fig_name, xlabel, ylabel,
                       title=None, multifig=False, outside_legend=True, fwidth=fwidth, fheight=fheight, savefig=True,
                       show_params=['xlabel', 'ylabel', 'legend', 'title', 'lm', 'pl', 'ma']):
    """
    :param x: data in x-axis
    :param y: data in y-axis
    :param data_label: data names, array
    :param col: data col, array
    :param ls: line styles
    :param lm_mtx: LM results, same shape as y
    :param pl_mtx: PL results, same shape as y
    :param ma_mtx: MA results, same shape as y
    :param deg: PL degs
    :param m: MA m
    :param fig_name: figure name for saving, without .jpg
    :param title: fig title
    :param xlabel: as named
    :param ylabel: as named
    :param multifig: BOOL
    :param show_params: key properties to show
    :return:
        Plot and save, example see Annual Temperature.
    """
    start, end = m // 2 - 1, m // 2 - m  # for MA
    # two conditions:
    if not multifig:
        fig, ax = plt.subplots()
        fig.set_size_inches(fwidth, fheight)
        if len(y.shape) == 1:
            ax.plot(x, y, label=data_label, marker=marker, c=col, ls=ls[0])
            if 'lm' in show_params:
                ax.plot(x, lm_mtx, c=col, label='Linear regressions', ls=ls[1])
            # polynomials:
            if 'pl' in show_params:
                ax.plot(x, pl_mtx, c=col, label=f'Polynomials (deg. {deg})', ls=ls[2])
            # moving averages:
            if 'ma' in show_params:
                ax.plot(x[start:end], ma_mtx, c=col, label=f'Moving average (m={m})', ls=ls[3])

        else:
            # data:
            for i, data in enumerate(y):
                ax.plot(x, data, label=data_label[i], marker=marker, c=col[1], ls=ls[i])
                # smoothed lines:
                # LM lines:
            for i in range(len(lm_mtx[:, 0])):
                if 'lm' in show_params:
                    ax.plot(x, lm_mtx[i, :], c=col[2], label='Linear regressions' if i == 0 else None, ls=ls[1])
                # polynomials:
                if 'pl' in show_params:
                    ax.plot(x, pl_mtx[i, :], c=col[0], label=f'Polynomials (deg. {deg})' if i == 0 else None, ls=ls[1])
                # moving averages:
                if 'ma' in show_params:
                    ax.plot(x[start:end], ma_mtx[i, :], c=col[3], label=f'Moving average (m={m})' if i == 0 else None,
                            ls=ls[1])
        # set params:
        if 'title' in show_params:
            ax.set_title(title)
        if 'xlabel' in show_params:
            ax.set_xlabel(xlabel)
        if 'ylabel' in show_params:
            ax.set_ylabel(ylabel)
        if 'legend' in show_params:
            ax.legend(bbox_to_anchor=(1.01, 1.01)) if outside_legend else ax.legend()
        plt.tight_layout()
        if savefig:
            # savefig
            fig.savefig(f'plots\\{fig_name}.jpg', dpi=300)
            print('Saved.')
    else:
        nrow, ncol = 2, 2
        fig, ax = plt.subplots(nrow, ncol)
        fig.set_size_inches(fwidth * 2, fheight * 3)
        for i in range(nrow):
            for j in range(ncol):
                ind = i + j if i == 0 else i + j + 1
                for k, data in enumerate(y):
                    ax[i, j].plot(x, data, label=data_label[i], marker=marker, c='k',
                                  ls='-', alpha=1 if k == ind else subp_a)
                # after all data lines, then smoothed lines:
                for z in range(len(lm_mtx[:, 0])):
                    ax[i, j].plot(x, lm_mtx[z, :], c=col[2] if z == ind else 'k', alpha=1 if z == ind else subp_a,
                                  label='Linear regressions' if i == 0 else None, ls=ls[1])
                    # polynomials:
                    ax[i, j].plot(x, pl_mtx[z, :], c=col[0] if z == ind else 'k', alpha=1 if z == ind else subp_a,
                                  label=f'Polynomials (deg. {deg})' if i == 0 else None, ls=ls[1])
                    # moving averages:
                    ax[i, j].plot(x[start:end], ma_mtx[z, :], c=col[3] if z == ind else 'k',
                                  alpha=1 if z == ind else subp_a,
                                  label=f'Moving average (m={m})' if i == 0 else None, ls=ls[1])
                # set params:
                ax[i, j].set_title(title[ind])
                ax[i, j].set_xlabel(xlabel)
                ax[i, j].set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(f'plots\\{fig_name}.jpg', dpi=300)


def plot_data_separation(x, y, data_label, col, ls, deg, m, fig_name, xlabel, ylabel, title=None, outside_legend=True):
    start, end = m // 2 - 1, m // 2 - m  # for MA
    fig, ax = plt.subplots()
    fig.set_size_inches(fwidth, fheight)
    if len(y.shape) is not 1:
        ind_start, ind_end = get_nas_ind(y[0, :])  # note that avg T has NAN at beginning
        for i, data in enumerate(y):
            ax.plot(x, data, label=data_label[i], marker=marker, c='k', ls=ls[i])

        # 1st half
        tp, lm_mtx, lm_coef, pl_mtx, pl_coef, pl_cov, ma_mtx = data_smoothing(y[:, :ind_start], x[:ind_start], deg, m,
                                                                              True)
        for i in range(len(lm_mtx[:, 0])):
            ax.plot(x[:ind_start], lm_mtx[i, :], c=col[2], label='Linear regressions' if i == 0 else None, ls=ls[1])
            # polynomials:
            ax.plot(x[:ind_start], pl_mtx[i, :], c=col[0], label=f'Polynomials (deg. {deg})' if i == 0 else None,
                    ls=ls[1])
            # moving averages:
            ax.plot(x[start:end + ind_start], ma_mtx[i, :], c=col[3],
                    label=f'Moving average (m={m})' if i == 0 else None, ls=ls[1])

        # 2nd half
        tp, lm_mtx, lm_coef, pl_mtx, pl_coef, pl_cov, ma_mtx = data_smoothing(y[:, ind_end + 1:], x[ind_end + 1:], deg,
                                                                              m, True)
        for i in range(len(lm_mtx[:, 0])):
            ax.plot(x[ind_end + 1:], lm_mtx[i, :], c=col[2], ls=ls[1])
            # polynomials:
            ax.plot(x[ind_end + 1:], pl_mtx[i, :], c=col[0], ls=ls[1])
            # moving averages:
            ax.plot(x[start + 1 + ind_end:end], ma_mtx[i, :], c=col[3], ls=ls[1])
        # add vertical lines
        ax.axvline(x[ind_start], c='k')
        ax.axvline(x[ind_end], c='k')
    # set params:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.01, 1.01)) if outside_legend else ax.legend()
    plt.tight_layout()
    # savefig
    fig.savefig(f'plots\\{fig_name}.jpg', dpi=300)
    print('Saved.')


########################################################################################################################
# seasonal temperatere with yearly avg
def interval_avged_t(interval, original_dt, plot_x, original_x, pl_deg, ma_m, y_label=None, y_unit=None,
                     seasonal=False):
    """
    :param interval: time interval
    :param original_dt:
    :param plot_x: x values for plotting (e.g., 1981-1990)
    :param original_x: with length of original data (e.g., 137 for HKO Temperature)
    :param pl_deg: polynomial param deg
    :param ma_m: moving average param m
    :param y_label: first plot y label
    :param y_unit: as named
    :return:
    """
    len_dt = len(plot_x)
    if seasonal:
        t_sea = np.zeros((4, 3, len_dt))
    else:
        t_sea = np.zeros((1, 3, len_dt))
    t_sea_lm = np.zeros(t_sea.shape)
    fill = original_x[0] % interval
    ind = interval - fill + 1

    deg, m = pl_deg, ma_m
    input_yr = np.linspace(1, len_dt, len_dt)

    for i, sea in enumerate(t_sea):
        for j, type in enumerate(sea):
            for k, ele in enumerate(type):
                if k == 0:
                    # first 1884-1890
                    if i == 0:
                        print(original_x[0:ind])
                    ys = original_dt[i][j, 0:ind]
                    type[k] = np.nanmean(ys)
                    if np.all(np.isnan(ys)):
                        t_sea_lm[i][j, k] = np.nan
                    else:
                        t_sea_lm[i][j, k] = lm(lmx=np.linspace(1, ind, ind), lmy=ys)[1][2]
                else:
                    if i == 0:
                        print(original_x[(k * interval) - fill + 1:interval * (k + 1) - fill + 1])
                    ys = original_dt[i][j, (k * interval) - fill + 1:interval * (k + 1) - fill + 1]
                    type[k] = np.nanmean(ys)
                    if np.all(np.isnan(ys)):
                        t_sea_lm[i][j, k] = np.nan
                    else:
                        t_sea_lm[i][j, k] = lm(lmx=np.linspace(1, interval, interval), lmy=ys)[1][2]

    for i, dt in enumerate(t_sea):
        # smooth data
        decade_t, lm_mtx, lm_coef, pl_mtx, pl_coef, pl_cov, ma_mtx = data_smoothing(dt, input_yr, deg, m,
                                                                                    AutoFillNAN=True)
        # plot years mean values:
        fig, ax = plt.subplots()
        if interval > 5:
            fig.set_size_inches(7, 5)
        else:
            fig.set_size_inches(8.5, 5)
        for j in range(3):
            ax.plot(plot_x, dt[j, :], c=t_col[1], marker=marker, label=t_name[j], ls=t_ls[j])
        for j in range(3):
            ax.plot(plot_x, lm_mtx[j, :], c=t_col[2], ls='-', label='LM' if j == 0 else None)
            ax.plot(plot_x, pl_mtx[j, :], c=t_col[0], ls='-', label='PL' if j == 0 else None)
            ax.plot(plot_x[1:-1], ma_mtx[j, :], c=t_col[3], ls='-', label='MA ' if j == 0 else None)
        plt.xticks(rotation=45 if interval > 5 else 90, fontsize=small_s)
        ax.set_title(f'{seasons_name[i]}' if seasonal else None)
        ax.set_xlabel('Period', fontsize=small_s)
        ax.set_ylabel(fr'{y_label} {y_unit}', fontsize=small_s)
        ax.legend(bbox_to_anchor=(1.01, 1.01))
        # ax.tick_params(axis='x', length=6, width=2, labelrotation=45)
        fig.tight_layout()
        fig.savefig(f'plots\\{interval}yr_T' + seasons_name[i] if seasonal else None + '.jpg', dpi=300)

    for i, sea in enumerate(t_sea_lm):
        # plot years coef:
        fig, ax = plt.subplots()
        if interval > 5:
            fig.set_size_inches(7, 5)
        else:
            fig.set_size_inches(8.5, 5)
        # ax.locator_params(tight=True,nbins=5)
        plt.axhline(y=0, c='k', ls='-', alpha=subp_a * 0.2)
        for j, type in enumerate(sea):
            ax.plot(plot_x, type, label=t_name[j], ls=t_ls[j], c='k')
        ax.set_title(f'{interval}-year rate ({seasons_name[i] if seasonal else None})')
        ax.set_xlabel('Period', fontsize=small_s)
        ax.set_ylabel(f'Rate {y_unit}', fontsize=small_s)
        ax.legend()
        plt.xticks(rotation=45 if interval > 5 else 90, fontsize=small_s)
        # set same length
        ax.set_ylim(get_limit(sea, scale=1.1, equal=True))
        # ax.tick_params(axis='x', length=0.5, reset=True)
        fig.tight_layout()
        fig.savefig(f'plots\\{interval}yr_Trate' + seasons_name[i] if seasonal else None + '_rate.jpg', dpi=300)

    return t_sea, t_sea_lm


# Note: this function is not complete.

########################################################################################################################
# 50yr bar plots:
def fifty_aveged_t(original_dt, plot_x, original_x, savename=None, titles=None):
    """
    :param original_dt: data for calculation
    :param plot_x: x values for plotting
    :param original_x: x values of original data
    :param savename: name for saving as jpg
    :param titles: plot title (2)
    :return: plots
    """
    t = np.zeros((3, 3))
    rate = np.zeros(t.shape)
    std1 = np.zeros(t.shape)
    std2 = np.zeros(rate.shape)
    for i, x in enumerate(t):
        for j, y in enumerate(x):
            if j == 0:
                dt_arr = original_dt[i, 0:np.where(original_x == 1921)[0][0] + 1]
            elif j == 1:
                dt_arr = original_dt[i, np.where(original_x == 1921)[0][0]:np.where(original_x == 1971)[0][0]]
            else:
                dt_arr = original_dt[i, np.where(original_x == 1971)[0][0]:]
            t[i, j] = np.nanmean(dt_arr)
            x = np.linspace(1, len(dt_arr), len(dt_arr))
            rate[i, j] = lm(lmx=x, lmy=dt_arr)[1][2]
            std1[i, j] = np.nanstd(dt_arr)
            std2[i, j] = lm(lmx=x, lmy=dt_arr)[1][4][1]

    # plot
    # set limits:

    y1 = np.nanmin(t) - std1[np.where(t == np.nanmin(t))[0][0], np.where(t == np.nanmin(t))[1][0]]
    y2 = np.nanmax(t) + std1[np.where(t == np.nanmax(t))[0][0], np.where(t == np.nanmax(t))[1][0]]
    y3 = np.nanmin(rate) - std2[np.where(rate == np.nanmin(rate))[0][0], np.where(rate == np.nanmin(rate))[1][0]]
    y4 = np.nanmax(rate) + std2[np.where(rate == np.nanmax(rate))[0][0], np.where(rate == np.nanmax(rate))[1][0]]
    if y3 > 0:
        y3 = 0
    ylims = [(y1 * 0.9, y2 * 1.1),
             (y3 * 1.1, y4 * 1.1)]
    ylabels = [r'Mean temperature [$\degree$C]', r'Rate [$\degree$C/yr]']

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(15, 7)
    X = np.arange(3)
    for i in range(3):
        ax[0].bar(X + 0.25 * i, t[i, :], yerr=std1[i, :], capsize=10, width=0.25, label=t_name[i])
        ax[1].bar(X + 0.25 * i, rate[i, :], yerr=std2[i, :], capsize=10, width=0.25, label=t_name[i])
        if y3 != 0:
            ax[1].axhline(y=0, c='k', ls='-', alpha=subp_a)
    for j in range(2):
        ax[j].set_ylim(ylims[j])
        ax[j].set_xticks([r + 0.25 for r in range(3)])
        ax[j].set_xticklabels(plot_x)
        ax[j].legend()
        ax[j].set_title(titles[j])
        ax[j].set_ylabel(ylabels[j])
    fig.tight_layout()
    if savename is not None:
        fig.savefig(f'plots\\{savename}.jpg', dpi=300)


########################################################################################################################
# compute x- y- limits:
def get_limit(data, err=None, scale=0.95, equal=False):
    """
    :param data: array
    :param err: error data
    :param scale: as named
    :return: turple for set_lim
    """

    def get_largest(data):
        max_, min_ = np.nanmax(data), np.nanmin(data)
        if max_ >= min_:
            return max_
        elif max_ < min_:
            return min_

    if err is None:
        min = np.nanmin(data) * scale
        max = np.nanmax(data) / scale
    else:
        min = (np.nanmin(data) - np.nanmax(err)) * scale
        max = (np.nanmax(data) + np.nanmax(err)) / scale
    if equal:
        l_ = get_largest(data)
        min, max = -l_ * scale, l_ * scale
    # return tuple
    return (min, max)


########################################################################################################################
# compute avg rate by saperating a dataset:
def compute_before_after(data, n):
    """
    :param data: input data for separation
    :param n: n data from the end
    :return: mean before n, std of that mean, rate before n, and std for that rate.
             Second row replace 'before' by 'after'.
    """
    x = np.linspace(1, len(data), len(data))
    before = data[0:-n]
    assert len(before) == len(data) - n, "Diff. length of before."
    before_avg = np.nanmean(before)
    # LM:
    blm = lm(lmx=x[0:-n], lmy=before)
    before_rate = blm[1][2]
    before_avg_std = np.nanstd(before)
    before_rate_std = blm[1][4][1]

    after = data[-n:]
    assert len(after) == n, "Diff. length of after."
    after_avg = np.nanmean(after)
    # LM:
    alm = lm(lmx=x[-n:], lmy=after)
    after_rate = alm[1][2]
    after_avg_std = np.nanstd(after)
    after_rate_std = alm[1][4][1]

    return [[before_avg, before_avg_std, before_rate, before_rate_std],
            [after_avg, after_avg_std, after_rate, after_rate_std]]


########################################################################################################################
# show recent comparison
def show_recentyr_compare(data, year, n):
    try:
        print(f'For {data.name}')
    except:
        pass
    if n >= len(data[np.invert(np.isnan(data))]):
        # select only non na values
        data = data[np.invert(np.isnan(data))]
        mean = np.nanmean(data)
        rate = lm(lmx=year[-len(data):], lmy=data)[1][2]
        print(f'In recent {n} years, mean={mean:.2f}, rate={rate:.2f})')
    else:
        # note that lm() will automatically fill NA, so make sure omitting meaningless NAs!
        if len(data) < 2020 - 1947:
            data = data[np.invert(np.isnan(data))]
        r = compute_before_after(np.array(data), n)
        print(f'Before {np.array(year)[-n]} mean={r[0][0]:.2f}, rate={r[0][2]:.2f}')
        print(f'After {np.array(year)[-n]} mean={r[1][0]:.2f}, rate={r[1][2]:.2f}')


########################################################################################################################
# find highest x number in dataset:
def find_largest(data, n):
    """
    :param data: dataset
    :param n: highest n number
    :return: n floats
    """
    # to np array
    data = np.array(data)
    # to 1D
    data = data.ravel()
    # set na to 0
    data[np.isnan(data)] = 0
    # order: 1,2,3,4,5
    results = np.sort(data)[-n:]
    return results


########################################################################################################################
# count data that > than n
def count_larger(data, n):
    """
    :param data: 1 or 2D
    :param n: threshold
    :return: length of data that larger than n
    """
    data = np.array(data)
    data = data.ravel()
    data[np.isnan(data)] = 0
    results = len(data[data > n])
    return results


########################################################################################################################
# ask font type: CN or EN
def ask_font():
    """
    :return: user input for Chinese or English version
    """
    ans = input(f"Specify Chinese [CN] or English [EN] for plots: ")
    if ans in ['CN', 'cn', 'Chinese', 'chinese', 'c']:
        return 'c'
    else:
        return 'e'


########################################################################################################################
# compute rate of change of slope
def compute_rate_of_change(data, n, mode='pm'):
    def get_index(i, n):
        hn = n // 2
        if mode == 'pm':
            ind1 = i - hn
            ind2 = i + hn + 1
        elif mode == 'p':
            ind1 = i
            ind2 = i + n + 1
        elif mode == 'm':
            ind1 = i - n
            ind2 = i + 1
        else:
            raise exception("Wrong mode input")
        return ind1, ind2

    # fill in NA
    if np.any(np.isnan(data)):
        dim = len(data.shape)
        data = handle_missingData(data, dim, True)
    length_ = len(data)
    x = np.linspace(1, length_, length_)
    results_ = []
    count, start_ind = 0, 0
    for i in range(length_):
        try:
            ind1, ind2 = get_index(i, n)
            if ind1 >= 0 and ind2 <= length_:
                select_ = data[ind1:ind2]
                coef_ = lm(lmx=x[ind1:ind2], lmy=select_)[1][2]
                results_.append(coef_)
                if count == 0:
                    start_ind = i
                    count += 1
                end_ind = i
            else:
                pass
        except:
            pass
    return results_, start_ind, end_ind + 1


########################################################################################################################
# seaborn boxplot with points
def sb_boxplot(data, title):
    fig = plt.figure()
    ax = sns.boxplot(data=data)
    ax = sns.swarmplot(data=data, color=".25")
    ax.set_title(title)


# detrend 1D or 2D dataframe
def detrend(data):
    def remove_lm(data):
        lm_ = lm(lmx=np.linspace(1, len(data), len(data)), lmy=data)
        new_dt = data - lm_[0]
        return new_dt

    if len(data.shape) == 1:
        # if it is 1 D data
        data = np.array(data)
        if not np.any(np.isnan(data)):
            new_dt = remove_lm(data)
            return new_dt
    elif len(data.shape) == 2:
        cols_ = data.columns
        ncol = data.shape[1]
        data = np.array(data)
        for i in range(ncol):
            data[:, i] = remove_lm(data[:, i])
        new_df = pd.DataFrame(data, columns=cols_)
        return new_df


########################################################################################################################
# a function to make multiple-space-delimiter txt to comma-delimiter txt
# the values in the file should align with the columns: ind_list is the list of indexes that align to columns
# in UA data: [0, 20, 31, 42, 44, 56, 67, 78, 89]
def mts_to_csv(addr, filename, ind_list, rep='N'):
    def check_blank(line, index):
        if line[index] == " ":
            # if it is blank
            return True
        else:
            return False

    def str_assignment(str, value, index):
        ls = list(str)
        ls[index] = value
        new_str = ''.join(ls)
        return new_str

    def fillin_na(line, ind_list):
        for i, ind in enumerate(ind_list):
            if check_blank(line, ind):
                line = str_assignment(line, rep, ind)
        return line

    # read input file
    fin = open(addr + "\\" + filename, "rt")
    # target filw
    fout = open(addr + "\\" + filename[:-4] + "_comma.txt", "wt")

    # write
    for i, line in enumerate(tqdm(fin)):
        # first is header
        if i == 0:
            cols = line
            length = len(cols.split())
            fout.write(",".join(line.split()) + "\n")
        else:
            if len(line.split()) != length:
                line = fillin_na(line, ind_list)
                # change N to NA
                line = line.replace("N", "NA")
                fout.write(",".join(line.split()) + "\n")

    fin.close()
    fout.close()
    print("Done")


########################################################################################################################
# a function for hypsometric equation


########################################################################################################################
# print something at the end, also show figures if true
def end_of_code(start_time, show_plots=True):
    def show_copyright():
        print(
            "Please be reminded that all data used are subject to the intellectual property rights owned by Hong Kong Observatory (HKO).")

    def show_notes_byAuthor():
        print("Notes: No data were made by HKO during Japanese occupation (1940-46).")
        print("       Mean values were used to replace NAs when plotting linear trends.")

    def show_time(start_time):
        stop_time = time.time()
        print(f"End of code. (Elapsed time: {stop_time - start_time:.1f}s)")

    print("\n")
    show_notes_byAuthor()
    show_copyright()
    show_time(start_time)
    if show_plots:
        plt.show()
