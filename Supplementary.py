########################################################################################################################
# This file was used for updating KSC SR data.
########################################################################################################################
import numpy as np
import pandas as pd
from modules import *

dt = pd.read_csv('Data\\SR_KSC_read.csv', skiprows=3)
dt = np.array(dt)[0:-4, ]
tp_yr = np.linspace(2008, 2020, 13, dtype=int)
tp_m = np.linspace(1, 12, 12, dtype=int)
results = np.zeros((len(tp_yr), 12))

for i, x in enumerate(tp_yr):
    for j, y in enumerate(tp_m):
        yr_dt = dt[dt[:, 0].astype(int) == int(x)]
        myr_dt = yr_dt[yr_dt[:, 1].astype(int) == int(y)]

        results[i, j] = np.nanmean(myr_dt[:, 3].astype(float))
results = pd.DataFrame(results)
results.to_csv("Data\\SR_tp_K.csv")

########################################################################################################################
# Compare temp and SR
from modules import *

mSR_ = pd.read_csv('Data\\SR_KP.csv', skiprows=1)  # same as annual_all_hk[2]
HKO = pd.read_csv('Data\\byStation\\HKO.csv')
avg_T = HKO['avg_T']
# prepare data for plotting and calculation:
x = np.array(mSR_['Mean'])
y = np.array(avg_T[-len(x):])
# do PL
pl_mtx = pl(plx=x, ply=y, deg=2, cov=True)
pl_tp, pl_coef, pl_cov = pl_mtx[0], pl_mtx[1], pl_mtx[2]  # note: coef to be poly1d!
pl_eq = np.poly1d(pl_coef)
pl_x = np.linspace(np.min(x), np.max(x), 1000)
pl_pred = pl_eq(pl_x)
pl_ssr = np.polyfit(x, y, 2, full=True)[1][0]

# # PL2: red line
# pl_mtx2 = pl(plx=y, ply=x, deg=2, cov=True)
# pl_tp2, pl_coef2, pl_cov2 = pl_mtx2[0], pl_mtx2[1], pl_mtx2[2]  # note: coef to be poly1d!
# pl_eq2 = np.poly1d(pl_coef2)
# pl_x2 = np.linspace(np.min(y), np.max(y), 1000)
# pl_pred2 = pl_eq(pl_x2)
# pl_ssr2 = np.polyfit(y,x,2,full=True)[1][0]

# do LM
lm_mtx = lm(lmx=x, lmy=y)
lm_tp, lm_coefs = lm_mtx[0], lm_mtx[1]
lm_pred = [lm_coefs[1] + x * lm_coefs[2] for x in pl_x]
lm_ssr = lm_coefs[3]

# print pearson r
pr_stat = stats.pearsonr(x=x, y=y)
print(f'Pearson r = {pr_stat[0]}, p-value = {pr_stat[1]}')
sr_stat = stats.spearmanr(a=x, b=y)
print(f'Spearman rank r = {sr_stat[0]}, p-value = {sr_stat[1]}')

# plot
fig = plt.figure()
fig.set_size_inches(8.8, 7.5)
plt.locator_params(axis='both', nbins=6)
plt.scatter(x, y, marker='o', c='white', edgecolor='k')
plt.plot(pl_x, lm_pred, c='k', ls='-', label='Linear regression line')
plt.plot(pl_x, pl_pred, c='k', ls='--', label='Polynomial (deg.=2)')
# plt.plot(pl_x2,pl_pred2)  # TODO Can I add red line for vertical poly?? see PL2
plt.ylabel(r'Annual mean temperature [$\degree$C]')
plt.xlabel('Annual mean daily global solar radiation [MJ/m$^2$]')
plt.text(15.2, 23.45, fr'R$^2$={lm_coefs[0]:.2f}')
plt.text(15.2, 23.35, f'SSR={lm_ssr:.2f}')
plt.text(15.7, 23, f'SSR={pl_ssr:.2f}')
plt.arrow(16.27, 22.98, -0.2, -0.07, head_width=0.03)
plt.arrow(15.63, 23.33, 0, -0.07, head_width=0.03)
plt.legend()
plt.tight_layout()
fig.savefig('plots\\SRvsT.jpg', dpi=300)

########################################################################################################################
# Compute 30year avg soil temperature
from modules import *

soil_t = pd.read_csv('Data\\others\\soil_hko_kp.csv')
soil_t_mtx = np.zeros((2, 12, 7 * 2))
sta = ['HKO', 'KP']
# compute:
for i, station in enumerate(soil_t_mtx):
    for j, month in enumerate(station):
        for k, type in enumerate(month):
            # select recent 30 year
            dt = soil_t[soil_t['yyyy'] >= 1991]
            # select station:
            dt = dt[dt['station'] == sta[i]]
            # compute
            month[k] = np.nanmean(dt[dt['mm'] == j + 1].iloc[:, 3 + k])
dt1 = pd.DataFrame(soil_t_mtx[0],
                   index=[f'{i}' for i in range(1, 13)],
                   columns=soil_t.columns[3:] + sta[0])
dt1.to_csv(f'Data\\others\\soil_{sta[0]}_1991-2020.csv')
dt2 = pd.DataFrame(soil_t_mtx[1],
                   index=[f'{i}' for i in range(1, 13)],
                   columns=soil_t.columns[3:] + sta[1])
dt2.to_csv(f'Data\\others\\soil_{sta[1]}_1991-2020.csv')

########################################################################################################################
# make grass temperature mo to yr
from modules import *

dt = pd.read_csv('Data\\others\\grass_hko_kp_ksc_tkl_tms_monthly.csv')
len_yr = int(len(dt) / 12)
year = np.linspace(dt['yyyy'].iloc[0], 2020, len_yr)
mtx_ = np.zeros((len_yr, len(dt.columns) - 1))
cols = dt.columns[dt.columns != 'mm']
assert len(cols) == 5 + 1
mtx_[:, 0] = year
for i, x in enumerate(mtx_):
    for j, y in enumerate(x):
        if j == 0:
            pass
        else:
            # yearly mean: 1-12 mo
            select_yr = dt[dt['yyyy'] == year[i]]
            # start from index 2: (0==year, 1==mo)
            select_station = select_yr.iloc[:, j + 1]
            assert select_station.name == cols[j]
            if np.all(np.isnan(select_station)):
                x[j] = np.nan
            elif np.sum(np.isnan(select_station)) >= 6:
                # if half of data is na
                x[j] = np.nan
            else:
                x[j] = np.nanmean(select_station)
# to dataframe and save
df = pd.DataFrame(mtx_, columns=cols)
df.to_csv('Data\\grass_T_yr.csv', index=False, na_rep='NA')

########################################################################################################################
# make soil temperature to yearly
from modules import *

st = pd.read_csv('Data\\others\\soil_hko_kp.csv')
stations = ['HKO', 'KP']
year_hko = np.linspace(1968, 2020, 2020 - 1968 + 1)
mtx_hko = np.zeros((len(year_hko), 15))
cols = st.columns[st.columns != 'mm']
cols = cols[cols != 'station']
mtx_hko[:, 0] = year_hko
for i, x in enumerate(mtx_hko):
    for j, y in enumerate(x):
        if j == 0:
            pass
        else:
            # yearly mean: 1-12 mo
            select_yr = st[st['yyyy'] == year_hko[i]]
            # start from index 2: (0==year, 1==mo)
            select_station = select_yr[select_yr['station'] == stations[0]]
            assert np.all(select_station['station'] == 'HKO')
            select_type = select_station.iloc[:, j + 2]
            assert select_type.name == cols[j]
            if np.all(np.isnan(select_type)):
                x[j] = np.nan
            elif np.sum(np.isnan(select_type)) >= 6:
                # if half of data is na
                x[j] = np.nan
            else:
                x[j] = np.nanmean(select_type)
df1 = pd.DataFrame(mtx_hko, columns=cols)
assert len(df1.iloc[:, 0]) == len(st[st['station'] == 'HKO'].iloc[:, 0]) / 12
# repeat for kp
year_kp = np.linspace(1978, 2020, 2020 - 1978 + 1)
mtx_kp = np.zeros((len(year_kp), 15))
mtx_kp[:, 0] = year_kp
for i, x in enumerate(mtx_kp):
    for j, y in enumerate(x):
        if j == 0:
            pass
        else:
            # yearly mean: 1-12 mo
            select_yr = st[st['yyyy'] == year_kp[i]]
            # start from index 2: (0==year, 1==mo)
            select_station = select_yr[select_yr['station'] == stations[1]]
            assert np.all(select_station['station'] == 'KP')
            select_type = select_station.iloc[:, j + 2]
            assert select_type.name == cols[j]
            if np.all(np.isnan(select_type)):
                x[j] = np.nan
            elif np.sum(np.isnan(select_type)) >= 6:
                # if half of data is na
                x[j] = np.nan
            else:
                x[j] = np.nanmean(select_type)
df2 = pd.DataFrame(mtx_kp, columns=cols)
assert len(df2.iloc[:, 0]) == len(st[st['station'] == 'KP'].iloc[:, 0]) / 12

# save df1 df2
df1.to_csv('Data\\soil_T_yr_HKO.csv', index=False, na_rep='NA')
df2.to_csv('Data\\soil_T_yr_KP.csv', index=False, na_rep='NA')

########################################################################################################################
# scraping rf data from HKO website:
from modules import *
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from tqdm import tqdm


# note: webdrive obtained from https://github.com/mozilla/geckodriver/releases
# current geckodriver version: 0.29.1 for win64
# note 2: open firefox first

def simple_scrapper(start_yr, end_yr, base_url, file_name):
    start, end = start_yr, end_yr
    # a for loop for scrape data from 1884-2020
    year = np.linspace(start, end, end - start + 1, dtype=int)
    for loop, yr in enumerate(tqdm(year)):
        # target link
        url = f'{base_url}{yr}'
        # use selenium
        driver = webdriver.Firefox()
        driver.get(url)
        # take time to load page
        time.sleep(5)
        try:
            soup = BeautifulSoup(driver.page_source)
            table = soup.find(lambda tag: tag.name == 'table' and tag.has_attr('id') and tag['id'] == 't1')
            rows = table.findAll(lambda tag: tag.name == 'tr')
        except:
            time.sleep(5)
            soup = BeautifulSoup(driver.page_source)
            table = soup.find(lambda tag: tag.name == 'table' and tag.has_attr('id') and tag['id'] == 't1')
            rows = table.findAll(lambda tag: tag.name == 'tr')
        # store into dataframe:
        df_ = np.zeros((len(rows) - 1, len(rows[0])), dtype=np.float)
        if 1940 <= yr <= 1946:
            # WWII period, no data available
            cols = np.array(rows[0], dtype=object).reshape(1, -1)[0]
            cols[cols == '\xa0'] = 'Day'
            df_[:] = np.nan
        else:
            for i, x in enumerate(rows):
                if i == 0:
                    # horizontal month list
                    cols = np.array(x, dtype=object).reshape(1, -1)[0]
                    cols[cols == '\xa0'] = 'Day'
                else:
                    dt = np.array(rows[i], dtype=object).T[0]
                    # remove Trace -> 0.05
                    # TODO add checking, check with 1900
                    if np.any(dt=='Trace') or np.any(dt == '***'):
                        dt[dt == 'Trace'] = 0.05
                        dt[dt == '***'] = np.nan
                    if np.any(dt == '\xa0'):
                        # empty cells -> na
                        dt[dt == '\xa0'] = np.nan
                    df_[i - 1, :] = dt.astype(np.float)
        df = pd.DataFrame(df_, columns=cols)
        df.to_csv(f'Data\\others\\{file_name}\\{yr}.csv')
        # close drive for re-open
        driver.close()


# Note: can modify to a function and scrape other data if wanted. Please be aware of data policy of HKO.

# # mean temp
# simple_scrapper(1884,1900,'https://www.hko.gov.hk/en/cis/dailyElement.htm?ele=TEMP&y=','temperature\\avg')
# # 1900 bugged
# simple_scrapper(1901,2020,'https://www.hko.gov.hk/en/cis/dailyElement.htm?ele=TEMP&y=','temperature\\avg')
#
# # max temp
# simple_scrapper(1884,1900,'https://www.hko.gov.hk/en/cis/dailyElement.htm?ele=MAX_TEMP&y=','temperature\\max')
# # 1900 bugged
# simple_scrapper(1901,2020,'https://www.hko.gov.hk/en/cis/dailyElement.htm?ele=MAX_TEMP&y=','temperature\\max')
#
# # min temp
# simple_scrapper(1884,1900,'https://www.hko.gov.hk/en/cis/dailyElement.htm?ele=MIN_TEMP&y=','temperature\\min')
# # 1900 bugged
# simple_scrapper(1901,2020,'https://www.hko.gov.hk/en/cis/dailyElement.htm?ele=MIN_TEMP&y=','temperature\\min')

########################################################################################################################
# use web scrapper to make a list of HKO publications
from modules import *
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import requests
from datetime import datetime

# function list
def get_elements(url, tag, cls=False, return_soup=False):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    ele = soup.find_all(tag, {'class': cls} if cls else None)
    if return_soup:
        return ele, soup
    return ele

# a function to find nrow ncol col_name
def get_rows_cols(url, cls=False):
    # read rows and cols:
    table = get_elements(url, "table", cls=cls)[0]
    if table.find_all('tbody'):
        tbody = table.find_all('tbody')[0]
    else:
        tbody = table
    trs = tbody.find_all('tr')
    rows = len(trs) - 1
    ths = trs[0].find_all('th')
    cols = len(ths)
    col_names = [txt.text for txt in ths if txt.text]
    for i in range(len(col_names)):
        for replacement in ['\r','\n',' ']:
            col_names[i] = col_names[i].replace(replacement, "")
    return rows, cols, col_names

def reformat(obj_arr):
    # list to remove
    list = ['\n', '\r', '  ']
    for i,x in enumerate(obj_arr):
        for j,y in enumerate(x):
            for el in list:
                try:
                    obj_arr[i,j] = obj_arr[i,j].replace(el, '')
                except:
                    # do nothing
                    pass
    return obj_arr

def check_hyperlink(trs):
    # pass all rows to check whether link is attached with text
    for elements in trs:
        link = elements.find('a')
        if link is not None:
            if link.text is not '':
                return True
    #print("Link along")
    return False

def get_publications(url, links_cls, filename, table_cls="data_table", show_data=True):
    print("Start scraping files...")
    if links_cls is None:
        if len(url) is not 1:
            url = list([url])
        else:
            pass
    else:
        # store all links with .htm ending
        elements = get_elements(url, "table", cls=links_cls)[0]
        rp_links = elements.find_all('a')
        reprint_urls = [a['href'].replace(shortened_dir, parent_dir) for a in rp_links]
        filtered_rp_urls = list(filter(lambda x: x.endswith('.htm'), reprint_urls))
        # add 1st page also
        url = list([url])
        url.extend(filtered_rp_urls)

    # read data
    all_in_one = []
    for n, link in enumerate(tqdm(url)):
        rows, cols, col_names = get_rows_cols(link, cls=table_cls)
        # print(col_names)
        # read info
        list_ = np.zeros(rows * cols, dtype=object)
        table = get_elements(link, "table", cls=table_cls)[0]
        tds = table.find_all("td")
        for i, el in enumerate(tds):
            # print(el.text, end='\n')
            if el.text.replace('\n',''):
                # if removed '\n' still has txt, it is title or something
                list_[i] = el.text
            #elif el.text=='':
            else:
                # check if "", it will be link with img (or itself)
                lk_ = el.find('a')['href']
                list_[i] = lk_.replace(shortened_dir, parent_dir)

        # reshape list here
        list_ = list_.reshape(rows, cols)
        col_names = list(col_names)
        # check whether link with text or along
        trs = table.find_all("tr")[1:]  # ignore title
        hyperl_exist = check_hyperlink(trs)
        if hyperl_exist:
            lks_ = np.zeros(rows, dtype='object')
            for i, el in enumerate(trs):
                if trs[i].find('a') is not None:
                    lks_[i] = trs[i].find('a')['href']
                    # replace with full link
                    lks_[i] = lks_[i].replace(shortened_dir, parent_dir)
                else:
                    lks_[i] = "-"
            # combine list with lks if hyperlink:
            list_ = np.append(list_,lks_.reshape(-1,1), axis=1)
            # renew col names
            col_names.extend(['Link'])
        # reformat list
        list_ = reformat(list_)
        if n == 0:
            all_in_one = pd.DataFrame(list_, columns=col_names)
        else:
            df = pd.DataFrame(list_, columns=col_names)
            all_in_one = all_in_one.append(df)

    # sort list
    tp_c = [f'col{n}' for n in range(len(col_names))]
    copy_ = all_in_one.copy()
    copy_.columns = tp_c
    # turn no. (object) into no. (int)
    try:
        copy_['col0'] = copy_['col0'].astype(int)
        copy_ = copy_.sort_values(by='col0',axis=0)
        all_in_one = copy_.copy()
        all_in_one.columns = col_names
    except:
        pass
    if show_data:
        print(all_in_one)
    all_in_one.to_csv(f"HKO_pub\\HKO_{filename}.csv", encoding="utf_8_sig", index=False)
    print("Completed scraping files.")
    print("########################################################################################################################")

# read all files and store into one excel database
def combine_files(list_filename, titles):
    def selection(cols):
        c = None
        if cols == 5:
            c = criteria_[0]
        elif cols == 4:
            c = criteria_[1]
        elif cols == 6:
            c = criteria_[2]
        return c
    print("Start merging files...")
    all_in_one = []
    type_ls = []
    for i,file in enumerate(tqdm(list_filename)):
        df = pd.read_csv(f'HKO_{file}.csv')
        df.columns = [f'col{n}' for n in range(len(df.columns))]
        rows, cols = df.shape
        # select certain info
        c = selection(cols)
        filtered_df = df[c]
        # reset title
        if cols == 4:
            if file == 'op':
                filtered_df.columns = ['標題','作者','期刊/年份']
            else:
                filtered_df.columns = ['標題','期刊/年份','超連結(url)']
        else:
            filtered_df.columns = titles[2:]
        # create 2nd col: types
        type = np.zeros(rows,dtype=object)
        type[:] = types_[i]
        if i == 0:
            all_in_one = filtered_df.copy()
            type_ls = type.copy()
        else:
            # TODO will append according to col names?
            all_in_one = all_in_one.append(filtered_df, ignore_index=True)
            type_ls = np.append(type_ls, type)
    rows, cols = all_in_one.shape
    num_ = pd.DataFrame(np.linspace(1,rows,rows,dtype=int))
    type_ls = pd.DataFrame(type_ls.reshape(-1,1), columns=list(['type']))
    # combine all
    temp_ = pd.concat([type_ls,all_in_one], axis=1)
    results_ = pd.concat([num_,temp_], axis=1)
    results_.columns = titles.copy()
    # replace '\n' and empty
    results_ = pd.DataFrame(reformat(np.array(results_)), columns=titles.copy())
    # # add two lines AFTER reformat
    # new_lines = add_empty_lines(3, cols+2,
    #                             caption="資料出處: 香港天文台（https://www.hko.gov.hk/tc/publica/publist.htm）",
    #                             update="HKT "+f"{datetime.now().strftime('%X %B %d, %Y')}")
    # # combine them
    # new_lines.columns = titles
    # results_ = new_lines.append(results_)
    results_.to_csv(f"HKO_pub\\HKO_all_list({datetime.now().strftime('%d-%b-%Y')}).csv",encoding='utf-8-sig', index=False, na_rep='---')
    print("Completed merging files.")
    print("########################################################################################################################")

# no used
def add_empty_lines(nlines, cols, caption=False, update=False):
    re_ = np.zeros((nlines, cols), dtype=object)
    re_[:] = "  "
    if caption:
        re_[0,0] = caption
    if update and nlines>1:
        re_[1,0] = "Last update: "+ update
    return pd.DataFrame(re_)

# some configs:
mother_list = '/tc/publica/publist.htm'
parent_dir = "https://www.hko.gov.hk"
shortened_dir = "../.."
# read urls and store:
elements_ = get_elements(parent_dir + mother_list, "a")
urls_ = [a['href'].replace(shortened_dir, parent_dir) for a in elements_ if a.text]
filtered_urls_ = list(filter(lambda x: x.endswith('.htm'), urls_))
types_ = [t.text for t in elements_][-len(filtered_urls_):]
links_cls_ = [None, None, None,
              "pubreprint_page_list_100", "pubreprint_page_list", None,
              "pubreprint_page_list", None, None]
filenames_ = ['gen','dt','tm','TN','TNlocal','rdTN','rp','hp','op']
# criteria for select title, author, year, link
criteria_ = [['col1','col2','col3','col4'],
             ['col1','col2','col3'],
             ['col1','col2','col3','col5']]
titles_ = ['項目', '類別', '標題', '作者', '期刊/年份', '超連結(url)']

# Main dish: For (1)-(7), except (2)
for i,url in enumerate(filtered_urls_):
    try:
        get_publications(url,links_cls_[i],filenames_[i],show_data=False)
    except:
        pass
# lastly, store into one
combine_files(filenames_, titles_)
# done.

########################################################################################################################
# pdf to searchable pdf: OCR
import ocrmypdf
import os
from modules import *
# see https://ocrmypdf.readthedocs.io/en/latest/installation.html#installing-on-windows
# if __name__ == '__main__':
#     ocrmypdf.ocr('peacock_Hong Kong Meteorological Records and Climatological Notes 60 years 1884-1939, 1947-1950.pdf',
#                  'output.pdf',
#                  deskew=True)
def trans_pdf2OCR(dir,deskew=False, skip_text=False, redo_ocr=False, save_dir=None):
    pdf_list = os.listdir(dir)
    if save_dir:
        pass
    else:
        os.mkdir(dir+'\\OCR')
        save_dir = dir+'\\OCR'
    for file in pdf_list:
        if os.path.isfile(save_dir+f"\\{file}"):
            pass
        else:
            # if original has txt, still do OCR, can textify images
            ocrmypdf.ocr(dir+f"\\{file}", save_dir+f"\\{file}", deskew=deskew, skip_text=skip_text,redo_ocr=redo_ocr)
    print("Done.")
# make OCR directors report
# trans_pdf2OCR(dir='',
#               save_folder="DR_searchable",
#               deskew=False,
#               skip_text=False,
#               redo_ocr=True)
# for air pollution monitoring results 69-80
year = np.linspace(1969, 1980, 12, dtype=int)
folder = "Air_pollution_monitoring_results_1969-80"
for i,x in enumerate(year):
    dir = folder +f"\\{x}"
    print(dir)
    trans_pdf2OCR(dir)

########################################################################################################################
# two figures: one for temp, one for rf, 1951-1980
# x = month, y = C or mm
from modules import *

def select_data(data, period):
    start_yr, end_yr = period
    _ = data.columns[0]
    # 2 layer selections
    s1_data = data[np.array(data[_], dtype=int) >= start_yr]
    f_data = s1_data[np.array(s1_data[_], dtype=int) <= end_yr]
    return f_data

plt.rcParams['font.family'] = cn_font
p = ['1951-1980','1991-2020']
x_l = ['一月','二月','三月','四月','五月','六月','七月','八月','九月','十月','十一月','十二月']

data = pd.read_csv('Data\\T_mo.csv')
cols = data.columns
# select only avg data
ind1, ind2 = np.where(data['Year']=='Avg')[0][0], np.where(data['Year']=='Min')[0][0]
data = pd.DataFrame(data.values[ind1+1:ind2,:], columns=cols)
# select 1951-1980
period = (1951, 1980)
df1 = select_data(data, period)
# select 1991-2020
period = (1991,2020)
df2 = select_data(data, period)

# bar plot
col_ = cols[np.logical_not(df1.columns=='Year')]
m1 = df1[col_].astype(float).mean(0)
print(f'Yearly mean of 1951-1980: {m1.mean():.1f}')
m2 = df2[col_].astype(float).mean(0)
print(f'Yearly mean of 1991-2020: {m2.mean():.1f}')
mean_ = pd.concat([m1,m2],axis=1)
# rename mean_ index to CN
mean_.index = x_l
diff_ = pd.DataFrame(np.array(m2) - np.array(m1), index=mean_.index, columns=['兩段時期之溫度差'])
mean_.columns = p
# bar plot
fig,ax = plt.subplots()
mean_.plot.bar(y=p,ax=ax, xlabel='月份', ylabel=f'溫度 [{t_unit}]', figsize=(fwidth, fheight))
ax.set_ylim((10,30))
ax2 = ax.twinx()
diff_.plot(ax=ax2, c='k', ylabel=f'差距 [{t_unit}]',legend=False,marker=marker)
ax2.legend(loc='upper left')
ax2.set_ylim((0.2,1.4))
fig.tight_layout()
fig.savefig('plots\\T_mo_5180-9120mean.jpg',dpi=300)

# rf
data = pd.read_csv('Data\\rf_mo.csv')
cols = data.columns
period = (1951,1980)
df1 = select_data(data, period)
period = (1991,2020)
df2 = select_data(data, period)
col_ = cols[np.logical_not(cols=='Year')]
m1 = df1[col_].mean(axis=0)
print(f'Yearly mean of 1951-1980: {m1.mean():.1f}')
m2 = df2[col_].mean(axis=0)
print(f'Yearly mean of 1991-2020: {m2.mean():.1f}')
mean_ = pd.concat([m1,m2], axis=1)
mean_.columns = p
mean_.index = x_l
diff_ = pd.DataFrame(np.array(m2)-np.array(m1), index=mean_.index, columns=['兩段時期之降雨差'])
# bar plot
#plt.rcParams['font.size'] = 100
fig,ax = plt.subplots()
mean_.plot.bar(y=p, ax=ax, xlabel='月份', ylabel=f'月總降雨 [毫米]', figsize=(fwidth, fheight))
ax2 = ax.twinx()
diff_.plot(ax=ax2, legend=False, c='k', marker=marker, ylabel='差距 [毫米]')
ax2.legend(loc='upper left')
fig.tight_layout()
fig.savefig('plots\\rf_mo_5180-9120mean.jpg',dpi=300)

########################################################################################################################
# handle KP 1992 discrepancy
from modules import *

def select_data(data, period):
    start_yr, end_yr = period
    _ = data.columns[0]
    # 2 layer selections
    s1_data = data[np.array(data[_], dtype=int) >= start_yr]
    f_data = s1_data[np.array(s1_data[_], dtype=int) <= end_yr]
    return f_data

def fillNA_by_comparison(data, period):
    # only works for KP HKO, for flexibility, need improvement
    # show fig
    fig = plt.figure()
    plt.scatter(data['Year'], data['KP'], c='white', marker='o', edgecolors='k', label='KP')
    plt.gca().set(
        xlabel='Year',
        ylabel=f'Temperature {t_unit}'
    )
    plt.legend()
    fig.tight_layout()

    d_ = select_data(data, period)
    d_ = np.array(d_)
    d_[np.where(d_[:,0]==1992)[0][0],2:] = np.nan
    lm_ = lm(lmx=d_[:,1], lmy=d_[:,2])
    int_, slope_ = lm_[1][1], lm_[1][2]
    _ = np.linspace(np.nanmin(d_[:,1]), np.nanmax(d_[:,1]))
    pred_ = int_ + slope_ * _
    print(f'Equation: y = {int_:.4f} + {slope_:.4f}x')
    # show fig
    fig = plt.figure()
    plt.scatter(d_[:, 1], d_[:, 2], c='white', marker='o', edgecolors='k', label=f'Temperature {t_unit}')
    str = f'+{int_:.2f}, r2={lm_[1][0]:.2f}' if int_>0 else f'{int_:.2f}, r2={lm_[1][0]:.2f}'
    plt.plot(_, pred_, c='k', label=f'LM, y={slope_:.2f}x' + str)
    # plt.plot(data[:,1], pl_[0], c='k', label='PL (deg.=2)')s
    plt.gca().set(
        xlabel='HKO',
        ylabel='KP',
        title=f'{period[0]}-{period[1]}'
    )
    plt.legend()
    plt.tight_layout()

    # get 1992 pred KP value
    value = int_ + slope_ * d_[np.where(d_[:, 0] == 1992)[0][0], 1]
    print(f'Predicted KP 1992 = {value:.3f} {t_unit}')

    # compute 1991-2020 rate
    d_ = np.array(data.copy())
    d_[np.where(d_[:, 0] == 1992)[0][0], 2:] = value
    d_ = d_[d_[:,0]>=1991]
    lm_ = lm(lmx=d_[:,0], lmy=d_[:,2])
    print(f'Adjusted rate (C/100yr): {lm_[1][2]*100:.2f}, SE: {lm_[1][4][1]*100:.2f}')

    # plot 1991-2020 and LM
    plt.figure()
    plt.plot(d_[:,0], d_[:,2],c='k')
    plt.plot(d_[:,0], lm_[0], c='k', ls='--', label=F'y={lm_[1][1]:.2f}+{lm_[1][2]:.2f}x')
    plt.gca().set(
        xlabel='HKO',
        ylabel='KP',
        title=f'1991-2020 with adjusted 1992'
    )
    plt.legend()
    plt.tight_layout()

# method 1: HKO vs KP
data = pd.read_csv('Data\\Temperature_max.csv')
data = data[['Year','HKO','KP']]
fillNA_by_comparison(data=data, period=(1985,1999))

data = pd.read_csv('Data\\Temperature.csv')
data = data[['Year','HKO','KP']]
fillNA_by_comparison(data=data, period=(1985,1999))

data = pd.read_csv('Data\\Temperature_min.csv')
data = data[['Year','HKO','KP']]
fillNA_by_comparison(data=data, period=(1985,1999))

# method 2:
data = pd.read_csv('Data\\Temperature_max.csv')
data = data[['Year','HKO','KP']]
data = data.values[-30:]
lm_ = lm(lmx=data[:,0], lmy=data[:,2])
print(f'100 year rate: {lm_[1][2]*100:.2f}, SE: {lm_[1][4][1]*100:.2f}')

data = pd.read_csv('Data\\Temperature.csv')
data = data[['Year','HKO','KP']]
data = data.values[-30:]
lm_ = lm(lmx=data[:,0], lmy=data[:,2])
print(f'100 year rate: {lm_[1][2]*100:.2f}, SE: {lm_[1][4][1]*100:.2f}')

data = pd.read_csv('Data\\Temperature_min.csv')
data = data[['Year','HKO','KP']]
data = data.values[-30:]
lm_ = lm(lmx=data[:,0], lmy=data[:,2])
print(f'100 year rate: {lm_[1][2]*100:.2f}, SE: {lm_[1][4][1]*100:.2f}')


########################################################################################################################
# check outliers
from modules import *
ls = ['Temperature_max.csv','Temperature.csv','Temperature_min.csv']
for i,file in enumerate(ls):
    data = pd.read_csv('Data\\'+file)
    data = data[data['Year'] >= 1991]
    stations = data.columns
    data = data[stations[1:]]
    sb_boxplot(data=data, title=t_name[i])
    # detrend
    data = detrend(data)
    sb_boxplot(data=data, title="Detrended: "+t_name[i])

# found maxT: WGL, HKS; avgT: NONE; minT: TKL
data = pd.read_csv('Data\\'+ls[0])
data = data[data['Year'] >= 1991]
stations = data.columns
year = data['Year']
# plot max T
fig = plt.figure()
# plt.plot(year, data['WGL'], c='green', label='WGL')
# lm_ = lm(year, data['WGL'])
# plt.plot(year, lm_[0],c='green')
plt.plot(year, data['HKS'], c='pink', label='HKS')
lm_ = lm(year, data['HKS'])
plt.plot(year, lm_[0], c='pink')
plt.title('Max T')
plt.legend()
plt.tight_layout()

# min T: TKL
# found maxT: WGL, HKS; avgT: NONE; minT: TKL
data = pd.read_csv('Data\\'+ls[2])
data = data[data['Year'] >= 1991]
stations = data.columns
year = data['Year']
# plot max T
fig = plt.figure()
plt.plot(year, data['TKL'], c='brown', label='TKL')
lm_ = lm(year, data['TKL'])
plt.plot(year, lm_[0],c='brown')
plt.title('Min T')
plt.legend()
plt.tight_layout()

########################################################################################################################
# compute yearly mean RH
from modules import *

address = input(f'Input address: ')
data1 = pd.read_csv(address+'\\RH_47-53.csv')
data2 = pd.read_csv(address+'\\RH_54-60.csv')
data = data1.append(data2, ignore_index=True)
year = np.linspace(1947, 1960, 14,dtype=int)
results = np.zeros((len(year), 2))
results[:] = np.nan
results[:,0] = year
for i,x in enumerate(results):
    select_ = data[data['yyyymmdd']//10000 == year[i]]
    print(f'year is {np.median(select_["yyyymmdd"])}')
    length_ = len(select_)
    results[i,1] = np.sum(select_['RH']) / length_
df_ = pd.DataFrame(results, columns=['Year','RH'])
df_.to_csv(address+'\\Yearly_RH_47-60.csv',index=False)

########################################################################################################################
# handle HKO output flight docs for mr shun
from modules import *
from  matplotlib import cm
cmap = cm.get_cmap('tab20b')
add_ = input(f'Input address and file name: ')
data = pd.read_csv(add_)
# all data to 10k
data[data.columns[1:]] = data[data.columns[1:]].div(10000)

list = ['Year','Total','From_DR']
list_cn = ['年份','天文台可用數據','擷取自天文台長報告']
selection = data[list]
# to 2017
selection = selection[selection['Year'] <= 2017]
selection.columns = list_cn

#plot
plt.rcParams['font.family'] = cn_font
show_legend=True
fig, ax = plt.subplots()
fig.set_size_inches(fwidth, fheight)
c = ['b','r','orange']
# selection.plot.bar(x='年份', ax=ax, edgecolor='k', zorder=3)
for i,name in enumerate(selection.columns[1:]):
    selection[[selection.columns[0], name]].plot.bar(x=selection.columns[0], zorder=3, ax=ax, color=cmap(0.1+i*0.15), edgecolor='k', width=0.65, legend=show_legend)
ax.grid(zorder=0)
ax.set_ylabel('飛行文件數目 [萬]')
ax.set_xlabel('年份')
if show_legend:
    ax.legend(edgecolor='k')
plt.locator_params(axis='x', nbins=12)
fig.tight_layout()
# fig.savefig(f'1948-2017天文台發出飛行文件數目_ver3.jpg', dpi=300)

# plot stacked bar char
monthly= data[data.columns[:-1]]
monthly.columns =  ['年份','一月','二月','三月','四月','五月','六月','七月','八月','九月','十月','十一月','十二月']
# plot
from  matplotlib import cm
cmap = cm.get_cmap('tab20b')
fig,ax = plt.subplots()
fig.set_size_inches(12, 8)
monthly.plot.bar(x='年份', ax=ax, stacked=True, edgecolor='k', zorder=3, cmap=cmap)
ax.grid(zorder=0)
ax.set_ylabel('飛行文件數目 [件]')
ax.set_xlabel('年份')
ax.set_title('天文台爲離港航班提供飛行文件數目')
ax.legend(bbox_to_anchor=(1.01,1.01), edgecolor='k')
fig.tight_layout()

########################################################################################################################
# AMIDS data
from modules import *
from  matplotlib import cm
cmap = cm.get_cmap('tab20b')

addr = input('Input address: ')
filename = input('Input filename:')
data = pd.read_csv(addr+filename)
# select until 2017
data = data[data['Year'] <= 2017]

# fig 1 bar chart of no. of visit
plt.rcParams['font.family'] = cn_font
fig,ax = plt.subplots()
fig.set_size_inches(8, 6)
temp_ = data[['Year', 'Number_of_visit']]
values = temp_.iloc[:,1]/1e8
temp_.iloc[:,1] = values
temp_.columns = ['年份', '年總和']
temp_.plot.bar(x='年份',ax=ax, edgecolor='k', zorder=3, legend=False, width=0.7, cmap=cmap)
ax.ticklabel_format(axis='y',useMathText=True)
ax.grid(zorder=0)
ax.set_ylabel('瀏覽量 [億]')
ax.set_xlabel('年份')
ax.set_title('「航空氣象資料發送系統」瀏覽量')
#ax.legend(edgecolor='k')
fig.tight_layout()
# fig.savefig('瀏覽量.jpg', dpi=300)

# # fig 2 bar chart and line plot of visit and orgs
# fig,ax = plt.subplots()
# fig.set_size_inches(11, 7)
# temp_ = data[['Year','Number_of_visit','Number_of_orgs']]
# temp_['Number_of_visit'] = temp_.iloc[:,1]/1e8
# temp_.columns = ['年份','總瀏覽量','用戶數目']
# p1 = ax.bar(x=temp_['年份'].astype(object),height=temp_['總瀏覽量'], edgecolor='k', zorder=3, label='總瀏覽量')
# ax2 = ax.twinx()
# p2 = ax2.plot(temp_['年份'].astype(object),temp_['用戶數目'], c='k', marker='x', label='用戶數目')
# ax.grid(zorder=0)
# ax.set_ylabel('瀏覽量 [億]')
# ax.set_xlabel('年份')
# ax.set_title('「航空氣象資料發送系統」瀏覽量')
# ax2.set_ylabel('用戶數目')
# ls = [p1]+p2
# ax2.legend(ls,[l.get_label() for l in ls], edgecolor='k')
# fig.tight_layout()

########################################################################################################################
# HKO Online info visit number
# unit: raw data == millions
from modules import *
from  matplotlib import cm
#from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
cmap = cm.get_cmap('tab20b')

addr = input('Input address: ')
filename = input('Input filename:')
data = pd.read_csv(addr+filename)
data.iloc[:,1:] = data.iloc[:,1:]*1e6/1e8
# to 2017
data = data[data['Year'] <= 2017]

# plot
plt.rcParams['font.family'] = cn_font
fig,ax = plt.subplots()
fig.set_size_inches(11, 7)
temp_ = data[['Year','mobile','web']]
temp_.columns = ['年份','我的天文台','天文台網站']
temp_.plot.bar(x='年份', ax=ax, stacked=False, zorder=3, edgecolor='k', cmap=cmap, width=0.8)
plt.yscale('log')
ax.grid(zorder=0)
ax.legend(edgecolor='k')
ax.set_ylabel('網頁數 [億]')
ax.set_title('天文台網上資訊服務瀏覽數字 (網頁數)')
# ax.yaxis.set_major_formatter(ScalarFormatter())  # it shows 0 for <1
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
fig.tight_layout()
# fig.savefig('天文台網上資訊服務瀏覽數字.jpg', dpi=300)

########################################################################################################################
# 3.8.2.2 rainfall
from modules import *

data = pd.read_csv('Data\\byStation\\HKO_20211016.csv')
data = data[['Year','rf']]

# plot data and lm
plt.rcParams['font.family'] = cn_font
fig = plt.figure()
fig.set_size_inches(fwidth, fheight)
plt.plot(data['Year'], data['rf'], c='b', marker='o')
plt.grid()
lm_ = lm(lmx=data['Year'], lmy=data['rf'])
plt.plot(data['Year'], lm_[0], c='b')
plt.ylabel('雨量 [毫米]')
fig.tight_layout()

# plot max 1 hr rf
data = pd.read_csv('Data\\others\\rf_max1hr.csv')
# new array to keep max
arr_ = np.zeros(len(data['Year']))
dt_ = data['Total']
for i,x in enumerate(arr_):
    if i ==0:
        arr_[i] = dt_[i]
    else:
        if dt_[i] > arr_[i-1]:
            arr_[i] = dt_[i]
            print(f"{data['Year'][i]}: {dt_[i]}")
        else:
            arr_[i] = arr_[i-1]

# plot
fig = plt.figure()
fig.set_size_inches(fwidth, fheight)
plt.plot(data['Year'][1:], arr_[1:], c='b')
plt.grid()
plt.ylabel('一小時雨量最高記錄 [毫米]')
plt.locator_params(axis='x', nbins=10)
fig.tight_layout()

########################################################################################################################
# mts to csv
# note that datafile is large
from modules import *

plt.rcParams['font.family'] = cn_font

addr = input('Address: ')
filename = input('Filename: ')
ind_list = [0, 20, 31, 42, 44, 56, 67, 78, 89]
mts_to_csv(addr, filename, ind_list)

data = pd.read_csv(addr+"\\"+filename[:-4]+"_comma.txt")

# compute yearly mean
start, end = 1957, 2020
year = np.linspace(start, end, end-start+1, dtype=int)
# desired pressure level
pls_ = [850, 700, 500, 400, 300, 250, 200, 150, 100, 50]
# 1st col year; 2nd mean geo-height; 3rd mean_T
ua_yearly = np.zeros((len(pls_), len(year), 3))
ua_yearly_08, ua_yearly_20 = np.zeros(ua_yearly.shape), np.zeros(ua_yearly.shape)
ua_yearly[:,:,0], ua_yearly_08[:,:,0], ua_yearly_20[:,:,0] = year, year, year

for i,p in enumerate(pls_):
    # select certain pressure level: raw data are 0.1 hPa
    selection1 = data[data['PRESSURE']//10 == p]
    for j,yr in enumerate(tqdm(year)):
        # then, select certain year
        selection2 = selection1[selection1['DTG']//1000000 == yr]
        h = selection2['HEIGHT']  # in m
        t = selection2['TEMP']  # in 0.1 C
        # assign mean values
        ua_yearly[i,j,1] = h.mean()
        ua_yearly[i,j,2] = t.mean()/10  # unit 0.1 C to C

        # only 08
        selection3 = selection2[selection2['DTG'] % 100 == 8]
        h = selection3['HEIGHT']  # in m
        t = selection3['TEMP']  # in 0.1 C
        # assign mean values
        ua_yearly_08[i,j,1] = h.mean()
        ua_yearly_08[i,j,2] = t.mean()/10  # unit 0.1 C to C

        # only 20
        selection4 = selection2[selection2['DTG'] % 100 == 20]
        h = selection4['HEIGHT']  # in m
        t = selection4['TEMP']  # in 0.1 C
        # assign mean values
        ua_yearly_20[i,j,1] = h.mean()
        ua_yearly_20[i,j,2] = t.mean()/10  # unit 0.1 C to C

# save these data to csv
# for i, dt in enumerate(ua_yearly):
#     addr = "C:\\Users\\Arthurli\\Desktop\\2021_HongKongChronicles\\Assignments\\Mr Lam CY\\data(confi)\\UA_alldata_1956_202109\\annual_mean"
#     df = pd.DataFrame(dt, columns=['Year','Geo-height','Temp'])
#     df.to_csv(addr+f"\\all_{pls_[i]}.csv", index=False)
# for i, dt in enumerate(ua_yearly_08):
#     addr = "C:\\Users\\Arthurli\\Desktop\\2021_HongKongChronicles\\Assignments\\Mr Lam CY\\data(confi)\\UA_alldata_1956_202109\\annual_mean"
#     df = pd.DataFrame(dt, columns=['Year','Geo-height','Temp'])
#     df.to_csv(addr+f"\\08_{pls_[i]}.csv", index=False)
# for i, dt in enumerate(ua_yearly_20):
#     addr = "C:\\Users\\Arthurli\\Desktop\\2021_HongKongChronicles\\Assignments\\Mr Lam CY\\data(confi)\\UA_alldata_1956_202109\\annual_mean"
#     df = pd.DataFrame(dt, columns=['Year','Geo-height','Temp'])
#     df.to_csv(addr+f"\\20_{pls_[i]}.csv", index=False)

def to_anomaly(data):
    data = np.array(data)
    m = np.nanmean(data)
    new_d = data - m
    return new_d

# plot data: 08
ua850 = ua_yearly_08[0]
fig = plt.figure()
fig.set_size_inches(fwidth, fheight)
plt.plot(year, ua850[:,1], c='k', label='850 百帕斯卡')
lm_ = lm(lmx=year, lmy=ua850[:,1])
plt.plot(year, lm_[0], c='b')
plt.legend()
plt.ylabel('位勢高度 [米]')
fig.savefig(f'temp\\ua850.jpg', dpi=300)

# plot all:
for i,x in enumerate(pls_):
    fig = plt.figure()
    fig.set_size_inches(fwidth * 0.8, 3)
    dt = to_anomaly(ua_yearly_08[i][:,1])
    plt.plot(year, dt, label=f'{x} 百帕斯卡', c='k')
    plt.axhline(y=0, c='k')
    plt.ylim(get_limit(dt, scale=2.2, equal=True))
    plt.ylabel('位勢高度距平 [米]')
    plt.legend()
    fig.tight_layout()
    fig.savefig(f'temp\\ua{x}_gh.jpg', dpi=300)

# all in one
fig,ax = plt.subplots(5,2)
fig.set_size_inches(fwidth * 0.8 * 2, 4 * 5)
for i in range(len(pls_)):
    if i <5:
        row = i
        col = 0
    else:
        row = i - 5
        col = 1
    dt = to_anomaly(ua_yearly_08[i][:,1])  #1 for gh, 2 for temp
    ax[row,col].plot(year, dt, c='k', label=f'{pls_[i]} 百帕斯卡')
    ax[row,col].axhline(y=0, c='grey')
    ax[row,col].set_ylabel(f'位高距平 [米]')
    ax[row,col].legend(loc='lower right')
    ax[row,col].set_ylim(get_limit(dt, scale=2.2, equal=True))
fig.tight_layout()
#fig.savefig('temp\\ua_allinone.jpg',dpi=300)

########################################################################################################################
# check HKO max T and KP max T
from moduels import *

names = ['Temperature_max','Temperature','Temperature_min']

for i, file in enumerate(names):
    data = pd.read_csv(f'Data\\{file}.csv')
    # select 1947-2020
    data = data[data['Year']>= 1947]

    fig = plt.figure()
    fig.set_size_inches(fwidth, fheight)
    plt.plot(data['Year'], data['HKO'], c='k', marker=marker, label='HKO')
    lm_ = lm(lmx=data['Year'], lmy=data['HKO'])
    slope, rate = lm_[1][1], lm_[1][2]
    plt.annotate(f'LM: y={slope:.2f}+{rate:.6f}x', xy=(0.7, 0.1), xycoords='axes fraction')
    plt.plot(data['Year'], lm_[0], c='k')

    plt.plot(data['Year'], data['KP'], c='b', marker=marker, label='KP')
    lm_ = lm(lmx=data['Year'][-54:], lmy=data['KP'][-54:])
    slope, rate = lm_[1][1], lm_[1][2]
    plt.annotate(f'LM: y={slope:.2f}+{rate:.6f}x', xy=(0.7, 0.05), xycoords='axes fraction', c='b')
    plt.plot(data['Year'][-54:], lm_[0], c='b')
    plt.ylabel(f'{t_name[i]} [{t_unit}]')
    plt.legend()
    fig.tight_layout()

# all in one
fig = plt.figure()
fig.set_size_inches(fwidth, fheight)
for i, file in enumerate(names):
    data = pd.read_csv(f'Data\\{file}.csv')
    # select 1947-2020
    data = data[data['Year']>= 1947]

    plt.plot(data['Year'], data['HKO'], c='k', marker=marker, label='HKO')
    lm_ = lm(lmx=data['Year'], lmy=data['HKO'])
    slope, rate = lm_[1][1], lm_[1][2]
    # plt.annotate(f'LM: y={slope:.2f}+{rate:.6f}x', xy=(0.7, 0.1), xycoords='axes fraction')
    plt.plot(data['Year'], lm_[0], c='k')

    plt.plot(data['Year'], data['KP'], c='b', marker=marker, label='KP')
    lm_ = lm(lmx=data['Year'][-54:], lmy=data['KP'][-54:])
    slope, rate = lm_[1][1], lm_[1][2]
    # plt.annotate(f'LM: y={slope:.2f}+{rate:.6f}x', xy=(0.7, 0.05), xycoords='axes fraction', c='b')
    plt.plot(data['Year'][-54:], lm_[0], c='b')
plt.ylabel(f'T [{t_unit}]')
# plt.legend()
fig.tight_layout()
