########################################################################################################################
# This file was used for updating KSC SR data.
########################################################################################################################
import numpy as np
import pandas as pd
from modules import *
#############################################
########################################################################################################################
# scraping climate data from HKO website:
from modules import *
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

# TODO: add comments and description
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

