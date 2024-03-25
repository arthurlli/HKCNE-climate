########################################################################################################################
# This file was used for updating KSC SR data.
########################################################################################################################
# scraping climate data from HKO website:
#from modules import *
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

# TODO: add comments and description
# note: webdrive obtained from https://github.com/mozilla/geckodriver/releases
# current geckodriver version: 0.29.1 for win64
# note 2: open firefox first
def main():
    def simple_scrapper(start_yr, end_yr, base_url, file_name, sv_path):
        def check_path_exist(path):
            if not os.path.exists(path):
                os.makedirs(path)
                return
            return
        
        start, end = start_yr, end_yr
        # a for loop for scrape data from 1884-2020
        year = np.linspace(start, end, end - start + 1, dtype=int)
        for loop, yr in enumerate(tqdm(year)):
            # target link
            url = f'{base_url}{yr}'
            # use selenium to open Fireforx
            driver = webdriver.Firefox()
            driver.get(url)
            # take sleep time to load page
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
            df_tmp = np.zeros((len(rows) - 1, len(rows[0])), dtype=np.float)
            if 1940 <= yr <= 1946:
                # WWII period, no data available
                cols = np.array(rows[0], dtype=object).reshape(1, -1)[0]
                cols[cols == '\xa0'] = 'Day'
                df_tmp[:] = np.nan
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
                        # TODO at 1900, solve None tpye here
                        df_tmp[i-1, :] = dt.astype(np.float)
            df = pd.DataFrame(df_tmp, columns=cols)
            check_path_exist(path=sv_path)
            # save to ./Data/{defined filename}/{year}.csv
            df.to_csv(f'{sv_path}/{file_name}/{yr}.csv')
            # close drive for re-open
            driver.close()

        
    # Note: can modify to a function and scrape other data if wanted. Please be aware of data policy of HKO.

    # # mean temp
    simple_scrapper(1884,1900,'https://www.hko.gov.hk/en/cis/dailyElement.htm?ele=TEMP&y=',
                    file_name='temperature\\avg',sv_path=f"Data")

    simple_scrapper(1900,1901,'https://www.hko.gov.hk/en/cis/dailyElement.htm?ele=TEMP&y=',
                    file_name='temperature\\avg',sv_path=f"Data")
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


if __name__=="__main__":
    main()
# end
