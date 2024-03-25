# HKCNE-climate
Source codes for generating figures and computing statistics for the open-access book "Hong Kong Chronicles - Natural Environment" chapter 3 section 5 "Meteorology and Climate: Anthropogenic Climate Change".

# tmp
1) web scrapping -> ./data
2) plot and compute statistics from ./data

# Directory tree
```
├── HKCNE-climate/
|   ├─ README.md
|   ├─ geckodriver.exe (download at https://github.com/mozilla/geckodriver)
|   ├─ modules.py
|   ├─ web_scraping.py
|   ├─ Data/
|       ├─ temperature
|       ├─ ...
|   ├─ modules/
|       ├─ (under construction)
└──
```

# [web_scraping.py] Sample output format
| (Index) | Day | Jan | Feb | ... | Nov | Dec |
| ------- | --- | --- | --- | --- | --- | --- |
|    0    |  1  |     |     | ... | 25.2| 17.8|
|    1    | ... | ... | ... | ... | ... | ... |

# TODO
1) add comments 
2) add modules.py (check functions used in py)
3) modify function names -> readibility

# Note
It's under construction...
