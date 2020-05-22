"""
    Fill the blanks of MRHIER.RRF (verification needed)
    
    Author: Louis Qin <louisqin@ucla.edu> or <qyl0509@icloud.com>
"""

import pandas as pd

# import MRHIER.RRF
mrhier = pd.read_csv('../../2018AB_RRF/META/MRHIER.RRF', sep='|', header=None, dtype=object)
mrhier = mrhier.drop(9, axis=1)
mrhier.columns = ['CUI', 'AUI', 'CXN', 'PAUI', 'SAB', 'RELA', 'PTR', 'HCD', 'CVF']

# Filling AUIs SABs RELAs PTRs should gives us a more 'complete' MRHIER
for index, row in mrhier.iterrows():
	if pd.isna(row.AUI):
		row['AUI'] = mrhier.iloc[index-1]['AUI']
		row['SAB'] = mrhier.iloc[index-1]['SAB']
		row['RELA'] = mrhier.iloc[index-1]['RELA']
		auis = mrhier.iloc[index-1]['PTR'].split(sep='.')
		row['PTR'] = auis[0] + '.' + auis[1] + row['PTR'][1:]

mrhier.to_csv('MRHIER_filled.csv', sep='|', index=False)
