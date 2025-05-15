import pandas as pd

sampleDf =pd.get_dummies(pd.DataFrame({'DOCUMENT_TYPE': ['Incoming invoice',
  'Incoming invoice',
  'Incoming invoice',
  'Incoming invoice',
  'Incoming invoice',
  'Incoming invoice'],
 'PRODUCT_CLASS': ['SEA-CP_01',
  'SEA-CP_02',
  'SEA-CP_03',
  'SEA-CP_04',
  'SEA-CP_05',
  'SEA-CP_01'],
 'TAX_RATE': ['19', '19', '19', '7', '19', '7'],
'INDUSTRIAL_SECTOR': ['A', 'A', 'A', 'A', 'A', 'A'],
 'ACCOUNT': ['Account_3400',
  'Account_3400',
  'Account_3400',
  'Account_3300',
  'Account_4605',
  'Account_3300']}))

sampleDf.to_csv("assets/toy_accounting.csv")

#vat_list = [col for col in sampleDf.columns if 'TAX_RATE' in col]
#account_list = [col for col in sampleDf.columns if 'ACCOUNT' in col]
#invoice_list = [col for col in sampleDf.columns if 'DOCUMENT_TYPE' in col]
#industry_list = [col for col in sampleDf.columns if 'INDUSTRIAL_SECTOR' in col]
#product_list = [col for col in sampleDf.columns if 'PRODUCT_CLASS' in col]