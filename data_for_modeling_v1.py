# our first code for putting together data for running regressions against ratings

from standard_imports import *


info = pd.read_parquet(data_path / 'company info.parquet')
ratings = pd.read_parquet(data_path / 'S&P ratings.parquet') \
    .rename({'date':'rating_date', 'end_date':'rating_end_date'}, axis = 1)

# The first thing we do is to remove all companies without an industry indicator
# and any companies in the financial, real estate, and utility industries
ratings = pd.merge(ratings, info[['gvkey', 'gsector']], on = 'gvkey')
ratings = ratings[~ratings.gsector.isna()]
ratings = ratings[~ratings.gsector.isin(['40', '55', '60'])]

# identifying the first default for all defaulting comkpanies
defaults = ratings[ratings.rtg_symbol.isin(['D', 'SD', 'R'])].copy()
defaults = defaults[['gvkey', 'rating_date']].groupby('gvkey', as_index = False).rating_date.min()
defaults.columns = ['gvkey', 'default_date']

# we first pull in the financial data and filter it
financials = pd.read_parquet(data_path / 'compustat financials.parquet') \
    .rename({'at':'total_assets', 'lt':'total_liabilities', 'ceq':'common_equity', 'act':'current_assets',
             'lct':'current_liabilities', 'rect':'acct_recv', 'ap':'acct_payable', 'dlc':'short_debt',
             'dltt':'long_debt', 'dltis':'issued_debt',  'dvt':'debt_value', 'che':'cash_short_inv',
             'xint':'interest_exp', 'xrd':'rsch_devel', 'xsga':'admin_exp', 'oibdp':'op_inc_b4_dep',
             'sale':'total_sales', 'ni':'net_income', 'oancf':'op_cash_flow', 'fincf':'fin_cash_flow',
             'csho':'common_shares', 'pstk':'preferred_stock', 'dvp':'preferred_dividends',
             'dp':'depreciation', 'txt':'taxes', 'dv':'dividends', 'urect':'net_acct_recv',
             'revt':'total_revenues',
            }, axis = 1)
financials = financials[~pd.isnull(financials.total_assets)]
financials = financials[financials.total_assets > 0]

# making sure we restrict ourselves to the rated universe
# we don't know the default history of the non-rated universe
financials = pd.merge(financials, ratings, on = 'gvkey')
financials = financials[(financials.data_date >= financials.rating_date) & \
                        (financials.data_date < financials.rating_end_date)]
financials = financials[~financials.rtg_symbol.isin(['NR','R','SD','D'])] \
    .drop_duplicates()

#################################################################################################
# the following sets a default flag to 1 anytime there 
# is a default occurring 6 to 18 months after the statement date
financials = pd.merge(financials, defaults[['gvkey', 'default_date']], on = 'gvkey', how = 'left')
financials.loc[pd.isnull(financials.default_date), 'default_date'] = pd.Timestamp(2199,12,31)

# I'm removing any records that are after about six months prior to a default
# Most analysts already know a company is in distress during this window.
# Also, most financials aren't released for six to ten weeks after the statement date
financials['default_minus_six_months'] = financials.default_date - DateOffset(months = 6)
financials = financials[financials.data_date < financials.default_minus_six_months]

financials['date_plus_six_months'] = financials.data_date + DateOffset(months = 6)
financials['date_plus_18_months'] = financials.data_date + DateOffset(months = 18)

financials['default_flag'] = 0
financials.loc[(financials.default_date > financials.date_plus_six_months) \
             & (financials.default_date <= financials.date_plus_18_months), 'default_flag'] = 1
financials = financials.drop_duplicates()

financials = financials.drop(['default_minus_six_months', 'date_plus_six_months', 'date_plus_18_months'], axis = 1)
#################################################################################################

# Making sure we have enough runway to encapsulate the entire default window
max_default_date = financials[financials.default_flag == 1].default_date.max()
financials = financials[financials.data_date <= max_default_date - DateOffset(months = 18)]

#################################################################################################

# --- Profitability Ratios ---
financials['roa'] = financials['net_income'] / financials['total_assets']
financials['roe'] = financials['net_income'] / financials['common_equity']
financials['operating_margin'] = financials['ebit'] / financials['total_sales']
financials['net_profit_margin'] = financials['net_income'] / financials['total_sales']
financials['ceq_to_assets'] = financials['common_equity'] / financials['total_assets']

# --- Leverage Ratios --- 
financials['total_debt'] = financials['short_debt'] + financials['long_debt']

financials['debt_to_equity'] = financials['total_debt'] / financials['common_equity']
financials['debt_to_capital'] = financials['total_debt'] / (financials['total_debt'] \
                                + financials['common_equity'] + financials.get('preferred_stock', 0))
financials['liabilities_to_assets'] = financials['total_liabilities'] / financials['total_assets']
financials['debt_to_assets'] = financials['total_debt'] / financials['total_assets']

financials['tangible_assets'] = financials['total_assets'] - financials['intan'] - financials['gdwl']
financials['liab_to_tang_assets'] = financials['total_liabilities'] / financials['tangible_assets']
financials['debt_to_tang_assets'] = financials['total_debt'] / financials['tangible_assets']

# Coverage Ratios
financials['interest_coverage'] = financials['ebit'] / financials['interest_exp']
financials['dscr'] = financials['ebit'] / (financials['interest_exp'] + financials['short_debt'])
financials['fcrr'] = (financials['ebit'] + financials['interest_exp'] + financials['preferred_dividends']) \
        / (financials['interest_exp'] + financials['preferred_dividends'])

# Liquidity Ratios
financials['current_ratio'] = financials['current_assets'] / financials['current_liabilities']
financials['quick_ratio'] = (financials['current_assets'] - financials['invt']) / financials['current_liabilities']  
financials['quick_ratio_alt'] = (financials['cash_short_inv'] + financials['acct_recv']) / financials['current_liabilities']
financials['cash_ratio'] = financials['cash_short_inv'] / financials['current_liabilities']

# --- Intangible Ratios ---
financials['intan_to_assets'] = financials['intan'] / financials['total_assets']
financials['goodwill_to_assets'] = financials['gdwl'] / financials['total_assets']

# --- Accruals Ratio ---
financials['avg_at'] = financials.groupby('gvkey')['total_assets'].transform(lambda x: (x.shift(1) + x) / 2)
financials['accruals'] = (financials['net_income'] - financials['op_cash_flow']) / financials['avg_at']
financials['accruals_alt'] = financials['op_cash_flow'] / financials['net_income']

# --- Efficiency Ratios ---
financials['inventory_turnover'] = financials['cogs'] / financials['invt']
financials['receivables_turnover'] = financials['total_sales'] / financials['acct_recv']
financials['asset_turnover'] = financials['total_sales'] / financials['total_assets']

# --- Cash Flow Ratios ---
financials['ocf_to_debt'] = financials['op_cash_flow'] / financials['total_debt']
financials['free_cash_flow'] = financials['op_cash_flow'] - financials['capx']
financials['fcf_to_debt'] = financials['free_cash_flow'] / financials['total_debt']

# --- Dividend Ratios ---
financials['dividend_yield'] = financials['dividends'] / financials['prcc_f']
financials['dividend_payout'] = financials['dividends'] / financials['net_income']

# --- Operational Efficiency ---
financials['sg&a_intensity'] = financials['admin_exp'] / financials['total_sales']
financials['rd_intensity'] = financials['rsch_devel'] / financials['total_sales']
financials['capex_ratio'] = financials['capx'] / financials['total_sales']

# --- Growth Ratios ---
financials['sales_growth'] = financials.groupby('gvkey')['total_sales'].pct_change(fill_method=None)
financials['earnings_growth'] = financials.groupby('gvkey')['net_income'].pct_change(fill_method=None)
financials['ocf_growth'] = financials.groupby('gvkey')['op_cash_flow'].pct_change(fill_method=None)
financials['asset_growth'] = financials.groupby('gvkey')['total_assets'].pct_change(fill_method=None)
financials['dividend_growth'] = financials.groupby('gvkey')['dividends'].pct_change(fill_method=None)

# --- Market Valuation Ratios ---
financials['book_value_per_share'] = financials['common_equity'] / financials['common_shares']
financials['price_to_book'] = financials['prcc_f'] / financials['book_value_per_share']

financials['eps'] = financials['net_income'] / financials['common_shares']
financials['price_to_earnings'] = financials['prcc_f'] / financials['eps']

list_of_factors = ['roa', 'roe', 'operating_margin', 'net_profit_margin', 'ceq_to_assets',
                   'debt_to_equity', 'debt_to_capital', 'liabilities_to_assets', 'debt_to_assets',
                   'liab_to_tang_assets', 'debt_to_tang_assets', 'interest_coverage', 'dscr', 'fcrr', 
                   'current_ratio', 'quick_ratio', 'quick_ratio_alt', 'cash_ratio',
                   'intan_to_assets', 'goodwill_to_assets', 'accruals', 'accruals_alt',
                   'inventory_turnover', 'receivables_turnover', 'asset_turnover',
                   'ocf_to_debt', 'free_cash_flow', 'fcf_to_debt','dividend_yield', 'dividend_payout',
                   'sg&a_intensity', 'rd_intensity', 'capex_ratio',
                   'sales_growth', 'earnings_growth', 'ocf_growth', 'asset_growth', 'dividend_growth',
                   'book_value_per_share', 'price_to_book','eps', 'price_to_earnings', 'total_assets']


# --- Normalized Variables ---
for _ in ['acct_payable', 'op_inc_b4_dep', 'op_cash_flow', 
          'fin_cash_flow', 'preferred_dividends', 'taxes', 'dividends', 
          'net_acct_recv']:
    financials[_ + '_tos'] = financials[_] / financials['total_sales']
    list_of_factors.append(_ + '_tos')

for _ in ['current_assets', 'current_liabilities', 
          'short_debt', 'long_debt', 'issued_debt', 'debt_value', 'cash_short_inv', 
          'rsch_devel', 'admin_exp', 'preferred_stock', 'depreciation']:
    financials[_ + '_toa'] = financials[_] / financials['total_assets']
    list_of_factors.append(_ + '_toa')

# --- Other Interesting Variables
financials['interest_rate'] = financials['interest_exp'] / (financials['short_debt'] + financials['long_debt'])
list_of_factors.append('interest_rate')

list_of_factors = list(set(list_of_factors))

# pre-clean all factors once
for factor in list_of_factors:
    financials[factor] = financials[factor].astype(float)
financials = financials.replace([np.inf, -np.inf], np.nan)

financials = financials \
    .sort_values(['gvkey', 'data_date']) \
    .drop_duplicates() \
    .reset_index(drop = True)

print(f'\nThere are {len(list_of_factors):,} factors in the variable list_of_factors.\n')

print(f'There are {len(financials):,} rows in financials.')
print(f'There are {len(financials.gvkey.unique()):,} unique companies.')
print(f'there are {financials.default_flag.sum()} defaults in the data.\n')

sector_map = {'10':'Energy', '15':'Materials', '20':'Industrials', '25':'Consumer Discretionary',
              '30':'Consumer Staples', '35':'Health Care', '45':'Information Technology',
              '50':'Communication Services'}

for sector, sector_name in sector_map.items():
    a = financials[financials.gsector == sector]
    print(f'{len(a):>8,}    {sector_name}')
print()