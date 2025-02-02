import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm  
from statsmodels.formula.api import ols 
from sklearn.linear_model import LinearRegression
import seaborn as sns
from statsmodels.tsa.stattools import adfuller 

# Load datasets
file2 = pd.read_csv("icici_cp_last_one_year.csv")
file = pd.read_csv("yesback_data.csv")
file3 = pd.read_csv("HDFCBANKnew.csv")
file4 = pd.read_csv("axis.csv")
file5 = pd.read_csv("sbi.csv")
file6 = pd.read_csv("Nifty Bank Historical Data.csv")

# Initialize DataFrames
df = pd.DataFrame()
fdf = pd.DataFrame()

# Extract closing prices from various banks
df['date'] = file['Date']
df['yesbank_cp'] = file['Close']
df['icici_cp'] = file2['Close']
df['hdfc_cp'] = file3['Close']
df['axis_cp'] = file4['Close']
df['sbi_cp'] = file5['Close']

print(df.columns)

# Function to compute error ratio
def finding_errorratio(abc, xyz):
    # Convert values to float
    df[abc] = [float(str(i).replace(",", "")) for i in df[abc]]
    df[xyz] = [float(str(i).replace(",", "")) for i in df[xyz]]
    
    df[abc] = df[abc].astype(float)
    df[xyz] = df[xyz].astype(float)
    
    # Perform linear regression
    features = [abc]
    x = df[features]
    y = df[xyz]
    regression_model = LinearRegression()
    regression_model.fit(x, y)
    
    df['y_new'] = regression_model.coef_ * x + regression_model.intercept_
    df['residuals'] = df['y_new'] - df[xyz]
    
    # Compute error ratio
    model = ols(formula=f'{abc} ~ {xyz}', data=df).fit()
    bse_series = pd.Series(model.bse)
    num = bse_series.iloc[0]    
    den = df['residuals'].std()
    error_ratio = num / den
    
    return error_ratio

# Function to compute p-value using ADF test
def finding_p(abc, xyz):
    index = len(fdf)
    df[abc] = [float(str(i).replace(",", "")) for i in df[abc]]
    df[xyz] = [float(str(i).replace(",", "")) for i in df[xyz]]
    
    df[abc] = df[abc].astype(float)
    df[xyz] = df[xyz].astype(float)
    
    fdf.at[index, 'company-x'] = abc
    fdf.at[index, 'company-y'] = xyz
    
    features = [abc]
    x = df[features]
    y = df[xyz]
    regression_model = LinearRegression()
    regression_model.fit(x, y)
    fdf.at[index, 'slope'] = regression_model.intercept_
    
    df['y_new'] = regression_model.coef_ * x + regression_model.intercept_
    df['residuals'] = df['y_new'] - df[xyz]
    
    res = adfuller(df['residuals'])
    fdf.at[index, 'std_err_of_residuals'] = df['residuals'].std()
    fdf.at[index, 'pvalue'] = res[1]
    
    return res[1]

# Determine p-value by selecting the order with a lower error ratio
def finding_pvalue(abc, xyz):
    er1 = finding_errorratio(abc, xyz)
    er2 = finding_errorratio(xyz, abc)
    
    if er1 < er2:
        return finding_p(abc, xyz)
    else:
        return finding_p(xyz, abc)

# Compare stocks and store those with p-value < 0.05
lst = list(df.columns.values)
lstsize = len(lst)
i = 1
for x in lst:
    j = i + 1
    while j < lstsize:
        pvalue = finding_pvalue(lst[i], lst[j])
        if pvalue < 0.1:
            df2 = fdf[fdf.pvalue <= 0.05].copy()
        j += 1
    i += 1
print(fdf)
print(df2)
print("End of loop")   

# Load test data
tdata1 = pd.read_csv("HDFCBANK-16to16march.csv")
tdata2 = pd.read_csv("SBI-16thjuneto16thmarch.csv")
tdata3 = pd.read_csv("YESBANK.NS-16to16.csv")
tdata4 = pd.read_csv("AXISBANK.NS-16to16.csv")

tdf = pd.DataFrame()
tdf['date'] = tdata1['Date']
tdf['hdfc_cp'] = tdata1['Close']
tdf['sbi_cp'] = tdata2['Close']
tdf['yesbank_cp'] = tdata3['Close']
tdf['axisbank_cp'] = tdata4['Close']

# Function to compute residuals and plot deviations
def finding_residual(abc, xyz):
    tdf[abc] = [float(str(i).replace(",", "")) for i in tdf[abc]]
    tdf[xyz] = [float(str(i).replace(",", "")) for i in tdf[xyz]]
    
    tdf[abc] = tdf[abc].astype(float)
    tdf[xyz] = tdf[xyz].astype(float)
    
    features = [abc]
    x = tdf[features]
    y = tdf[xyz]
    regression_model = LinearRegression()
    regression_model.fit(x, y)
    
    tdf['y_new'] = regression_model.coef_ * x + regression_model.intercept_
    tdf['residuals'] = tdf['y_new'] - tdf[xyz]
    
    std_residuals = df['residuals'].std()
    tdf['z_value'] = tdf['residuals'] / std_residuals
    
    plt.figure(figsize=(10, 5))
    for i in range(len(tdf) - 1):
        x_values = [tdf['date'].iloc[i], tdf['date'].iloc[i + 1]]
        y_values = [tdf['z_value'].iloc[i], tdf['z_value'].iloc[i + 1]]
        color = "green" if max(y_values) > 2.5 or min(y_values) < -2.5 else "blue"
        plt.plot(x_values, y_values, color=color)
    
    plt.axhline(y=2.5, color='r', linestyle='dashed')
    plt.axhline(y=-2.5, color='r', linestyle='dashed')
    plt.show()

finding_residual('sbi_cp','hdfc_cp')   
print(tdf[tdf.z_value.abs() > 2.5])

tdf.plot(x='date', y=['sbi_cp', 'hdfc_cp'], kind='line', figsize=(10, 10))
plt.show()
