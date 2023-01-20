# %%
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# %%
df = pd.read_excel('./kakao-market-price.xlsx')
df = df.rename(columns={'Datetime(2022-10-13)': 'Date'})
df.head()

# %%
df['Date'].dtype, df['Open'].dtype, df['High'].dtype, df['Low'].dtype, df['Close'].dtype

# %%
# Reverse & Set 'Date' to index
df = df.iloc[::-1].reset_index().drop('index', axis=1)
df.head()

# %%
df['Date'].dt.hour.unique()

# %%
date = df['Date'].dt.date.astype(str)
bins = [0, 6.5, 12, 15, 24]
labels = ['_Closed', '_AM', '_PM', '_Closed']
period = pd.cut(df['Date'].dt.hour, bins=bins, labels=labels, include_lowest=True, ordered=False)

df['DayIndex'] = date.str.cat(period)
df.set_index('DayIndex')
df.head()

# %%
period.unique()

# %%
group = df.groupby('DayIndex', as_index=False)
new_df = group['Open'].first()
new_df['High'] = group['High'].max()['High']
new_df['Low'] = group['Low'].min()['Low']
new_df['Close'] = group['Close'].last()['Close']
new_df = new_df.set_index('DayIndex')
new_df.head()

# %%
new_df.set_index('DayIndex')

# %%
new_df[['Open', 'High', 'Low', 'Close']].head()

# %%
eval_vector = new_df[['Open', 'High', 'Low', 'Close']].to_numpy().reshape(-1, 1)
#scaler = StandardScaler().fit(eval_vector)
# 일반적으로 시계열 데이터는 MinMaxScaling 이 적절하므로, MinMaxScaling을 적용함.
scaler = MinMaxScaler().fit(eval_vector)

# %%
new_df['Open'] = scaler.transform(new_df['Open'].to_numpy().reshape(-1, 1))
new_df['High'] = scaler.transform(new_df['High'].to_numpy().reshape(-1, 1))
new_df['Low'] = scaler.transform(new_df['Low'].to_numpy().reshape(-1, 1))
new_df['Close'] = scaler.transform(new_df['Close'].to_numpy().reshape(-1, 1))
new_df.head()

# %%
new_df.to_excel('kakao-stock-preprocessed.xlsx')


