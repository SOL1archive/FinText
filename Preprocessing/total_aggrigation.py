# %%
import pickle

import pandas as pd

# %%
article_df = pd.read_excel('../data-dir/kakao-article-preprocessed.xlsx')
community_df = pd.read_excel('../data-dir/kakao-community-preprocessed.xlsx')
stock_df = pd.read_excel('../data-dir/kakao-stock-preprocessed.xlsx')
total_df = pd.DataFrame()

# %%
total_df = stock_df.set_index('DayIndex').drop('Unnamed: 0', axis=1)
total_df.head()

# %%
article_df.head()

# %%
community_df = community_df.drop(['view', 'good', 'bad'], axis=1)

# %%
community_df.head()

# %%
total_df['CommunityText'] = community_df.groupby('DayIndex')['CommunityText'].apply(list)
total_df['ArticleText'] = article_df.groupby('DayIndex')['ArticleText'].apply(list)

# %%
total_df = total_df.dropna(axis=0)
total_df.head()

# %%
total_df.to_pickle('../data-dir/data-df.pkl')
