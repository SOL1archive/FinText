# %%
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# %%
df_article = pd.read_excel('../data-dir/kakao-text-data.xlsx', sheet_name='article')
df_community = pd.read_excel('../data-dir/kakao-text-data.xlsx', sheet_name='community')

# %%
df_article = df_article.iloc[::-1].reset_index().drop('index', axis=1)
df_community = df_community.iloc[::-1].reset_index().drop('index', axis=1)

# %% [markdown]
# ## Drop & Rename Columns

# %%
df_article['useful'].unique(), df_article['wow'].unique(), df_article['touched'].unique(), df_article['analytical'].unique(), df_article['recommend'].unique()

# %%
df_article = df_article.drop(
    [
        'article_No', 
        'article_id', 
        'press_id', 
        'author', 
        'useful', 
        'wow', 
        'touched', 
        'analytical', 
        'recommend'
    ], 
    axis=1
)
df_article = df_article.rename(columns={'published_datetime': 'Date'})

# %%
df_article.head()

# %%
df_community = df_community.drop(['article_no'], axis=1)
df_community = df_community.rename(columns={'body': 'article', 'published_datetime': 'Date'})

# %%
df_community.head()

# %%
df_article.isna().any().any(), df_community.isna().any().any()

# %% [markdown]
# ## Cleaning Data

# %% [markdown]
# ### Article

# %%
# 제목 전처리
# '[~~~]' 제거

df_article['title'] = df_article['title'].str.replace('\[.*?\]', '')
df_article['title'] = df_article['title'].str.strip()

# %%
# 본문 전처리
# URL, '[~~~]', ^n, (~~~ 기자) 등 제거

df_article['article'] = df_article['article'].str.replace('\[.*?\]', '')
df_article['article'] = df_article['article'].str.replace('\([0-9]*\)', '')
df_article['article'] = df_article['article'].str.replace('\(.*?기자\)', '')
df_article['article'] = df_article['article'].str.replace('(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)', '')
df_article['article'] = df_article['article'].str.replace('^n', '')
df_article['article'] = df_article['article'].str.replace('n(?![a-zA-Z ])', '')
df_article['article'] = df_article['article'].str.replace('▲', '')
df_article['article'] = df_article['article'].str.replace('... 기자', '')
df_article['article'] = df_article['article'].str.replace('\([A-Z]*?\)', '')
df_article['article'] = df_article['article'].str.replace('\([가-힣:]*?\)', '')
df_article['article'] = df_article['article'].str.replace('\(사진제공*?\)', '')
df_article['article'] = df_article['article'].str.replace('\(사진 제공*?\)', '')
df_article['article'] = df_article['article'].str.replace('\(대표*?\)', '')
df_article['article'] = df_article['article'].str.replace('\([0-9가-힣a-zA-Z ]{,10}\)', '')
df_article['article'] = df_article['article'].str.strip()

# %% [markdown]
# ### Community
# 커뮤니티 게시글에 대한 전처리는 답이 안보임.

# %%
df_community['title'] = df_community['title'].str.replace('(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)', '')
df_community['title'] = df_community['title'].str.replace('\[.*?\]', '')
df_community['title'] = df_community['title'].str.strip()

# %%
df_community['article'] = df_community['article'].str.replace('(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)', '')
df_community['article'] = df_community['article'].str.replace('^n', '')
df_community['article'] = df_community['article'].str.replace('n(?![a-zA-Z ])', '')
df_community['article'] = df_community['article'].str.strip()

# %% [markdown]
# ## Concating Strings

# %%
df_list = [df_article, df_community]

for i, df in enumerate(df_list):
    df['text'] = df['title'].str.cat(df['article'])
    df_list[i] = df.drop(['title', 'article'], axis=1)

# %%
df_list[0]

# %%
for i, df in enumerate(df_list):
    bins = [0, 7.5, 12, 15, 24]
    labels = ['_Closed', '_AM', '_PM', '_Closed']
    period = pd.cut(df['Date'].dt.hour, bins=bins, labels=labels, include_lowest=True, ordered=False)
    period[df['Date'].dt.day_of_week > 4] = '_Closed'
    
    date = df['Date'].dt.date.astype(str)
    df['DayIndex'] = date.str.cat(period)
    df = df.drop('Date', axis=1)
    df = df.set_index('DayIndex')
    df_list[i] = df

df_article, df_community = df_list

# %%
df_community.head()

# %% [markdown]
# ## Metric Index

# %%
df_community['MetricIndex'] = (df_community['good'] - df_community['bad']) / df_community['view']
#df_community['MetricIndex'] = MinMaxScaler((-0.5, 0.5)).fit_transform(df_community['MetricIndex'].values.reshape(1,-1)).reshape(-1, 1)
df_community = df_community.drop(['view', 'good', 'bad'], axis=1)

# %%
df_community = df_community.rename(columns={'text': 'CommunityText'})
df_article = df_article.rename(columns={'text': 'ArticleText'})

# %%
df_article.to_excel('../data-dir/kakao-article-preprocessed.xlsx')
df_community.to_excel('../data-dir/kakao-community-preprocessed.xlsx')
