import pandas as pd

def load_data():
    financial_df = pd.read_csv("Lecture 6 - Record linking/Data/Large_linking_toydata/Financial_Data.csv")
    news_df = pd.read_csv("Lecture 6 - Record linking/Data/Large_linking_toydata/News_Data.csv")
    return financial_df, news_df

def filter_records(financial_df, news_df):
    nvda_records = financial_df[financial_df['Ticker'] == 'NVDA']
    nvidia_articles = news_df[news_df['Company'] == 'NVIDIA']
    return nvda_records, nvidia_articles

def map_ticker_to_company(financial_df):
    ticker_to_company = {
        'TSLA': 'Tesla',
        'IBM': 'IBM',
        'AAPL': 'Apple',
        'NVDA': 'NVIDIA',
        'SAP': 'SAP',
        'AMZN': 'Amazon',
        'GOOG': 'Google',
        'MSFT': 'Microsoft',
        'NFLX': 'Netflix',
        'META': 'Meta',
        'ORCL': 'Oracle'
    }
    financial_df['Company'] = financial_df['Ticker'].map(ticker_to_company)
    return financial_df

def link_records(financial_df, news_df):
    news_companies = set(news_df['Company'])
    financial_df['Has_News'] = financial_df['Company'].apply(lambda x: x in news_companies)
    linked_financial_count = financial_df['Has_News'].sum()
    unlinked_financial = financial_df[~financial_df['Has_News']]
    financial_companies = set(financial_df['Company'])
    news_df['Has_Financial'] = news_df['Company'].apply(lambda x: x in financial_companies)
    unlinked_news = news_df[~news_df['Has_Financial']]
    return linked_financial_count, unlinked_financial, unlinked_news

def print_summary(financial_df, news_df, linked_financial_count, unlinked_financial, unlinked_news):
    print("Number of NVDA records in financial data:", len(financial_df[financial_df['Ticker'] == 'NVDA']))
    print("Number of articles mentioning NVIDIA:", len(news_df[news_df['Company'] == 'NVIDIA']))
    print("Number of financial records successfully linked to news articles:", linked_financial_count)
    print("Number of financial records without matching news articles:", len(unlinked_financial))
    print("These records might not link due to data quality or the company never appearing in news.")
    print("Number of news articles without a matching financial record:", len(unlinked_news))
    print("These occur for companies not in the financial dataset (e.g., Intel, AMD, LinkedIn, Twitter).")
    print("\nSummary:")
    print("Financial records total:", len(financial_df))
    print("News articles total:", len(news_df))
    print("Linked financial records:", linked_financial_count)
    print("Unlinked financial records:", len(unlinked_financial))
    print("Unlinked news articles:", len(unlinked_news))

def record_linking():
    financial_df, news_df = load_data()
    nvda_records, nvidia_articles = filter_records(financial_df, news_df)
    financial_df = map_ticker_to_company(financial_df)
    linked_financial_count, unlinked_financial, unlinked_news = link_records(financial_df, news_df)
    print_summary(financial_df, news_df, linked_financial_count, unlinked_financial, unlinked_news)

if __name__ == "__main__":
    record_linking()
