from textblob import TextBlob
import pandas as pd
import streamlit as st

import cleantext


st.header("Sentiment Analysis")
with st.expander("Analyse Text"):
    text = st.text_input('Text here:')
    if (text):
        blob = TextBlob(text)
        st.write("Polarity:",round(blob.sentiment.polarity,2))
        st.write("Subjectivity:",round(blob.sentiment.subjectivity,2))
    
    pre = st.text_input('Clean text:')
    if pre:
        st.write(cleantext.clean(pre, clean_all = False, extra_spaces = True,stopwords = True, lowercase = True, numbers = True, punct = True))
        
with st.expander('Analyser Csv'):
    upl = st.file_uploader('upload file')
    
    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity
    
    def analyze(x):
        if x>=0.5:
            return 'Positive'
        elif x<=-0.5:
            return 'Negative'
        else:
            return 'Netural'
    if upl:
        df = pd.read_csv(upl,sep='\t')
        df['Score'] = df['Review'].apply(score)
        df['analysis'] = df['Score'].apply(analyze)
        st.write(df.head())
        
        @st.cache_data
        def convert_df(df):
            return df.to_csv()
        
        csv = convert_df(df)
        
        st.download_button(
            label='Download data as csv',
            data = csv,
            file_name = 'sentiment.csv',
            mime = 'txt/csv'
        )
