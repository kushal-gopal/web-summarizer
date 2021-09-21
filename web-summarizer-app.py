######################
# Import libraries
######################

import streamlit as st
import nltk
from newspaper import Article
from transformers import pipeline
from summarizer import TransformerSummarizer

######################
# Page Title
######################

st.write("""
# News Website Summarizer
This app summarizes your favourite news site!
***
""")

######################
# Input Text Box
######################

st.header('Enter the url')

url = "https://edition.cnn.com/2021/07/22/sport/mexico-france-2020-tokyo-olympics-spt-intl/index.html"

url = st.text_area("URL", url, height=150)

st.write("""
***
""")

nltk.download('punkt')

article = Article(url, language="en")

# To download the article
article.download()

# To parse the article
article.parse()

# To perform natural language processing ie..nlp
article.nlp()

# Article Title
st.header("Article's Title:")
article.title

# To extract text
st.write("""
## Article's Text:""" + article.text)

# Article Summary
st.header('Article Summary')
article.summary


# using pipeline API for summarization task
summarization = pipeline("summarization")

summary_api = summarization(article.text)[0]['summary_text']

st.header('Article Summary with pipelineAPI')
summary_api

st.write("""
***
""")


#GPT2_model = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")
#summary_gpt2 = ''.join(GPT2_model(article.text, min_length=60))

#st.header('Article Summary with GPT-2 Model')
#summary_gpt2
