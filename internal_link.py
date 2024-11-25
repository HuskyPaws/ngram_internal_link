import streamlit as st
import pandas as pd
import requests
import json
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import re
from openai import OpenAI
import sqlite3
from sqlalchemy import create_engine
import asyncio
import aiohttp
import logging
from tqdm import tqdm
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize session state variables
if 'all_available_keywords' not in st.session_state:
    st.session_state['all_available_keywords'] = []
if 'selected_keywords' not in st.session_state:
    st.session_state['selected_keywords'] = []
if 'content' not in st.session_state:
    st.session_state['content'] = None
if 'summary' not in st.session_state:
    st.session_state['summary'] = None
if 'keywords' not in st.session_state:
    st.session_state['keywords'] = []
if 'trigram_df' not in st.session_state:
    st.session_state['trigram_df'] = None
if 'custom_keywords' not in st.session_state:
    st.session_state['custom_keywords'] = []
if 'scraped_data_cache' not in st.session_state:
    st.session_state['scraped_data_cache'] = {}

# Sidebar steps explanation
st.sidebar.title("Steps to Use the Tool")
st.sidebar.markdown("1. Upload the bi-gram and tri-gram CSV file.")
st.sidebar.markdown("2. Enter a URL to scrape content from.")
st.sidebar.markdown("3. Review and adjust the extracted keywords.")
st.sidebar.markdown("4. Extract internal linking opportunities from the n-gram sheet.")
st.sidebar.markdown("5. Review the identified internal linking opportunities.")

# Page title
st.title("Internal Linking Opportunities Tool")

# OpenAI API key input
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# ScrapeOwl API key input
scrapeowl_api_key = st.sidebar.text_input("Enter your ScrapeOwl API Key:", type="password")

# Initialize OpenAI client
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)

# File uploader for n-gram CSV
uploaded_file = st.sidebar.file_uploader("Upload the bi-gram and tri-gram CSV file:", type=["csv"])
if uploaded_file is not None and st.session_state['trigram_df'] is None:
    st.session_state['trigram_df'] = pd.read_csv(uploaded_file)

# Input URL from user
url = st.sidebar.text_input("Enter the URL to scrape content from:")

# Choose method for keyword generation
keyword_method = st.sidebar.radio("Select Keyword Generation Method:", ("AI Generated", "N-gram Extraction"), index=0)

# Choose scraping method
scraping_method = st.sidebar.radio("Select Scraping Method:", ("Regular Scraping", "ScrapeOwl"), index=0)

# Function to scrape content from the URL asynchronously
async def scrape_content(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, "html.parser")
                    paragraphs = soup.find_all('p')
                    text_content = " ".join([p.get_text() for p in paragraphs])
                    return text_content
                else:
                    logging.error(f"Error: Received status code {response.status} for URL: {url}")
                    st.error(f"Error: Unable to retrieve content from the URL. Status code: {response.status}")
                    return ""
    except Exception as e:
        logging.error(f"Exception occurred while scraping URL {url}: {str(e)}")
        st.error(f"Exception occurred: {str(e)}")
        return ""

# Function to scrape content using ScrapeOwl API
def scrape_content_scrapeowl(url):
    if not scrapeowl_api_key:
        st.error("Please enter your ScrapeOwl API key.")
        return ""
    try:
        scrapeowl_url = "https://api.scrapeowl.com/v1/scrape"
        object_of_data = {
            "api_key": scrapeowl_api_key,
            "url": url,
            "premium_proxies": True,
            "country": "us",
            "elements": [
                {
                    "type": "xpath",
                    "selector": "//body"
                }
            ],
            "json_response": True
        }
        data = json.dumps(object_of_data)
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(scrapeowl_url, data, headers=headers)
        response_data = response.json()
        if response_data.get('status') == 200:
            text_content = response_data['data'][0]['results'][0]['text']
            # Clean up the text content by removing newlines and extra spaces
            text_content = re.sub(r'\n+', ' ', text_content).strip()
            text_content = re.sub(r'\s+', ' ', text_content)
            return text_content
        else:
            st.error(f"Error: Unable to retrieve content from ScrapeOwl. Status: {response_data.get('status')}")
            return ""
    except Exception as e:
        logging.error(f"Exception occurred while using ScrapeOwl for URL {url}: {str(e)}")
        st.error(f"Exception occurred: {str(e)}")
        return ""

# Function to summarize content using OpenAI API
def summarize_content(content):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key.")
        return ""
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following content focusing only on the core topic and omitting names, company names, or any unnecessary information: {content}"}
            ]
        )
        summary = completion.choices[0].message.content.strip()
        return summary
    except Exception as e:
        logging.error(f"Error with OpenAI API: {str(e)}")
        st.error(f"Error with OpenAI API: {str(e)}")
        return ""

# Function to generate keywords from scraped content using AI
def generate_keywords_ai(content):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key.")
        return []
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Analyze the following content to determine its core topic. Then, generate a comma-separated list of specific anchor text keywords that are contextually relevant and semantically close to the identified core topic. Only provide the keywords without any additional explanation or comments: {content}"}
            ]
        )
        keywords = completion.choices[0].message.content.strip().split(', ')
        # Filter keywords to ensure they are specific to the core topic
        specific_keywords = [kw for kw in keywords if len(kw.split()) > 1]
        return specific_keywords
    except Exception as e:
        logging.error(f"Error with OpenAI API: {str(e)}")
        st.error(f"Error with OpenAI API: {str(e)}")
        return []

# Function to generate keywords from scraped content using CountVectorizer
def generate_keywords(text):
    # Remove special characters and numbers
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Use CountVectorizer to extract bi-grams and tri-grams
    vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words='english')
    ngrams = vectorizer.fit_transform([clean_text])
    keywords = vectorizer.get_feature_names_out()
    return keywords

# Callback for custom keywords
def update_custom_keywords():
    if st.session_state.custom_keywords_input:
        # Split by comma and clean up
        new_keywords = [kw.strip().lower() for kw in st.session_state.custom_keywords_input.split(',') if kw.strip()]
        
        # First update all_available_keywords
        for kw in new_keywords:
            if kw not in st.session_state['all_available_keywords']:
                st.session_state['all_available_keywords'].append(kw)
        
        # Then update selected_keywords
        for kw in new_keywords:
            if kw not in st.session_state['selected_keywords']:
                st.session_state['selected_keywords'].append(kw)
        
        # Clear the input
        st.session_state.custom_keywords_input = ""

# Callback for selected keywords
def update_selected_keywords():
    st.session_state['selected_keywords'] = st.session_state.keywords_select

# Scrape content and generate keywords if URL is provided
if url and st.sidebar.button("Scrape and Generate Keywords"):
    if url in st.session_state['scraped_data_cache']:
        st.success("URL already scraped. Using cached data.")
        st.session_state['content'] = st.session_state['scraped_data_cache'][url]['content']
        st.session_state['summary'] = st.session_state['scraped_data_cache'][url]['summary']
        st.session_state['keywords'] = st.session_state['scraped_data_cache'][url]['keywords']
        
        # Update all_available_keywords with cached keywords
        for kw in st.session_state['keywords']:
            if kw not in st.session_state['all_available_keywords']:
                st.session_state['all_available_keywords'].append(kw)
    else:
        with st.spinner("Scraping content and generating keywords..."):
            if scraping_method == "Regular Scraping":
                content = asyncio.run(scrape_content(url))
            else:
                content = scrape_content_scrapeowl(url)
            
            if content:
                st.session_state['content'] = content
                summary = summarize_content(content)
                st.session_state['summary'] = summary
                
                # Generate keywords based on user selection
                if keyword_method == "AI Generated":
                    keywords = generate_keywords_ai(content)
                else:
                    keywords = list(generate_keywords(content))
                
                st.session_state['keywords'] = keywords
                
                # Update all_available_keywords
                for kw in keywords:
                    if kw not in st.session_state['all_available_keywords']:
                        st.session_state['all_available_keywords'].append(kw)
                
                # Cache the results
                st.session_state['scraped_data_cache'][url] = {
                    'content': content,
                    'summary': summary,
                    'keywords': keywords
                }

# Display content and keywords if they exist
if st.session_state['content'] is not None:
    st.subheader("Scraped Content Summary")
    st.text_area("Content Summary", st.session_state['summary'], height=150)

    st.subheader("Keywords and Filtering")
    
    # Custom keywords input with clear button
    col1, col2 = st.columns([3, 1])
    with col1:
        custom_keywords_input = st.text_input(
            "Add custom keywords (separated by commas):",
            key='custom_keywords_input',
            on_change=update_custom_keywords
        )
    with col2:
        if st.button("Add Keywords"):
            update_custom_keywords()

    # First ensure all selected keywords are in all_available_keywords
    for kw in st.session_state['selected_keywords']:
        if kw not in st.session_state['all_available_keywords']:
            st.session_state['all_available_keywords'].append(kw)
    
    # Now filter default values to only include those that exist in options
    valid_defaults = [kw for kw in st.session_state['selected_keywords'] 
                     if kw in st.session_state['all_available_keywords']]

    # Display all available keywords in multiselect
    selected_keywords = st.multiselect(
        "Select or adjust keywords to search for internal linking opportunities:",
        options=st.session_state['all_available_keywords'],
        default=valid_defaults,
        key='keywords_select',
        on_change=update_selected_keywords
    )

# Search for internal linking opportunities
if st.button("Find Internal Linking Opportunities") and st.session_state['trigram_df'] is not None:
    st.subheader("Internal Linking Opportunities")
    opportunities = []
    
    # Use selected_keywords for searching with exact matching
    for keyword in st.session_state['selected_keywords']:
        # Create pattern that matches the keyword as a whole word
        pattern = r'\b' + re.escape(keyword) + r'\b'
        keyword_regex = re.compile(pattern, re.IGNORECASE)
        matches = st.session_state['trigram_df'][
            (st.session_state['trigram_df']['N-gram'].str.contains(keyword_regex, na=False, regex=True)) & 
            (st.session_state['trigram_df']['URL'] != url)
        ]
        opportunities.append(matches)
    
    if opportunities:
        opportunities_df = pd.concat(opportunities).drop_duplicates()
        st.write(opportunities_df[['URL', 'N-gram']])
    else:
        st.write("No internal linking opportunities found.")

# Suggesting improvements section
st.sidebar.subheader("Suggested Improvements")
st.sidebar.markdown("- Implement caching for URL scraping to reduce redundant network requests.")
st.sidebar.markdown("- Add functionality to handle large n-gram sheets efficiently by processing in chunks or using a database.")
st.sidebar.markdown("- Enable more advanced keyword extraction methods, such as Named Entity Recognition (NER), to identify linking opportunities with higher relevance.")
st.sidebar.markdown("- Store n-gram data in a database (e.g., SQLite) for efficient querying and handling of large datasets.")
st.sidebar.markdown("- Use indexing on n-gram columns to speed up search operations.")
st.sidebar.markdown("- Implement asynchronous processing to handle web scraping and data processing tasks without blocking the user interface.")
st.sidebar.markdown("- Show progress bars or indicators during long-running operations to enhance user experience.")
