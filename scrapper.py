import requests
from bs4 import BeautifulSoup
import os

# Create a directory for saving documents
if not os.path.exists('document'):
    os.mkdir('document')

# Function to download and save a PDF file
def download_pdf(url, filename):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join('document', filename), 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

# Function to scrape and download PDFs from a web page
def scrape_pdfs(url):
    try:
        page = requests.get(url)
        if page.status_code == 200:
            soup = BeautifulSoup(page.content, 'html.parser')
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and href.endswith('.pdf'):
                    # Construct the full URL if necessary
                    if not href.startswith('http'):
                        href = url + href
                    download_pdf(href, href.split('/')[-1])
        else:
            print(f"Failed to access {url}. Status code: {page.status_code}")
    except Exception as e:
        print(f"Error accessing {url}: {e}")

# Replace with the URL of the website you want to scrape
target_website = 'https://www.engpaper.com/data-science-ieee-paper.html'

# Start scraping
scrape_pdfs(target_website)
