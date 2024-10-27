import re
import csv
from time import sleep
from bs4 import BeautifulSoup
import pandas as pd
import requests

headers = {
	'accept': '*/*',
	'accept-encoding': 'gzip, deflate, br',
	'accept-language': 'en-US,en;q=0.9',
	'referer': 'https://www.google.com',
	'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36 Edg/85.0.564.44'
}

def get_article(card):
	headline = card.find('h4','s-title').text
	source = card.find('span','s-source').text
	posted = card.find('span','s-time').text.replace('.','').strip()
	description = card.find('p','s-desc').text.strip()
	raw_link = card.find('a').get('href')
	unquoted_link = requests.utils.unquote(raw_link)
	pattern = re.compile(r'RU=(.+)\/RK')
	clear_link = re.search(pattern,unquoted_link).group(1)
   
	article = (headline,source,posted,description,clear_link)
	return article

def get_the_news(search, max_num=10):
	#Run the main program
	template = 'https://news.search.yahoo.com/search?p={}'
	url = template.format(search)
  
	articles = []
	links = set()
	cnt = 0

	while True:
		response = requests.get(url,headers=headers)
		soup = BeautifulSoup(response.text,"html.parser")
		cards = soup.find_all('div','NewsArticle')

		#extract articles from page
		for card in cards:
			article = get_article(card)
			link = article[-1]
			if not link in links:
				links.add(link)
				articles.append(article)
			cnt += 1
			if cnt % 10 == 0:
				print(f"Collected the {cnt}th article")
			if cnt >= max_num:
				break
		if cnt >= max_num:
			break
        
		#Find the next page
		try:
			url = soup.find('a','next').get('href')
			sleep(2)
		except AttributeError:
			break
	
	# Save article data
	df = pd.DataFrame(articles)
	
	return df


df_trump = get_the_news('trump')