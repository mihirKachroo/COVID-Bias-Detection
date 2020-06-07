import pandas as pd
from flask import Flask, jsonify, request
from NLPModel import bias
from NLPModel.NewsList import getNewsSource
from urllib.request import urlopen
from html.parser import HTMLParser

inputURL='https://www.cnn.com/2020/05/22/media/trump-fox-news-john-roberts/index.html'

result2=getNewsSource.politicalBias(inputURL)

class TitleParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.match = False
        self.title = ''

    def handle_starttag(self, tag, attributes):
        self.match = True if tag == 'title' else False

    def handle_data(self, data):
        if self.match:
            self.title = data
            self.match = False

html_string = str(urlopen(inputURL).read())

parser = TitleParser()
parser.feed(html_string)
statement = parser.title

result = bias.compute_bias(statement)

catogory = ""
if(result>1):
    catogory = "very"
elif(result<1 and result>0.8):
    catogory="somewhat"
else:
    catogory = "slightly"



print("This article is "+catogory+" "+result2+" biased.")