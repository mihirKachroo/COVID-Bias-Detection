import csv
import sys
from urllib.parse import urlparse



def politicalBias(sentence_text):
    url = sentence_text
    o = urlparse(url)
    processedURL=o.netloc
    processedURL=processedURL[4:]
    x='This news source is not assigned in our database'
    with open("corpus.csv", "r") as f:
        reader = csv.reader(f)
        for line_num, content in enumerate(reader):
            if content[1] == processedURL:
                #print (content, line_num + 1)
                #print(content[3])
                #print(content[4])
                x=content[4]
    return x
