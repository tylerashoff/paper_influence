import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import requests
from bs4 import BeautifulSoup as bs
from lxml import etree, html

from sklearn.manifold import TSNE

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

import os
#os.chdir('./project')


class txt_analysis():

    def __init__(self, data):
        self.df = data
        self.abstracts = np.array([data.abstract]).T
        self.nlp = spacy.load('en')
        pass

    def web_scrape(self, urls):
        citations = []
        ind = 0
        for url in urls:
            ind += 1
            print(ind, '/', len(urls))
            #url = urls.iloc[0]#'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC156578/'

            if 'ncbi' in url:
                url = url+'citedby/'
                page = requests.get(url)
                tree = html.fromstring(page.content)
                citations_str = tree.xpath('//*[@id="maincontent"]/div[2]/form/h2/text()')
                pass
            
            elif 'doi' in url:
                page = requests.get(url)
                url = re.sub('%25','%',
                             str(re.sub('%3F', '?',
                                        str(re.sub('%2F', '/',
                                                   str(page.content[302:388]))))))
                page = requests.get('http://'+url[2:-1])
                tree = html.fromstring(page.content)
                citations_str = tree.xpath('//*[@id="citing-articles-header"]/h2/text()')
                pass

            citation_count = [int(s) for s in citations_str[0].split()
                              if s.isdigit()] if len(citations_str) > 0 else []
            citation_count = citation_count[0] if len(citation_count) > 0 else 0
            
            citations.append(citation_count)
            pass

        return(citations)
        
    def doc_vecs(self, docs, nlp, fill = None):
        
        for doc in docs:
            
            if not isinstance(doc[0], str):

                if fill is not None:
                    # set to fill value if no value for doc
                    vecs.append(np.ones(len(nlp('test').vector))*fill)
                    pass
                continue

            doc = nlp(doc[0])
            
            # lemmatize and remove stop words
            lemmed = ' '.join([token.lemma_ for token in doc
                               if not token.is_stop])

            vecs.append(nlp(lemmed).vector)
            pass
        return(np.array(vecs))

    def tsne(self, vectors, plot = False, n_components = 2, perplexity = 30):
        embed = TSNE(n_components, perplexity).fit_transform(vectors)

        if plot:
            plt.scatter(x = embed[:, 0],
                        y = embed[:, 1])
            plt.show()
            pass
        return(embed)

    def tell_me_when_done(self, phrase = 'all done you frickin genius', tell = True):
        os.system("say " + "'" + phrase + "'")
        pass
    
    def main(self):

        self.df['citations'] = self.web_scrape(self.df.url)
        
        #self.vectors = self.doc_vecs(self.abstracts, self.nlp)
        self.tell_me_when_done(tell = True)
        #self.tsne(self.vectors, plot = True)
        pass
    pass

meta_og = pd.read_csv('kaggle/CORD-19-research-challenge/metadata.csv')

finder = pd.Series([str(url) for url in meta_og.url])
meta = meta_og[finder.str.contains('ncbi')]#'doi|ncbi')]

txt = txt_analysis(meta.tail(100))
txt.main()
txt.df.citations.describe()

