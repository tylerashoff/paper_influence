import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

import multiprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re

from gensim.models import doc2vec
from collections import namedtuple

from sklearn.manifold import MDS

import requests
from lxml import html

from sklearn.manifold import TSNE

import os
#os.chdir('./project')


class txt_analysis():

    def __init__(self, data):
        self.df = data
        self.abstracts = np.array([data.abstract]).T
        pass

    def get_json_attrs(self, filename):
        '''
        support for gather_texts - pull json attributes
        '''
        filejson = json.load(open(filename))

        ref_count = len(filejson['bib_entries'])
        bodyjson = filejson['body_text']

        body = ''
        #conclusion = ''
        for section in bodyjson:
            body = body+' '+re.sub(',','',section['text'])
            #if section['section'][0:4].lower() == 'conc':
            #    conclusion = section['text']
            #    pass
            pass
        return(body, ref_count)

    def find_files(self, pmcid):

        main_path = 'kaggle/CORD-19-research-challenge/'
        json_paths = ['comm_use_subset/comm_use_subset/pmc_json/',
                      'custom_license/custom_license/pmc_json/',
                      'noncomm_use_subset/noncomm_use_subset/pmc_json/']
        paths = [main_path+json_path for json_path in json_paths]
        json_attrs = None
        # look through the different directories
        # and pull from json if id is found
        for path in paths:
            
            if str(pmcid)+'.xml.json' in os.listdir(path):
                filename = path + str(pmcid) + '.xml.json'
                json_attrs = np.append(pmcid,
                                       self.get_json_attrs(filename))
                pass
            pass
        return(json_attrs)
        
    def gather_texts(self, df, paths):
        '''
        find file id and pull json attributes
        '''
        pmcids = df.pmcid

        pool = multiprocessing.Pool(self.threads)
        attrs = pool.map(self.find_files, pmcids)
        
        # get back into correct shape
        attrs = [att for att in attrs if att is not None]
        attrs  = np.array(attrs).flatten().reshape(len(attrs), len(attrs[0]))

        return(attrs)
    
    def web_scrape(self, urls):
        '''
        scrape citation data from urls
        '''
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

    def process_docs(self, doc, minLength=2):
        '''
        Pre-process each document and return an nltk.Text object
        '''
        stops = stopwords.words('english')

        porter = nltk.PorterStemmer()
        corpum = nltk.Text([
            porter.stem(w.lower())
            for w in nltk.word_tokenize(doc)
            if (len(w) >= minLength and
                w.lower() not in stops and
                not np.any([l.isdigit() for l in w]))
        ])
        return (corpum)

    def build_corpus(self, docs):
        '''
        build corpus from documents
        '''
        pool = multiprocessing.Pool(self.threads)
        corpus = pool.map(self.process_docs, docs)
        
        return (corpus)

    def tsne(self, vectors, plot = False, n_components = 2, perplexity = 30):
        embed = TSNE(n_components, perplexity).fit_transform(vectors)

        if plot:
            plt.scatter(x = embed[:, 0],
                        y = embed[:, 1])
            plt.show()
            pass
        return(embed)

    def docvec(self, docs):
        '''
        perform doc to vec on a corpus
        '''

        corpus = self.build_corpus(docs)
        
        # Transform data
        docs = []
        analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
        for i, text in enumerate(corpus):
            words = text
            tags = [i]
            docs.append(analyzedDocument(words, tags))
            pass
        
        # Train mode
        model = doc2vec.Doc2Vec(docs, window = 300, min_count = 1, workers = 8)
        
        vecs = None
        for i in range(len(model.docvecs)):
            print(str(i)+'/'+str(len(model.docvecs)), end = '\r')
            vec = model.docvecs[i]
            if vecs is not None:
                vecs = np.vstack([vecs, vec])
                pass
            else:
                vecs = np.array([vec])
            pass
        return(vecs)

    def mdscaling(self, vectors):
        mds = MDS(n_components = 3).fit_transform(txt.docvecs)
        return(mds)
    
    def tell_me_when_done(self, phrase = 'all done you frickin genius'):
        os.system("say " + "'" + phrase + "'")
        pass
    
    def main(self, verbose = False, tell = False):
        self.verbose = verbose
        self.threads = 12

        main_path = 'kaggle/CORD-19-research-challenge/'
        json_paths = ['comm_use_subset/comm_use_subset/pmc_json/',
                      'custom_license/custom_license/pmc_json/',
                      'noncomm_use_subset/noncomm_use_subset/pmc_json/']
        paths = [main_path+json_path for json_path in json_paths]
        if self.verbose:
            print('gathering body text')
            pass
        self.document_attrs = self.gather_texts(self.df, paths)

        if self.verbose:
            print('running doc2vec on abstracts')
            pass
        #self.absvecs = self.docvec(self.df.abstract)
        
        if self.verbose:
            print('performing MDS')
            pass
        #self.mds = self.mdscaling(self.docvecs)
        
        # tell me when im done for long runtimes
        if tell:
            self.tell_me_when_done()
            pass
        
        #self.tsne(self.vectors, plot = True)
        pass
    pass


meta_og = pd.read_csv('kaggle/CORD-19-research-challenge/metadata.csv')

# find specific articles
finder = pd.Series([str(url) for url in meta_og.url])
meta = meta_og[finder.str.contains('ncbi')]#'doi|ncbi')]

cite = txt_analysis(meta.tail(100))
#cites = cite.web_scrape(meta.urls)

df = pd.read_csv('meta_data_citations.csv')
df_comp = df[df.abstract.notna()].iloc[18000:,:]

txt = txt_analysis(df_comp)
#txt.main(verbose = True,
#         tell = True)

v = np.array(json.load(open('./code/first_batch.json', 'r')))
vv = np.array(json.load(open('./code/second_batch.json', 'r')))
vvv = np.array(json.load(open('./code/third_batch.json', 'r')))
vvvv = np.array(json.load(open('./code/fourth_batch.json', 'r')))


#import pickle
#pickle.dump(txt, open("./code/txt.p", "wb"))
