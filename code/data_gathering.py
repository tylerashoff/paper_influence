import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from copy import deepcopy
import csv

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

    def doc_transform(self, docs):
        '''
        support doc2vec models
        '''
        corpus = self.build_corpus(docs)
        # Transform data for use in doc2vec
        docs = []
        analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
        for i, text in enumerate(corpus):
            words = text
            tags = [i]
            docs.append(analyzedDocument(words, tags))
            pass
        return(docs)
    
    def docvec(self, docs, model_name = None, save = False):
        '''
        perform initial doc to vec on a corpus
        '''
        if self.verbose:
            print('building doc2vec model')
            pass
        
        docs = self.doc_transform(docs)
        
        # Train mode
        model = doc2vec.Doc2Vec(docs, window = 300, min_count = 1,
                                workers = self.threads)
        if save:
            model = deepcopy(model)
            model.save(model_name+'.bin')
            pass
        return(model.docvecs) #[[vec] for vec in model.docvecs])
    
    def docvec_update(self, docs, model_name, save = False):
        '''
        online doc to vec updates
        '''
        if self.verbose:
            print('updating doc2vec model')
            pass
        
        docs = self.doc_transform(docs)

        model = doc2vec.Doc2Vec.load(model_name+'.bin')
        model.build_vocab(docs, update = True)
        model.train(docs, total_examples=model.corpus_count,
                    epochs = model.epochs)
        if save:
            model = deepcopy(model)
            model.save(model_name+'.bin')
            pass
        pass

    def docvec_files(self, dirs, model_name):
        '''
        update doc to vec for all files specified
        '''
        if self.verbose:
            print('updating for json files')
            pass
        
        filenames = os.listdir(dirs)

        pmcids = None
        refs = None
        
        for filename in filenames:
            
            json_file = np.array(json.load(open(dirs+filename, 'r')))

            if pmcids is None:
                pmcids = list(json_file[:, 0])
                refs = list(json_file[:, 2])
                self.docvec(json_file[:, 1], save = True,
                            model_name = model_name)
                pass
            else:
                pmcids.extend(list(json_file[:, 0]))
                refs.extend(list(json_file[:, 2]))
                self.docvec_update(json_file[:, 0], save = True,
                                   model_name = model_name)
                pass
            pass
        return(pmcids, refs)

    def infer_helper(self, doc):
        '''
        support for infer_vectors, appends vectors to list
        '''
        vec = self.model.infer_vector(doc)
        return(vec)
    
    def infer_vectors(self, model_name, save_name, dirs=None, docs = None, ids=None):
        '''
        infer vectors from documents
        returns array with pmcid, ref count, vectorization
        '''
        if self.verbose:
            print('infering vectors')
            pass

        self.model = doc2vec.Doc2Vec.load(model_name+'.bin')

        if dirs is not None:
            filenames = os.listdir(dirs)
            count = 0
            for filename in filenames:
                count += 1
                if count != 3:
                    continue
                
                if self.verbose:
                    print(filename)
                    pass
                json_doc = np.array(json.load(open(dirs+filename, 'r')))
                
                pmcids = np.array([json_doc[:, 0]]).T
                docs = json_doc[:, 1]
                refs = np.array([json_doc[:, 2]]).T
                json_doc = None
                
                docs = self.doc_transform(docs)
                docs = [[token for token in doc[0]] for doc in docs]
                
                pool = multiprocessing.Pool(self.threads)
                vecs = pool.map(self.infer_helper, docs)
                docs = None
                
                key  =  np.hstack([pmcids, refs])
                pmcids = None
                refs = None
                
                print('saving', end = '\r')
                name = save_name+str(count)+'.csv'
                csv.writer(open(name, 'a', newline='')).writerows(np.array(vecs))
                vecs = None
                name = save_name+str(count)+'_key'+'.csv'
                csv.writer(open(name, 'a', newline='')).writerows(key)
                pass
            pass
        elif docs is not None:
            docs = self.doc_transform(docs)
            docs = [[token for token in doc[0]] for doc in docs]
            
            pool = multiprocessing.Pool(self.threads)
            vecs = pool.map(self.infer_helper, docs)

            if ids is not None:
                vecs = np.hstack([np.array([ids]).T, np.array(vecs)])
                pass
            
            print('saving', end = '\r')
            name = save_name+'.csv'
            csv.writer(open(name, 'a', newline='')).writerows(np.array(vecs))
            vecs = None
            pass
        pass
    
    def tell_me_when_done(self, phrase = 'all done you frickin genius'):
        os.system("say " + "'" + phrase + "'")
        pass
    
    def main(self, verbose = False, tell = False):
        self.verbose = verbose
        self.threads = 12

        # Gather attributes from json
        '''
        main_path = 'kaggle/CORD-19-research-challenge/'
        json_paths = ['comm_use_subset/comm_use_subset/pmc_json/',
                      'custom_license/custom_license/pmc_json/',
                      'noncomm_use_subset/noncomm_use_subset/pmc_json/']
        paths = [main_path+json_path for json_path in json_paths]
        self.document_attrs = self.gather_texts(self.df, paths)

        # save Doc2Vec for body text
        
        dirs = './code/jsons/'
        model_name = './code/doc2vec_model'

        self.pmcids, self.refs = self.docvec_files(dirs = dirs,
                                                   model_name = model_name)
        self.infer_vectors(model_name = model_name, dirs = dirs,
                           save_name = './code/docvecs/docvecs')
        '''

        # save doc vecs for abstracts
        model_name = './code/doc2vec_abs'
        self.docvec(self.df.abstract, model_name=model_name, save=True)

        self.infer_vectors(model_name = model_name, docs = self.df.abstract,
                           ids = self.df.pmcid,
                           save_name = './code/docvecs/abs_docvec')

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
df_comp = df[df.abstract.notna()]

txt = txt_analysis(df_comp)
txt.main(verbose = True,
         tell = True)

