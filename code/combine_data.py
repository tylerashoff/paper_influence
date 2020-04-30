import pandas as pd
import numpy as np
from functools import reduce
import csv

df = pd.read_csv('./code/meta_data_citations.csv')
abstracts = pd.read_csv('./code/docvecs/abs_docvec.csv').to_numpy()
bodies = pd.read_csv('./code/docvecs/body_docvec.csv').to_numpy()

keys, body_inds = np.unique(bodies[:, 0],
                            return_index=True)

bodies = bodies[np.sort(body_inds), :]

inter, abs_mask, body_mask = np.intersect1d(list(abstracts[:, 0]),
                                            list(bodies[:, 0]),
                                            return_indices=True)

inter = reduce(np.intersect1d, (list(abstracts[:, 0]),
                                list(bodies[:, 0]),
                                list(df.pmcid)))

abs_mask = [ind in inter for ind in abstracts[:, 0]]
body_mask = [ind in inter for ind in bodies[:, 0]]
df_mask = [ind in inter for ind in df.pmcid]

abs_ids = np.array([abstracts[abs_mask, 0]]).T
abs_vecs = abstracts[abs_mask, 1:]

body_ids = np.array([bodies[body_mask, 0]]).T
body_refs = np.array([bodies[body_mask, 1]]).T
body_vecs = bodies[body_mask, 2:]

cites = np.array([df.citations[df_mask]]).T
dates = np.array([df.publish_time[df_mask]]).T
journal = np.array([df.journal[df_mask]]).T

total_df = np.hstack([cites, abs_ids, dates, journal, body_refs, abs_vecs, body_vecs])

csv.writer(open('./code/full_data.csv', 'a', newline='')).writerows(total_df)
