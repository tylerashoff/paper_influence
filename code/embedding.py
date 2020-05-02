import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.layers import Input, Dense, Lambda
from keras import backend as K
from sklearn import manifold
from sklearn import metrics
import datetime
import csv


class embeddings():

    def __init__(self, verbose=False):
        self.verbose = verbose
        pass

    def sampling(self, args):
        '''
        supports the var_autoencoder method
        '''
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def var_autoencoder(self, x_train, x_test, dim=2):
        '''
        return latent mean from a variational autoencoder
        '''
        if self.verbose:
            print('variational autoencoder')
        original_dim = x_train.shape[1]
        # network parameters
        input_shape = (original_dim,)
        intermediate_dim = 64
        batch_size = 32
        latent_dim = dim
        epochs = 10

        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        z = Lambda(
            self.sampling, output_shape=(latent_dim,),
            name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')

        reconstruction_loss = binary_crossentropy(inputs, outputs)

        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)

        vae.compile(optimizer='adam')

        vae.fit(
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None))

        latent_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
        return (latent_mean)

    def stress_r2(self, X1, X2):
        '''
        compute Kruskal's stress and R^2 values for embedding
        '''
        if self.verbose:
            print('stressing out')
            pass
        ds = np.triu(cdist(X1, X1)).flatten()
        fs = np.triu(cdist(X2, X2)).flatten()

        ds = ds[ds != 0]
        fs = fs[fs != 0]

        stress = np.sqrt(np.sum((fs - ds)**2) / np.sum(ds**2))
        r2 = np.corrcoef(fs, ds)[0, 1]
        return (stress, r2)

    def plotting(self,
                 X_fit,
                 labs=None,
                 embed='',
                 stress=0,
                 r2=0,
                 show=True,
                 filename=None):
        if self.verbose:
            print('plotting')
            pass
        fig = plt.figure()
        plt.style.use('tableau-colorblind10')
        ax = fig.add_subplot(111, projection='3d')

        if labs is None:
            ax.scatter(X_fit[:, 0], X_fit[:, 1], X_fit[:, 2])
            pass
        elif labs is not None:
            ax.scatter(X_fit[:, 0], X_fit[:, 1], X_fit[:, 2], s=labs)
            pass

        ax.legend(loc="upper right")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.title(
            embed + ' Embedding\n' + 'Kruskal stress: ' + str(round(stress, 4))
            + ', R-squared: ' + str(round(r2, 4)),
            loc='left')
        if filename is not None:
            plt.savefig('./final/' + filename + '.png')
            pass
        if show:
            plt.show()
            pass
        pass

    def spec(self, data, dim=3):
        '''
        embed data using spectral embedding
        '''
        if self.verbose:
            print('spectral embedding')
            pass
        spec = manifold.SpectralEmbedding(n_components=dim).fit_transform(data)
        return (spec)

    def smacof(self, data, dim=3):
        '''
        embed data using smacof method of mds
        '''
        if self.verbose:
            print('smacof')
            pass
        pairs = metrics.pairwise_distances(data)
        smac, _, _ = manifold.smacof(
            dissimilarities=pairs, n_components=dim).fit_transform(data)
        return (smac)

    def isomap(self, data, dim=3):
        '''
        embed data using Isomap
        '''
        if self.verbose:
            print('Isomap')
            pass
        isomap = manifold.Isomap(n_components=dim).fit_transform(data)
        return (isomap)

    def k_mean_dists(self, vecs, k=None, calc_pairs=True):
        '''
        calculate the mean distance for the k closest points
        if k is None then calculate total mean
        '''
        if self.verbose and calc_pairs:
            print('mean distances')
            pass

        if calc_pairs:
            pairs = metrics.pairwise_distances(vecs)
            pass
        else:
            pairs = vecs
            pass

        if k is not None:
            sorted_vecs = np.sort(pairs, axis=1)
            avg_dist = np.mean(sorted_vecs[:, 1:k + 1], axis=1)
            pass
        else:
            avg_dist = np.mean(pairs, axis=1)
            pass

        return (avg_dist)

    def n_month_k_mean_dists(self, vecs, dates, n=6, k=None):
        '''
        calculate mean distance for prior n months for k closes
        if k is None the calculate total for prior n months
        '''
        if self.verbose:
            print('six month means')
            pass

        pairs = metrics.pairwise_distances(vecs)

        avg_dists = []
        count = 0

        for date in dates:
            count += 1
            if self.verbose:
                print(count, '/', len(dates), end='\r')
                pass

            low = min(pd.date_range(end=date, periods=n, freq='M'))

            mask = (dates > low) & (dates <= date)
            avg_dist = self.k_mean_dists(
                np.array([pairs[count - 1, mask]]), k=k, calc_pairs=False)

            avg_dists.extend(avg_dist)
            pass

        return (np.array(avg_dists))

    def radius(self, vecs, sd_mult=0.1):
        '''
        count number of points in open ball of radius sd/sd_mult
        '''
        if self.verbose:
            print('radii')
            pass

        pairs = metrics.pairwise_distances(vecs)
        uptri = np.triu(pairs)

        sd = np.std(uptri[uptri != 0])
        rad = sd * sd_mult

        pairs = pairs + (np.eye(len(pairs)) * rad)
        return (np.sum(pairs < rad, axis=1))

    def save(self, data, filename):
        csv.writer(open(filename, 'w')).writerows(data)
        pass

    def main(self, data, dates, embedding=None, dim=3, k=None):

        if embedding is None:
            emb_data = data
            pass
        elif embedding == 'spectral':
            emb_data = self.spec(data, dim)
            pass
        elif embedding == 'isomap':
            emb_data = self.isomap(data, dim)
            pass
        elif embedding == 'varenc':
            emb_data = self.var_autoencoder(data, data, dim)
            pass
        else:
            print(str(embedding) + '- is an invalid embedding')
            return (0)

        ball = self.radius(emb_data)
        means = self.k_mean_dists(emb_data, k=k)
        month_means = self.n_month_k_mean_dists(emb_data, dates, n=6, k=k)

        return (emb_data, means, month_means, ball)

    pass


meta = pd.read_csv('./datasets/meta_data_citations.csv')
df = pd.read_csv('./datasets/preembedding_data.csv').to_numpy()

cites = df[:, 0]
pmcid = df[:, 1]
dates = pd.to_datetime(df[:, 2])
journal = df[:, 3]
refs = df[:, 4]
abs_vecs = df[:, 5:105]
body_vecs = df[:, 105:]

emb = embeddings(verbose=True)

embeddings = ['spectral', 'isomap', 'varenc', None]
for embedding in embeddings:
    if emb.verbose:
        print('abstract vecs')
        pass
    emb_data, means, month_means, ball = emb.main(
        abs_vecs, dates, embedding=embedding, dim=5, k=5)

    data = np.hstack([
        np.array([pmcid]).T,
        np.array([dates.astype(str)]).T,
        np.array([journal]).T,
        np.array([cites]).T,
        np.array([refs]).T,
        np.array([means]).T,
        np.array([month_means]).T,
        np.array([ball]).T, emb_data
    ])

    if emb.verbose:
        print('body vecs')
        pass
    emb_data, means, month_means, ball = emb.main(
        body_vecs, dates, embedding=embedding, dim=5, k=5)

    data = np.hstack([
        data,
        np.array([means]).T,
        np.array([month_means]).T,
        np.array([ball]).T, emb_data
    ])

    date_sort = np.argsort(dates)
    data = data[date_sort, :]

    filename = 'full_data'
    if embedding is not None:
        filename = str(embedding) + '_' + filename
        pass

    emb.save(filename='./datasets/' + filename + '.csv', data=data)

    pass
