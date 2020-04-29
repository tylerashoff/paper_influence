
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.layers import Input, Dense, Lambda
from keras import backend as K


class embeddings():

    def __init__(self):
        pass

    def sampling(self, args):

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    def var_autoencoder(self, x_train, x_test, dim = 2):
        original_dim = x_train.shape[1]
        # network parameters
        input_shape = (original_dim, )
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
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(latent_dim,),
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
        
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)
        
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        
        vae.compile(optimizer='adam')

        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        
        latent_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
        return(latent_mean)

    def stress_r2(self, X1, X2):
        ds = np.triu(cdist(X1, X1)).flatten()
        fs = np.triu(cdist(X2, X2)).flatten()
        
        ds = ds[ds!=0]
        fs = fs[fs!=0]
        
        stress = np.sqrt(np.sum((fs-ds)**2)/np.sum(ds**2))
        r2 = np.corrcoef(fs,ds)[0,1]
        return(stress, r2)
        
    def plotting(self, X_fit, embed='', stress=0, r2=0,
                 show=True, filename=None):
        fig = plt.figure()
        plt.style.use('tableau-colorblind10')
        ax = fig.add_subplot(111, projection='3d')
        for ind in range(X_fit.shape[0]):
            ax.scatter(X_fit[ind, 0],
                       X_fit[ind, 1],
                       X_fit[ind, 2])
            pass
        ax.legend(loc="upper right")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.title(embed+' Embedding\n'+
                  'Kruskal stress: '+str(round(stress, 4)) +
                  ', R-squared: '+str(round(r2, 4)),
                  loc = 'left')
        if filename is not None:
            plt.savefig('./final/'+filename+'.png')
            pass
        if show:
            plt.show()
            pass
        pass
    
    def main(self, data = None):

        embeded = self.var_autoencoder(data, data, dim = 3)
        return(embeded)
    pass


df = pd.read_csv('./code/full_data.csv').to_numpy()

cites = df[:, 0]
pmcid = df[:, 1]
refs = df[:, 2]
abs_vecs = df[:, 3: 103]
body_vecs = df[:, 103:]

emb = embeddings()
data = abs_vecs
#abs_emb = emb.var_autoencoder(data, data, dim = 3)
#emb.plotting(abs_emb)
