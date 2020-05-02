import numpy as np
import matplotlib.pyplot as plt
import pickle
import xgboost as xgb

imp = pickle.load(open('./figures/xgboost_imp.p', 'rb'))

labels = np.flip(['Journal', 'Year', 'Avg Abs 6M', 'Abs Close Count', 'Avg Body', 'Avg Abs', 'Body Close Count', 'Avg Body 6M'])

plot = xgb.plot_importance(imp)
plot.set_yticklabels(labels)
plot.set_xticks([])
plot.grid(False)
plot
plt.box(False)
plt.ylabel('')
plt.xlabel('F score', position=(0.35,0))
plt.suptitle('XGBoost Feature Importance', fontsize=12)
plt.title('Variational Autoencoder Embedding                            .',y=0.92, fontsize=10)
plt.tight_layout()
plt.savefig('./figures/xgboost_feature_importance.png')
plt.show()

