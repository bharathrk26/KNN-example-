import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
plt.style.use('ggplot')
data=pd.read_csv('https://raw.githubusercontent.com/yash91sharma/IMDB-Movie-Dataset-Analysis/master/movie_metadata.csv')
newdata=data.iloc[:,4:6]
newdata=newdata.dropna()
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=5)
kmeans.fit(newdata)
unique,counts=np.unique(kmeans.labels_,return_counts=True)
newdata['cluster']=kmeans.labels_
sns.set_style('whitegrid')
sns.lmplot('director_facebook_likes','actor_3_facebook_likes',data=newdata,hue='cluster',palette='coolwarm',size=0,aspect=1,fit_reg=False)

