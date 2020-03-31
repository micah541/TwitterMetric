import pandas as pd
import time
import sys
from collections import Counter
from twitter import *
import pickle

config = {}
execfile("config.py", config)
twitter = Twitter(auth = OAuth(config["access_key"], config["access_secret"], config["consumer_key"], config["consumer_secret"]))

##names_list was a list created elsewhere

df = pd.DataFrame(columns = ['screen_name', 'ids'])
df2 = pd.read_csv("names_list.csv", encoding = 'utf-8')
names = list(df2['screen_name'])


for l in names:
    try : query = twitter.friends.ids(screen_name = l)
    except : 
        if '401' in sys.exc_info()[1][0]: print "probably a protected account"
        else: 
            print sys.exc_info()[1][0]
            time.sleep(60)
            continue
    print l
    time.sleep(10)
    ids = query['ids']
    df.loc[len(df)] = [l, ids] 
    print len(df)
    time.sleep(50)

import pickle
pickle.dump(df, open("friends_list8765.pickle", "wb"))


### in another window

import pickle 
df = pickle.load(open("friends_list8765.pickle", "rb"))


from collections import Counter
id_list = list(df['ids'])
all_ids = [item for sublist in id_list for item in sublist]
c = Counter(all_ids)
top5000 = [k[0] for k in c.most_common(5000)]
#next we reduce the lists

reduced_list = [[a for a in l if a in top5000] for l in id_list]  #this took a few minutes
targets = [[top5000.index(p) for p in x] for x in reduced_list]
screen_names = df['screen_name']
sn_index = range(len(screen_names))
    


from keras.layers import LSTM, Dense, Dropout, Embedding, Masking
from keras.models import Sequential, Model
import numpy as np
import random

E = len(id_list)  
N = 5000
Widt = 800
Dropt = .7

model1 = Sequential()
model1.add(
    Embedding(
    input_dim=E,
    output_dim=8,
    weights=None,
    trainable=True))
model1.add(Dense(Widt, activation='sigmoid'))
model1.add(Dropout(Dropt))
model1.add(Dense(Widt, activation='sigmoid'))
model1.add(Dropout(Dropt))
model1.add(Dense(Widt, activation='sigmoid'))
model1.add(Dropout(Dropt))
model1.add(Dense(N, activation='sigmoid') )
model1.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def one_hot_target(lisn, N):
    vector = np.zeros(N)
    for i in lisn: vector[i]=1 
    return(vector)

def train_random(how_many):
    TS = 3500   
    for i in range(how_many):
        rs = random.sample(range(E),TS)
        X = rs
        y = [targets[x] for x in rs]
        y = [one_hot_target(p, N) for p in y]
        y = np.asarray(y)
        X = np.asarray(X)
        X = np.resize(X, [TS, 1])
        y = np.resize(y, [TS, 1, N])
        model1.fit(X,y, epochs = 2, batch_size = 600, verbose = 1)

def snap_shot():
    ssdf = pd.DataFrame(model1.get_layer(index = 0).get_weights()[0])
    ssdf['screen_name'] = screen_names
    ssdf.to_csv("snapshot8765.csv", index=False)


##in another window

import pandas as pd
from sklearn.metrics import euclidean_distances
import numpy as np


###for some reason I had two KimJobke in the file.  This accidentally gave me a test to see how "done" it was.  When the two KimJobke were very close, I was satisfied it had trained long enough


df = pd.read_csv("snapshot8765.csv")
names = list(df['screen_name'])
names[3997]= 'KimJobke2'
df['screen_name'] = names
df.index = df['screen_name'] 




def get_closest(sn, n):
    vector = df.loc[sn].as_matrix(columns = df.columns[:-1]).reshape(-1,1).transpose()
    matrix = df.as_matrix(columns = df.columns[:-1])#.transpose()
    distances = euclidean_distances(vector, matrix)
    idxs = np.argsort(distances)[::-1]
    sorted_dists = [distances[0][ix] for ix in idxs[0]]
    closest = [names[i] for i in list(idxs)[0][:n]]
    #print closest
    return (closest, sorted_dists[:n])    





def get_page(sn, n):  #produces an html code block
    closest, distanc = get_closest(sn, n)
    html_chunk = ''
    html_chunk+='\n'
    html_chunk+='<a name='
    html_chunk+=sn
    html_chunk+='>@'
    html_chunk+=sn
    html_chunk+='   '
    for k in range(n):
        name = closest[k]
        html_chunk+='<a href = "http://www.twitter.com/'
        html_chunk+= name
        html_chunk+= '">@'
        html_chunk+=  name  
        html_chunk+=  '</a>'
        html_chunk+=  '<a href = #'
        html_chunk+=  name 
        html_chunk+= '> # </a>'
        html_chunk+=' distance: '+str("%.2f" % round(distanc[k],2))+'  |  '
    return(html_chunk) 


page = open('results.html', 'w')
page.write("Instructions:  clicking the username brings you to twitter page.  Clicking the # brings you to their entry on this page.")
page.write("\n <hr><hr>")


for nam in names:
    try : next = get_page(nam, 15)
    except : 
        print "error with", nam
        continue
    page.write(next)
    page.write('<hr>')

page.close()
    
    
    




