import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler

ceo = pd.read_csv('C:/Users/jtsur/Desktop/iems_308/ceo.csv',header = None)
prc = pd.read_csv('C:/Users/jtsur/Desktop/iems_308/percentage.csv',header = None)
comp = pd.read_csv('C:/Users/jtsur/Desktop/iems_308/companies.csv',header = None)


with open('2014.txt', 'r') as file:
    data_1 = file.read().replace('-', '-').replace('\n', '')

with open('2013.txt', 'r',errors = 'ignore') as file:
    data_2 = file.read().replace('\n', '').replace('-', '-')    

with open('C:/Users/jtsur/Desktop/iems_308/2014/2014-01-02.txt', 'r',errors = 'ignore') as file:
    data_t = file.read().replace('\n', '').replace('-', '-')  
    
st = set(stopwords.words('english'))

prt = PorterStemmer()

data = data_1 + data_2
data = data.lower()

token = [x for x in word_tokenize(data) if x not in st]
token = pd.Series(token)
token = token[token != ',']
token = token[token != '.']
token = token[token != "'s"]
token = token[token != "''"]
token = token[token != "``"]
token = token[token != ":"]
token = token[token != ";"]
token = token[token != "&"]
token = token[token != "`"]
token = token[token != "$"]
token = token[token != "'"]
token = token[token != "'re"]
token = token[token != ")"]
token = token[token != "("]
token = token[token != "["]
token = token[token != "]"]
token = token[token != "@"]
token = token[token != "n't"]
token = token[token != "'ll"] 
token = token[token != "-"]
token = token[token != "?"]
token = token[token != "!"]
token = token[token != "'nt"]
token = token[token != "—"]
token = token[token != "`"]
token = token[token != "'ve"]
token.reset_index(drop = True,inplace = True)

prc = prc[0]
prc[5377] = 'percent'
prc[5378] = '%'
prc[5379] = 'percentage'
prc[5380] = 'points'
comp = comp[0]
ceo.dropna(inplace = True)
ceo = pd.concat([ceo[0],ceo[1]],ignore_index = True)
for i in range(len(ceo)):
    ceo[i] = ceo[i].lower()
for i in range(len(comp)):
    comp[i] = comp[i].lower()
for i in range(len(prc)):
    prc[i] = prc[i].lower()    

dat_ceo = pd.DataFrame({'typ' : np.zeros(len(ceo)), 'lent':np.zeros(len(ceo)),'lem' : np.zeros(len(ceo)) })
dat_prc = pd.DataFrame({'typ' : np.zeros(len(prc)), 'lent':np.zeros(len(prc)),'lem' : np.zeros(len(prc)) })
dat_comp = pd.DataFrame({'typ' : np.zeros(len(comp)), 'lent':np.zeros(len(comp)),'lem' : np.zeros(len(comp)) })


for i in range(len(ceo)):
    dat_ceo.iloc[i,0] = nltk.pos_tag([ceo[i]])[0][1]
    dat_ceo.iloc[i,1] = len(ceo[i])
    if (ceo[i] == prt.stem(ceo[i])):
        dat_ceo.iloc[i,2] = 1
    else:
        dat_ceo.iloc[i,2] = 0 
        
for i in range(len(comp)):
    dat_comp.iloc[i,0] = nltk.pos_tag([comp[i]])[0][1]
    dat_comp.iloc[i,1] = len(comp[i])
    if (comp[i] == prt.stem(comp[i])):
        dat_comp.iloc[i,2] = 1
    else:
        dat_comp.iloc[i,2] = 0 
for i in range(len(prc)):
    dat_prc.iloc[i,0] = nltk.pos_tag([prc[i]])[0][1]
    dat_prc.iloc[i,1] = len(prc[i])
    if (prc[i] == prt.stem(prc[i])):
        dat_prc.iloc[i,2] = 1
    else:
        dat_prc.iloc[i,2] = 0 



data_t = data_t.lower()
data_t = [x for x in word_tokenize(data_t) if x not in st]
data_t = [i for i in data_t if i not in ceo]
data_t = [i for i in data_t if i not in prc]
data_t = [i for i in data_t if i not in comp]
data_t = pd.Series(data_t)
data_t = data_t[data_t != ',']
data_t = data_t[data_t != '.']
data_t = data_t[data_t != "'s"]
data_t = data_t[data_t != "''"]
data_t = data_t[data_t != "``"]
data_t = data_t[data_t != ":"]
data_t = data_t[data_t != ";"]
data_t = data_t[data_t != "&"]
data_t = data_t[data_t != "`"]
data_t = data_t[data_t != "$"]
data_t = data_t[data_t != "'"]
data_t = data_t[data_t != "'re"]
data_t = data_t[data_t != ")"]
data_t = data_t[data_t != "("]
data_t = data_t[data_t != "["]
data_t = data_t[data_t != "]"]
data_t = data_t[data_t != "@"]
data_t = data_t[data_t != "n't"]
data_t = data_t[data_t != "'ll"]
data_t = data_t[data_t != "-"]
data_t = data_t[data_t != "?"]
data_t = data_t[data_t != "!"]
data_t = data_t[data_t != "'nt"]
data_t = data_t[data_t != "—"]
data_t = data_t[data_t != "`"]
data_t = data_t[data_t != "'ve"]
data_t.reset_index(drop = True,inplace = True)


dat_t = pd.DataFrame({'typ' : np.zeros(len(data_t)), 'lent':np.zeros(len(data_t)),'lem' : np.zeros(len(data_t)) })       


for i in range(len(data_t)):
    dat_t.iloc[i,0] = nltk.pos_tag([data_t[i]])[0][1]
    dat_t.iloc[i,1] = len(data_t[i])
    if (data_t[i] == prt.stem(data_t[i])):
        dat_t.iloc[i,2] = 1
    else:
        dat_t.iloc[i,2] = 0         

onehot = pd.get_dummies(dat_t['typ'],prefix = 'typ')
dat_t = pd.concat([dat_t,onehot],axis = 1)
onehot = pd.get_dummies(dat_ceo['typ'],prefix = 'typ')
dat_ceo = pd.concat([dat_ceo,onehot],axis = 1)
onehot = pd.get_dummies(dat_comp['typ'],prefix = 'typ')
dat_comp = pd.concat([dat_comp,onehot],axis = 1)
onehot = pd.get_dummies(dat_prc['typ'],prefix = 'typ')
dat_prc = pd.concat([dat_prc,onehot],axis = 1)

dat_t['indic'] = np.zeros(dat_t.shape[0])
dat_prc['indic'] = np.ones(dat_prc.shape[0])
dat_ceo['indic'] = np.ones(dat_ceo.shape[0])
dat_comp['indic'] = np.ones(dat_comp.shape[0])
dat_t.drop(['typ'],axis = 1,inplace = True)
dat_prc.drop(['typ'],axis = 1,inplace = True)
dat_ceo.drop(['typ'],axis = 1,inplace = True)
dat_comp.drop(['typ'],axis = 1,inplace = True)

dat_ceo = dat_ceo[dat_ceo['lem'] == 1]


ceo_d = pd.concat([dat_ceo,dat_t],ignore_index = True)
comp_d = pd.concat([dat_comp,dat_t],ignore_index = True)
prc_d = pd.concat([dat_prc,dat_t],ignore_index = True)
ceo_y = ceo_d['indic']
comp_y = comp_d['indic']
prc_y = prc_d['indic']
ceo_d.drop(['indic'],axis = 1,inplace = True)
comp_d.drop(['indic'],axis = 1,inplace = True)
prc_d.drop(['indic'],axis = 1,inplace = True)

ceo_d.fillna(0,inplace = True)
prc_d.fillna(0,inplace = True)
comp_d.fillna(0,inplace = True)


le = [len(i) for i in token]
e = nltk.pos_tag(token)
e = [i[1] for i in e]
stm = [prt.stem(i) for i in token]
k = [token == stm]
k = k[0]
k = k.replace(True,1)


dat = pd.DataFrame({'typ' : e, 'lent':le,'lem' : k })


onehot = pd.get_dummies(dat['typ'],prefix = 'typ')
dat = pd.concat([dat,onehot],axis = 1)
dat.drop(['typ'],inplace = True,axis = 1)    

w = [i for i in dat.columns if i not in ceo_d.columns]
dat.drop(w,axis = 1,inplace = True)



ceo_d.drop(['typ_:','typ_CC','typ_DT','typ_MD', 'typ_VBG', 'typ_VBN', 'typ_VBD','typ_VBZ','typ_VBP','typ_WDT'],axis = 1, inplace = True)
comp_d.drop(['typ_:','typ_CC','typ_DT','typ_MD', 'typ_VBG', 'typ_VBN', 'typ_VBD','typ_VBZ','typ_VBP','typ_WDT'],axis = 1, inplace = True)
prc_d.drop(['typ_:','typ_CC','typ_DT','typ_MD', 'typ_VBG', 'typ_VBN', 'typ_VBD','typ_VBZ','typ_VBP','typ_WDT'],axis = 1, inplace = True)
dat.drop(['typ_:','typ_CC','typ_DT','typ_MD', 'typ_VBG', 'typ_VBN', 'typ_VBD','typ_VBZ','typ_VBP','typ_WDT'],axis = 1, inplace = True)

ceo_d.columns = comp_d.columns
dat.columns = pd.Series(['lent', 'lem', 'typ_CD', 'typ_IN', 'typ_JJ', 'typ_JJR', 'typ_JJS',  'typ_NN', 'typ_NNS', 'typ_PRP', 'typ_RB', 'typ_RBR', 'typ_VB'], dtype='object')


normer = MinMaxScaler()
daty = np.array(dat['lent'])
daty = daty.reshape(-1,1)
normer.fit(daty)
dat['lent'] = normer.transform(daty)
ceoy = np.array(ceo_d['lent'])
ceoy = ceoy.reshape(-1,1)
ceo_d['lent'] = normer.transform(ceoy)
compy = np.array(comp_d['lent'])
compy = compy.reshape(-1,1)
comp_d['lent'] = normer.transform(compy)
prcy = np.array(prc_d['lent'])
prcy = prcy.reshape(-1,1)
prc_d['lent'] = normer.transform(prcy)

   
ceo_c = LinearSVC(loss='l2', penalty='l1', dual=False,max_iter = 100000)
ceo_c.fit(ceo_d,ceo_y)
prc_c = LinearSVC(loss='l2', penalty='l1', dual=False,max_iter = 100000)
prc_c.fit(prc_d,prc_y)
comp_c = LinearSVC(loss='l2', penalty='l1', dual=False,max_iter = 100000)
comp_c.fit(comp_d,comp_y)

t_ceo = ceo_c.predict(ceo_d)
ceo_p = sum(t_ceo == ceo_y)/len(ceo_y)
t_prc = prc_c.predict(prc_d)
prc_p = sum(t_prc == prc_y)/len(prc_y)
t_comp = comp_c.predict(comp_d)
comp_p = sum(t_comp == comp_y)/len(comp_y)

ceo_n = ceo_c.predict(dat)
comp_n = comp_c.predict(dat)
prc_n = prc_c.predict(dat)

prc_1 = pd.Series(prc_n)
prc_1 = prc_1[prc_1 == 1]
e1 = prc_1.index
prc_fin = token[e1]
prc_fin = ' '.join(prc_fin)
comp_1 = pd.Series(comp_n)
comp_1 = comp_1[comp_1 == 1]
e2 = comp_1.index
comp_fin = token[e2]
comp_fin = ' '.join(comp_fin)
ceo_1 = pd.Series(ceo_n)
ceo_1 = ceo_1[ceo_1 == 1]
e3 = ceo_1.index
ceo_fin = token[e3]
ceo_fin = ' '.join(ceo_fin)

ceo_w = open('ceo.txt','w')
ceo_w.write(ceo_fin)
ceo_w.close()
comp_w = open('comp.txt','w')
comp_w.write(comp_fin)
comp_w.close()
prc_w = open('prc.txt','w')
prc_w.write(prc_fin)
prc_w.close()