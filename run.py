import numpy as np
import pandas as pds
from preprocessing import smiles_to_seq, vectorize
import SSVAE

from preprocessing import get_property, canonocalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


# pre-defined parameters
frac=0.5
beta=10000.
char_set=[' ','1','2','3','4','5','6','7','8','9','-','#','(',')','[',']','+','=','B','Br','c','C','Cl','F','H','I','N','n','O','o','P','p','S','s','Si','Sn']
data_uri='./data/ZINC_310k.csv'
save_uri='./zinc_model.ckpt'

ntrn=300000
frac_val=0.05
ntst=10000


# data preparation
print '::: data preparation'

smiles = pds.read_csv(data_uri).as_matrix()[:ntrn+ntst,0] #0: SMILES
Y = np.asarray(pds.read_csv(data_uri).as_matrix()[:ntrn+ntst,1:], dtype=np.float32) # 1: MolWT, 2: LogP, 3: QED 

list_seq = smiles_to_seq(smiles, char_set)
Xs, X=vectorize(list_seq, char_set)

tstX=X[-ntst:]
tstXs=Xs[-ntst:]
tstY=Y[-ntst:]

X=X[:ntrn]
Xs=Xs[:ntrn]
Y=Y[:ntrn]

nL=int(len(Y)*frac)
nU=len(Y)-nL
nL_trn=int(nL*(1-frac_val))
nL_val=nL-nL_trn
nU_trn=int(nU*(1-frac_val))
nU_val=nU-nU_trn
perm_id=np.random.permutation(len(Y))

trnX_L=X[perm_id[:nL_trn]]
trnXs_L=Xs[perm_id[:nL_trn]]
trnY_L=Y[perm_id[:nL_trn]]

valX_L=X[perm_id[nL_trn:nL_trn+nL_val]]
valXs_L=Xs[perm_id[nL_trn:nL_trn+nL_val]]
valY_L=Y[perm_id[nL_trn:nL_trn+nL_val]]

trnX_U=X[perm_id[nL_trn+nL_val:nL_trn+nL_val+nU_trn]]
trnXs_U=Xs[perm_id[nL_trn+nL_val:nL_trn+nL_val+nU_trn]]

valX_U=X[perm_id[nL_trn+nL_val+nU_trn:]]
valXs_U=Xs[perm_id[nL_trn+nL_val+nU_trn:]]

scaler_Y = StandardScaler()
scaler_Y.fit(Y)
trnY_L=scaler_Y.transform(trnY_L)
valY_L=scaler_Y.transform(valY_L)


## model training
print '::: model training'

seqlen_x = X.shape[1]
dim_x = X.shape[2]
dim_y = Y.shape[1]
dim_z = 100
dim_h = 250

n_hidden = 3
batch_size = 200

model = SSVAE.Model(seqlen_x = seqlen_x, dim_x = dim_x, dim_y = dim_y, dim_z = dim_z, dim_h = dim_h,
                    n_hidden = n_hidden, batch_size = batch_size, beta = float(beta), char_set = char_set)

with model.session:
    model.train(trnX_L=trnX_L, trnXs_L=trnXs_L, trnY_L=trnY_L, trnX_U=trnX_U, trnXs_U=trnXs_U,
                valX_L=valX_L, valXs_L=valXs_L, valY_L=valY_L, valX_U=valX_U, valXs_U=valXs_U)
    model.saver.save(model.session, save_uri)

    ## property prediction performance
    tstY_hat=scaler_Y.inverse_transform(model.predict(tstX))

    for j in range(dim_y):
        print [j, mean_absolute_error(tstY[:,j], tstY_hat[:,j])]
        
        
    ## unconditional generation
    for t in range(10):
        smi = model.sampling_unconditional()
        print t, smi, get_property(smi)
    
    ## conditional generation (e.g. MolWt=250)
    yid = 0
    ytarget = 250.
    ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])
    
    for t in range(10):
        smi = model.sampling_conditional(yid, ytarget_transform)
        print t, smi, get_property(smi)