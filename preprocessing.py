import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from tensorflow.contrib.keras import preprocessing


def get_property(smi):

    try:
        mol=Chem.MolFromSmiles(smi) 
        property = [Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol), QED.qed(mol)]
        
    except:
        property = 'invalid'
           
    return property
    

def canonocalize(smi):

    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def vectorize(list_input, char_set):

    one_hot = np.zeros((list_input.shape[0], list_input.shape[1]+4, len(char_set)), dtype=np.int32)

    for si, ss in enumerate(list_input):
        for cj, cc in enumerate(ss):
            one_hot[si,cj+1,cc] = 1

        one_hot[si,-1,0] = 1
        one_hot[si,-2,0] = 1
        one_hot[si,-3,0] = 1

    return one_hot[:,0:-1,:], one_hot[:,1:,:]


def smiles_to_seq(smiles, char_set):

    char_to_int = dict((c,i) for i,c in enumerate(char_set))
    
    list_seq=[]
    for s in smiles:
        seq=[]                
        j=0
        while j<len(s):
            if j<len(s)-1 and s[j:j+2] in char_set:
                seq.append(char_to_int[s[j:j+2]])
                j=j+2
    
            elif s[j] in char_set:
                seq.append(char_to_int[s[j]])
                j=j+1
    
        list_seq.append(seq)
    
    list_seq = preprocessing.sequence.pad_sequences(list_seq, padding='post')
    
    return list_seq