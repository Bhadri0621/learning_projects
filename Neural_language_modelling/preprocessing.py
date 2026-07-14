import re
import torch
from vocabulary import word_to_index,index_to_word
def removing(x):
    text=x.lower()
    text=re.sub(r'\s+',' ',text).strip()
    text=re.sub(r'\\','',text)
    text=text.split()

    return text

def preprocessing(inp):
    input=removing(inp)
    vocabulary=set()
    for word in input:
        vocabulary.update(word)
    seq_length=3
    for i,word in enumerate(vocabulary):
       word_to_index[word]=i
       index_to_word[i]=word   

    num_corpus=[] 
    for word in input:
        num_corpus.append(word_to_index[word])
    seq_len=30
    padded=[0]*(seq_len-len(num_corpus))
    num_corpus=padded+num_corpus
    in_tensor=torch.tensor(num_corpus,dtype=torch.long)
    in_tensor=in_tensor.unsqueeze(0)
 
    return in_tensor
                      