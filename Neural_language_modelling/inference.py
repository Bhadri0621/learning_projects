from model import GRU,LSTMModel,VanillaRNN
import torch
from vocabulary import index_to_word
device='cpu'

hidden_size=96
embedding_dim=64
vocab_size=len(index_to_word)
def prediction(input,model):
    if model.lower()=='lstm':
        model=LSTMModel(vocab_size=vocab_size,embedding_dim=embedding_dim,hidden_size=hidden_size)
        model.load_state_dict(
          torch.load('lstm_best_model.wt',
                     map_location=device)
                    )
            
    elif model.lower()=='gru':
        model=GRU(vocab_size=vocab_size,embedding_dim=96,hidden_size=192)
        model.load_state_dict(
            torch.load('GRUbestmodel.wt',
                       map_location=device)
        )
    else:
        model=VanillaRNN(vocab_size=vocab_size,embedding_dim=embedding_dim,hidden_size=hidden_size)       
        model.load_state_dict(
            torch.load('vanillarnn.wt',
                       map_location=device)
        ) 
    
    model.eval()
    with torch.no_grad():
         prediction=model(input)
         pred_classes=torch.argmax(prediction,dim=1).item()
         maxword=index_to_word[pred_classes]
         k=3
         top_logits,top_indices=torch.topk(prediction,k=k,dim=1)
         top_probs=torch.softmax(top_logits,dim=-1)
         
         sample_pos=torch.multinomial(top_probs,num_samples=1)
         real_idx=torch.gather(top_indices,dim=-1,index=sample_pos).item()
         word=index_to_word[real_idx]


    return word,maxword