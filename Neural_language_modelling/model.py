import torch 
import torch.nn as nn 
import torch.optim as optimizer


class GRU(nn.Module):
  def __init__(self,embedding_dim,vocab_size,hidden_size):
    super().__init__()
    self.embedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
    self.gru=nn.GRU(input_size=embedding_dim,hidden_size=hidden_size,batch_first=True)
    self.dropout=nn.Dropout(0.3)
    self.fc1=nn.Linear(hidden_size,vocab_size)
    # self.fc2=nn.Linear(128,vocab_size)

  def forward(self,x):
    x=self.embedding(x)
    output,hidden=self.gru(x)
    result=output[:,-1,:]
    result=self.dropout(result)
    result=self.fc1(result)
    # result=self.fc2(result)

    return result

class VanillaRNN(nn.Module):
  def __init__(self,vocab_size,embedding_dim,hidden_size):
    super().__init__()
    self.embedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
    self.rnn=nn.RNN(input_size=embedding_dim,hidden_size=hidden_size,batch_first=True)
    self.fc1=nn.Linear(hidden_size,vocab_size)
    self.dropout=nn.Dropout(0.3)
    # self.fc2=nn.Linear(64,vocab_size)

  def forward(self,x):
    x=self.embedding(x)
    output,hidden=self.rnn(x)
    result=output[:,-1,:]
    result=self.dropout(result)
    result=self.fc1(result)
    # result=self.fc2(result)

    return result  

class LSTMModel(nn.Module):
  def __init__(self,vocab_size,embedding_dim,hidden_size):
    super().__init__()
    self.embedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
    self.lstm=nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,batch_first=True)
    self.fc1=nn.Linear(hidden_size,vocab_size)
    self.dropout=nn.Dropout(0.3)
    # self.fc2=nn.Linear(64,vocab_size)

  def forward(self,x):
    x=self.embedding(x)
    output,(hidden,cell)=self.lstm(x)
    result=output[:,-1,:]
    result=self.dropout(result)
    result=self.fc1(result)
    # result=self.fc2(result)

    return result  