import pickle as pkl


with open('word_to_index.pkl','rb') as f:
  word_to_index=pkl.load(f)

with open('index_to_word.pkl','rb') as f:
  index_to_word=pkl.load(f)


print(list(word_to_index)[:10])
print(list(index_to_word)[:10])

