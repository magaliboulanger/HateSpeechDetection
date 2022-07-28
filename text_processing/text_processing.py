from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

class TextProcessing: 

  def __init__(self):
    self.model = SentenceTransformer('all-MiniLM-L6-v2')

  def get_similarity(self, sentences_train, sentence_dev):
    sentences=[d['text'] for d in sentences_train]
    embeddings_train = self.model.encode(sentences, convert_to_tensor=True)
    embeddings_dev = self.model.encode(sentence_dev, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings_train, embeddings_dev)

    similarity=[]
    for i in range(len(sentences_train)):
      similarity.append({"cluster": sentences_train[i]["n_cluster"],"name": sentences_train[i]["img"], "label": sentences_train[i]["label"], "score": "{:.4f}".format(cosine_scores[i][0])})

    similarity = sorted(similarity, key = lambda x: x['score'], reverse=True)
    return similarity