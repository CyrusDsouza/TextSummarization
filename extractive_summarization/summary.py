# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import nltk
# nltk.download('punkt') # one time execution
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import os
# from keras.models import Model
# from keras.layers import Input, LSTM, Dense   

stop_words = stopwords.words('english')

word_embeddings = {}

with open('glove.6B.50d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs

class Summarize(object):

    def __init__(self, text):
        self.text = text

    def textRank(self, threshold_summary):
        # function to remove stopwords
        def _remove_stopwords(sen):
            sen_new = " ".join([i for i in sen if i not in stop_words])
            return sen_new

        # remove punctuations, numbers and special characters
        clean_sentences = pd.Series(self.text).str.replace("[^a-zA-Z]", " ")

        # make alphabets lowercase
        clean_sentences = [s.lower() for s in clean_sentences]

        clean_sentences = [_remove_stopwords(r.split()) for r in clean_sentences]

        sentence_vectors = []
        for i in clean_sentences:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((50,1))) for w in i.split()])/(len(i.split())+0.001)
                v = v.flatten()
            else:
                v = np.zeros((50,1))
            sentence_vectors.append(np.array(v))

        sim_mat = np.zeros([len(self.text), len(self.text)])
        for i in range(len(self.text)):
            for j in range(len(self.text)):
                if i != j:
                    try:
                        sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,50), sentence_vectors[j].reshape(1,50))[0,0]
                    except :
                        sim_mat[i][j] = 0.0 

        nx_graph = nx.from_numpy_matrix(sim_mat)
        scores = nx.pagerank(nx_graph)

        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(self.text)), reverse=True)

        print("{star} Extractive Summary from native TextRank {star}".format(star = "-"*50))
        for i in range(threshold_summary):
            print(ranked_sentences[i][1])
        print("{star}{star}".format(star = "-"*50))


    def gensim(self, threshold_summary, split = False):
        from gensim.summarization import summarize  
        print("{star} Extractive Summary from GENSIM {star}".format(star = "-"*50))
        print(summarize('\t'.join(self.text), threshold_summary, split = split))
        print("{star}{star}".format(star = "-"*50))


if __name__ == "__main__":
    # df = pd.read_csv("tennis_articles_v4.csv")
    # sentences = []
    text = """For the second time during his papacy, Pope Francis has announced a new group of bishops and archbishops set to become cardinals -- and they come from all over the world.Pope Francis said Sunday that he would hold a meeting of cardinals on February 14 "during which I will name 15 new Cardinals who, coming from 13 countries from every continent, manifest the indissoluble links between the Church of Rome and the particular Churches present in the world," according to Vatican Radio.New cardinals are always important because they set the tone in the church and also elect the next pope, CNN Senior Vatican Analyst John L. Allen said. They are sometimes referred to as the princes of the Catholic Church.The new cardinals come from countries such as Ethiopia, New Zealand and Myanmar."This is a pope who very much wants to reach out to people on the margins, and you clearly see that in this set," Allen said. "You're talking about cardinals from typically overlooked places, like Cape Verde, the Pacific island of Tonga, Panama, Thailand, Uruguay." But for the second time since Francis' election, no Americans made the list. "Francis' pattern is very clear: He wants to go to the geographical peripheries rather than places that are already top-heavy with cardinals," Allen said.Christopher Bellitto, a professor of church history at Kean University in New Jersey, noted that Francis announced his new slate of cardinals on the Catholic Feast of the Epiphany, which commemorates the visit of the Magi to Jesus' birthplace in Bethlehem."On feast of three wise men from far away, the Pope's choices for cardinal say that every local church deserves a place at the big table. "In other words, Francis wants a more decentralized church and wants to hear reform ideas from small communities that sit far from Catholicism's power centers, Bellitto said.That doesn't mean Francis is the first pontiff to appoint cardinals from the developing world, though. Beginning in the 1920s, an increasing number of Latin American churchmen were named cardinals, and in the 1960s, St. John XXIII, whom Francis canonized last year, appointed the first cardinals from Japan, the Philippines and Africa. In addition to the 15 new cardinals Francis named on Sunday, five retired archbishops and bishops will also be honored as cardinals.Last year, Pope Francis appointed 19 new cardinals, including bishops from Haiti and Burkina Faso. CNN's Daniel Burke and Christabelle Fombu contributed to this report."""