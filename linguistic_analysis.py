import numpy as np
import math
from numpy import dot
from numpy.linalg import norm

common_words = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with",
    "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
    "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just",
    "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back",
    "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us"
]

def analyse_tfidf(rows, data_to_analyse):
    words = [doc.lower().split() for doc in rows]
    vocab = sorted(set(sum(words, [])))
    vocab_dict = {k:i for i,k in enumerate(vocab)}

    # term frequencies:
    X_tf = np.zeros((len(data_to_analyse), len(vocab)), dtype=int)
    for i,elem in enumerate(data_to_analyse):
        for word in elem.split():
            if word in vocab_dict.keys():
                X_tf[i, vocab_dict[word]] += 1

    idf = np.zeros((len(vocab)), dtype=float)
    for i in range(len(idf)):
        count = 0
        for j in range(len(X_tf)):
            if X_tf[j, i] > 0:
                count += 1
        if count == 0:
            idf[i] = 1
        else:
            idf[i] = math.log(len(X_tf)/ count)
    
    # TFIDF
    X_tfidf = X_tf * idf

    norms = np.zeros((len(data_to_analyse)), dtype=float)
    for i in range(len(norms)):
        norm = np.linalg.norm(X_tfidf[i])
        norms[i] = norm
    
    return norms



def analyse_tfidf_cosine(rows, data_to_analyse, remove_common):
    words = [doc.lower().split() for doc in rows]
    vocab = set(sum(words, []))
    if(remove_common):
        vocab = vocab - set(common_words)
    vocab_list = sorted(vocab)
    vocab_dict = {k:i for i,k in enumerate(vocab)}

    # term frequencies:
    X_tf = np.zeros((len(data_to_analyse), len(vocab_list)), dtype=int)
    for i,elem in enumerate(data_to_analyse):
        for word in elem.lower().split():
            if word in vocab_dict.keys():
                X_tf[i, vocab_dict[word]] += 1

    idf = np.zeros((len(vocab_list)), dtype=float)
    for i in range(len(idf)):
        count = 0
        for j in range(len(X_tf)):
            if X_tf[j, i] > 0:
                count += 1
        if count == 0:
            idf[i] = 1
        else:
            idf[i] = math.log(len(X_tf)/ count)
    
    # TFIDF
    X_tfidf = X_tf * idf

    vector_one = np.ones((len(vocab_list)), dtype=float)

    cosinuses = np.zeros((len(data_to_analyse)), dtype=float)

    for i in range(len(data_to_analyse)):
        cosinuses[i] = dot(X_tfidf[i], vector_one) / (norm(X_tfidf[i]) * norm(vector_one))
    
    return cosinuses



def analyse_n_grams(rows, data_to_analyse, n, remove_common):
    n_grams = []
    n_gram_data_to_analyse_list = []
    characters_to_remove = ',. /'
    for doc in rows:
        split_text = doc.lower().translate(str.maketrans({'.': '', ',': ''})).split()
        doc_n_gram = [' '.join(split_text[i:i+n]) for i in range(0,len(split_text),n)]
        n_grams.extend(doc_n_gram)

    for doc in data_to_analyse:
        split_text = doc.lower().translate(str.maketrans({'.': '', ',': ''})).split()
        doc_n_gram = [' '.join(split_text[i:i+n]) for i in range(0,len(split_text),n)]
        n_gram_data_to_analyse_list.append(doc_n_gram)
    
    n_gram_dict = {k:i for i,k in enumerate(n_grams)}
    
    # term frequencies:
    X_tf = np.zeros((len(data_to_analyse), len(n_grams)), dtype=int)
    for i,elem in enumerate(n_gram_data_to_analyse_list):
        for word in elem:
            if word in n_gram_dict.keys():
                X_tf[i, n_gram_dict[word]] += 1

    idf = np.zeros((len(n_grams)), dtype=float)
    for i in range(len(idf)):
        count = 0
        for j in range(len(X_tf)):
            if X_tf[j, i] > 0:
                count += 1
        if count == 0:
            idf[i] = 1
        else:
            idf[i] = math.log(len(X_tf)/ count)
    
    # TFIDF
    X_tfidf = X_tf * idf

    vector_one = np.ones((len(n_grams)), dtype=float)

    cosinuses = np.zeros((len(data_to_analyse)), dtype=float)

    for i in range(len(data_to_analyse)):
        if norm(X_tfidf[i]) != 0:
            cosinuses[i] = dot(X_tfidf[i], vector_one) / (norm(X_tfidf[i]) * norm(vector_one))
        else: 
            cosinuses[i] = 0
    
    return cosinuses