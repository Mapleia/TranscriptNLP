import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import os
from pyemd import emd
import gensim.downloader as api
import itertools
import json

word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data


def sentence_ize():
    file_docs = []
    sentences = []
    # for each file, tokenize every sentence.
    for filename in os.listdir('TRANSCRIPTS/TEXT'):
        with open('TRANSCRIPTS/TEXT/' + filename) as f:

            tokens = sent_tokenize(f.read())
            sentences = sentences + tokens
            file_docs = file_docs.append({'name': filename, 'tokens': tokens})

    # train the model with new words.
    word_vectors.build_vocab(sentences, update=True)
    word_vectors.train(sentences)

    print("Text files tokenized.")
    similarity_scores = []

    print('Commencing loop for similarity test.')
    # for every combination in the list, compare to get the wmd.
    for a, b in itertools.combinations(file_docs, 2):
        sentence_a = a['tokens'].lower().split()
        sentence_b = b['tokens'].lower().split()
        print("===========================================")
        print('Starting loop with {a} and {b}'.format(a=a['name'], b=b['name']))
        similarity = word_vectors.wmdistance(sentence_a, sentence_b)
        comparison = {'a': a['name'], 'b': b['name'], 'similarity': similarity}
        similarity_scores.append(comparison)
        print(f"{similarity:.4f}")

    # write array result to json file
    with open('similarity_score.json', 'w') as json_file:
        json.dump(similarity_scores, json_file, indent=4)


if __name__ == "__main__":  # run the script
    sentence_ize()
