import csv

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import os
from pyemd import emd
import gensim.downloader as api
import itertools
import json
import gensim.models

score_word = False


def sentence_ize():
    file_docs = []
    sentences = []
    # for each file, tokenize every sentence.
    for filename in os.listdir('../TRANSCRIPTS/TEXT'):
        with open('../TRANSCRIPTS/TEXT/' + filename) as f:

            tokens = word_tokenize(f.read())
            sentences.append(tokens)
            file_docs.append({'name': filename, 'tokens': tokens})

    print("Text files tokenized.")

    # train the model with new words.
    # taken from https://stackoverflow.com/questions/22121028/update-gensim-word2vec-model
    word_vectors = api.load("glove-wiki-gigaword-100")
    print('word vector loaded')
    model = gensim.models.Word2Vec(sentences=sentences)

    model.build_vocab(word_vectors.index_to_key, update=True)
    print('vocabulary built')
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

    similarity_scores = []
    print('Commencing loop for similarity test.')
    # for every combination in the list, compare to get the wmd.
    # taken from https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.wmdistance.html

    for a, b in itertools.combinations(file_docs, 2):

        sentence_a = [x.lower() for x in a['tokens']]
        sentence_b = [x.lower() for x in b['tokens']]
        print("===========================================")
        print('Starting loop with {a} and {b}'.format(a=a['name'], b=b['name']))
        similarity = model.wv.wmdistance(sentence_a, sentence_b)
        comparison = {
            'a': a['name'].removesuffix('.txt'),
            'b': b['name'].removesuffix('.txt'),
            'similarity': similarity
        }
        similarity_scores.append(comparison)
        print(f"{similarity:.4f}")

    # write array result to json file
    with open('../TRANSCRIPTS/similarity_score.json', 'w') as json_file:
        json.dump(similarity_scores, json_file, indent=4)


def create_ratio_csv(ratio_list):
    ratio_list = sorted(ratio_list, key=lambda k: k['dateDiff'])
    labels = ["names", "likes", "dislikes", "ratio", "date"]
    data = [[ratio['name'], ratio['like'], ratio['dislikes'], ratio['ratio'], ratio['dateDiff']] for ratio in ratio_list]

    with open('../TRANSCRIPTS/ratios.csv', 'w') as h_values:
        write = csv.writer(h_values)
        write.writerow(labels)
        write.writerows(data)


def main():
    if score_word:
        sentence_ize()
    with open('../TRANSCRIPTS/ratio_list.json') as ratio_file:
        ratio_list = json.load(ratio_file)
        create_ratio_csv(ratio_list)


if __name__ == "__main__":  # run the script
    main()


