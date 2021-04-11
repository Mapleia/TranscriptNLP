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

    if score_word:
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
            comparison = {'a': a['name'], 'b': b['name'], 'similarity': similarity}
            similarity_scores.append(comparison)
            print(f"{similarity:.4f}")

        # write array result to json file
        with open('../TRANSCRIPTS/similarity_score.json', 'w') as json_file:
            json.dump(similarity_scores, json_file, indent=4)


def make_ratio_list():

    name_list = []
    for filename in os.listdir('../TRANSCRIPTS/TEXT'):
        name_list.append(filename)

    sorted(name_list)
    ratio_list = []
    with open('../TRANSCRIPTS/youtube_vids.json') as youtube:
        youtube_list = json.load(youtube)

        with open('../TRANSCRIPTS/DISLIKES_LIST.json') as dislike:
            dislike_list = json.load(dislike)

            for youtuber in youtube_list:
                likes = dislike_list[youtuber['id']]['likes']
                dislikes = dislike_list[youtuber['id']]['dislikes']
                if likes is not None and dislikes is not None:
                    ratio = likes/dislikes
                    obj = {
                        'id': youtuber['id'],
                        'name': youtuber['name'],
                        'like': likes,
                        'dislikes': dislikes,
                        'ratio': ratio
                    }
                    ratio_list.append(obj)

    with open('../TRANSCRIPTS/ratio_list.json', 'w') as json_file:
        json.dump(ratio_list, json_file, indent=4)

    similarity_scores = []

    for a, b in itertools.combinations(ratio_list, 2):
        if a['ratio'] is not None and b['ratio'] is not None:
            comparison = {
                'a': a['name'],
                'b': b['name'],
                'similarity': abs(a['ratio'] - b['ratio'])
            }
            similarity_scores.append(comparison)

    with open('../TRANSCRIPTS/similarity_score_for_ratios.json', 'w') as json_file:
        json.dump(similarity_scores, json_file, indent=4)


if __name__ == "__main__":  # run the script
    make_ratio_list()

