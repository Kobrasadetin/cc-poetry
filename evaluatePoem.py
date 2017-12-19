import re
from pathlib import Path
import evaluate.therex as TheRex
import nltk
import numpy
nltk.download('cmudict')
# input: poem split by rows. For example:
# lorem ipsum
# dolor sit amet
# = [['lorem', ipsum'],['dolor','sit','amet']]
#
# Function checks, in this example, if "ipsum" and "amet" rhymes.
# output:

def rhyming(text_sentences):
    def rhyme(inp, level):
        entries = nltk.corpus.cmudict.entries()
        syllables = [(word, syl) for word, syl in entries if word == inp]

        rhymes = []
        for (word, syllable) in syllables:
            rhymes += [word for word, pron in entries if pron[-level:] == syllable[-level:]]

        return set(rhymes)

    def unique(a):
        b = a.ravel().view(numpy.dtype((numpy.void, a.dtype.itemsize * a.shape[1])))
        _, unique_idx = numpy.unique(b, return_index=True)

        new_a = a[numpy.sort(unique_idx)]
        return new_a

    def match(word1, word2):
        # first, we don't want to report 'glue' and 'unglue' as rhyming words
        if word1.find(word2) == len(word1) - len(word2):
            return False
        if word2.find(word1) == len(word2) - len(word1):
            return False

        filepath = Path("rhyming_words.npy")

        if filepath.exists():
            rhyming_words = numpy.load("rhyming_words.npy")
        else:
            rhyming_words = {}

        pair = ";".join([word1, word2])

        if pair not in rhyming_words.item():
            print("Saving the pair...")
            do_they_rhyme = word1 in rhyme(word2, 1)
            rhyming_words.item()[pair] = do_they_rhyme
            numpy.save("rhyming_words", rhyming_words)
        else:
            do_they_rhyme = rhyming_words.item()[pair]

        return do_they_rhyme


    text_sentences = [sentence.split(" ") for sentence in text_sentences]

    last_words = []
    for i in range(len(text_sentences)):
        sentence = text_sentences[i]
        sentence = [re.sub("\W+", "", word) for word in sentence]
        sentence = [word for word in sentence if word != ""]
        if sentence:
            last_words.append(sentence[-1])
    # Remove empty strings from the list

    rhymes=0
    rhymed_pairs = []
    for i in range(len(last_words)-1):
        word = last_words[i]
        for j in range(i+1, min(i+5, len(last_words))):
            next_word = last_words[j]

            if match(word, next_word):
                #print("True: %s, %s" % (word, next_word))
                rhymes +=1
                rhymed_pairs.append(word + ", " + next_word)

    print("Rhymes: %s, Rows: %s, %% of rhymes compared to all: %s" % (rhymes, len(last_words), round(rhymes/len(last_words), ndigits=3)))
    print(rhymed_pairs)

    #print(text_sentences)
    print(last_words)

def common_categories(text_words):
    ''''Check how many words are from same category in the input string
    text_words: input poem as a string'''
    tr = TheRex.TheRex()

    filepath = Path("common_categories.npy")

    if filepath.exists():
        common_cat = numpy.load("common_categories.npy")
        common_cat = common_cat.item()
    else:
        common_cat = {}

    # tag each word
    tagged = nltk.pos_tag(text_words)

    # remove all other words except nouns
    nouns = [word for word in tagged if word[1] == 'NN']
    nouns = [word[0] for word in nouns]

    # for each concept word in list of nouns:
    #
    #common_cat={'cat':[1,2,3], 'dog':[5,5]}

    all_values = [item for sublist in common_cat.values() for item in sublist]
    print_cat = {}
    for concept in nouns:

        if concept not in all_values:
            dict = tr.member(concept)
            print(dict)

            if dict:
                categories = dict['categories'].keys()
                for category in categories:
                    if category not in common_cat:
                        common_cat[category] = []

                    if concept not in common_cat[category]:
                        common_cat[category].append(concept)

                    if item not in print_cat:
                        print_cat[item] = []
                    else:
                        print_cat[item].append(concept)


            numpy.save("common_categories.npy", common_cat)
        else:
            print("Concept in saved dict.")
            concept_categories = [concepts for concepts in common_cat if concept in concepts]
            for item in concept_categories:
                if item not in print_cat:
                    print_cat[item] = []
                    print_cat[item].append(concept)
                else:
                    print_cat[item].append(concept)
    print(print_cat)
    print_cat = {k: v for k, v in print_cat.items() if len(v) > 1}

    print("print_cat")
    print(print_cat)
    print("----")
    print(common_cat.keys())

    similar_categories1 = {k: v for k, v in common_cat.items() if len(v) > 1}

    print(similar_categories1)

    print("Count of categories presented in poem: %s" % len(common_cat.keys()))
    print("Count of categories that contain >1 words from poem: %s " % len(similar_categories1))