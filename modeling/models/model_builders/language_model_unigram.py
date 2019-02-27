from collections import defaultdict
import math

from collections import defaultdict
import math


def get_words(sentence):
    """
    Get a list of all words which are found in a given sentence.

    Parameters
    ----------
    sentence : str
        The sentence to find words in.

    Returns
    -------
    list
        A list containing all words found in the sentence.
    """
    return sentence.split()


def calculate_distance(word1, word2):
    """
    Calculate the distance between two words. This method could be replaced by a method calculating the Levenshtein
    distance.

    Parameters
    ----------
    word1 : str
        First word.
    word2 : str
        Second word.

    Returns
    -------
    float
        Distance between word1 and word2
    """
    return len(word1.replace(word2, ''))

# Define the training and test data
def get correct sentence(training_sentences, sentence):
    # Find all words in the data
    training_data = [get_words(sentence) for sentence in training_sentences]
    words = get_words(sentence)
    # Show the sentence with spelling errors
    print("Incorrect sentence:\t{0}\n".format(sentence))
    # Count unigrams
    counts = defaultdict(int)
    for sentence_data in training_data:
        for unigram in sentence_data:
            counts[unigram] += 1
    # Normalize probabilities
    totals = sum(counts.values())
    for unigram in counts:
        counts[unigram] /= totals

    # Now check every word in the test sentence
    correct_unigrams = []
    for word in words:
        if counts[word] == 0:
            # The word did not occur in the training data!
            print("Spelling error detected:\t{0}".format(word))
            # Now find a good replacement
            minimum_distance = math.inf
            best_replacement = None
            for vocab_word in counts:
                distance = calculate_distance(vocab_word, word)
                # Make sure the vocabulary word is not the same as the spelling error and make sure it is better than the
                # result found so far
                #if mindistance is equal to distance then take the probability
                if 0 < distance < minimum_distance:
                    minimum_distance = distance
                    best_replacement = vocab_word
                elif distance==minimum_distance:
                    if count[vocab_word]>count[best_replacement]:
                        minimum_distance = distance
                        best_replacement = vocab_word

            # Show the correction
            #print("Best found correct word:\t{0} (distance: {1})\n".format(best_replacement, minimum_distance))
            correct_unigrams.append(best_replacement)
        else:
            correct_unigrams.append(word)
        sentence = ' '.join(correct_unigrams)
        return sentence
