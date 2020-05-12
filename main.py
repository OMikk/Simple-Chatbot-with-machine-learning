import random
import json
import tensorflow
import tflearn
import numpy as np
import nltk§
import pickle
from os import path
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# constant
TRAIN_MODEL = False


with open("intents.json") as file:
    data = json.load(file)

# try to use saved data
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

# otherwise collect data for training model
except:
    # create lists§
    # words
    words = []
    # labels
    labels = []
    # list of patterns
    docs_x = []
    # words and groups
    docs_y = []

    ## loop trough intents.json to get data & word roots with stemming  ##

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            words_ex = nltk.word_tokenize(pattern)
            words.extend(words_ex)
            docs_x.append(words_ex)
            docs_y.append(intent["group"])

        if intent["group"] not in labels:
            labels.append(intent["group"])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]

    # remove duplicate elements
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    # create bag of words if word exist put 1 if not put 0
    for x, doc in enumerate(docs_x):
        bag = []

        words_ex = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in words_ex:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # create numpy arrays for training
    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

## training model neural network taking in bag of words outputs label & group response##

tensorflow.reset_default_graph()
# lenght of the training model
net = tflearn.input_data(shape=[None, len(training[0])])
# neuron hidden layers
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
# output layer with softmax activation (probability)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)


model = tflearn.DNN(net)


# if True train the model if False use existing one
if TRAIN_MODEL == False:
    model.load("model.tflearn")

# otherwise construct the model
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

    # generate & convert BoW to numpy array


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
# generate BoW
    for sw in s_words:
        for i, w in enumerate(words):
            if w == sw:
                bag[i] = 1

    return np.array(bag)

# function to ask user for questions


def talk():
    print("Chippy is happy to help you with all you need. Just ask a question! (type quit to exit the chat)")
    while True:
        ip = input("You: ")
        if ip.lower() == "quit":
            break
    # predict the group of which the question belongs to
        results = model.predict([bag_of_words(ip, words)])[0]
        results_index = np.argmax(results)
        group = labels[results_index]

        # if probability over 70% answer the question
        if results[results_index] > 0.7:
            for grp in data["intents"]:
                if grp['group'] == group:
                    responses = grp["responses"]
            print(random.choice(responses))
        else:
            print("Sorry I don't understand, try again.")

        # pick matching response
        # for grp in data["intents"]:
        #     if grp['group'] == group:
        #         responses = grp["responses"]
        # print(random.choice(responses))
        # else:
        #     print("I didn't get that.")
        # print(random.choice(responses))
talk()
