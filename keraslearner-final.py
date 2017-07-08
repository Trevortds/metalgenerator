
# coding: utf-8

# ## Initializing stuff
# 
# Changes: adding word tokenize, making the "chars" vector contain words rather than actual chars. Yes, this is confusing, but hopefully if I say it right up front it won't be so bad. 

# In[58]:

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict
import sys
import re




pronouncing_dict = cmudict.dict()
def nsyl(word):
    if word not in pronouncing_dict:
        if re.search('\W', word):
            # if the word has non-word characters
            return 0
        else:
            # take a guess: number of vowels
            return len(re.findall("[aeiou]", word))
    return [len(list(y for y in x if y[-1].isdigit())) for x in pronouncing_dict[word.lower()]][0]


# I'm making this from Trung Tran's LSTM tutorial. 
# I'm going to try to annotate it in my own words so that I can understand what's happening
# And then change it to do what I want. 

data = []
avgsyllables = 0
linecount = 0
characterlstm = True

with open("lyricsfixed.txt", 'r') as f:

    if characterlstm:
        for line in f:
            data += list(line)

    else:
        for line in f:
            
            for word in word_tokenize(line):
                data.append(word.lower())
                avgsyllables += nsyl(word)
            linecount += 1
            data.append("\n") # I want the machine to learn newlines
            # since they're part of the lyrics
            # and have similar meaning to words. 

        avgsyllables = avgsyllables / linecount

# features. This is actually words not characters. 
# also, this line is what killed my results. I trained a model for a full day
# it was performing great. 
chars = sorted(list(set(data)))





# ## Preparing data
# 
# Only change here is the len_sequence. I made it 700 words, it'll be learning from multiple songs at once. 

# In[59]:




# conversion to numbers. This is actually a really clever solution
ix_to_char = {ix:char for ix, char in enumerate(chars)}
char_to_ix = {char:ix for ix, char in enumerate(chars)}

# setting up parameters. 
num_features = len(chars)
# length of the group of words that the lstm will be shown at a time
len_sequence = 50
num_sequences = len(data)//len_sequence

print("features: ", num_features)
print(chars)
print("sequence length: ", len_sequence)
print("number of sequences: ",  num_sequences)


# need input and output tapes for the LSTM
X = np.zeros((len(data)//len_sequence, len_sequence, num_features))
y = np.zeros((len(data)//len_sequence, len_sequence, num_features))

# for each of the sequences
for i in range(0, len(data)//len_sequence):
    # select the characters in the data that correspond to this sequence
    X_sequence = data[i*len_sequence:(i+1)*len_sequence]
    # convert to numeric
    X_sequence_ix = [char_to_ix[value] for value in X_sequence]
    #initialize 
    input_sequence = np.zeros((len_sequence, num_features))
    for j in range(len_sequence):
        #make a 1-hot vector for each of the letters the lstm is being shown
        input_sequence[j][X_sequence_ix[j]] = 1

    X[i] = input_sequence
    
    # select targets: the symbol that follows
    y_sequence = data[i*len_sequence+1:(i+1)*len_sequence+1]
    # convert to numeric
    y_sequence_ix = [char_to_ix[value] for value in y_sequence]
    target_sequence = np.zeros((len_sequence, num_features))
    for j in range(len_sequence):
        target_sequence[j][y_sequence_ix[j]] = 1
        
    y[i] = target_sequence
    
    


# ## Setting up network
# 
# Taking cues from Tran, I set it up to have 700 hidden states, with 0.3 dropout at the first layer and three layers

# In[60]:
# initialize a sequential network
model = Sequential()

hidden_dim = 700
layer_num = 3

# add an initial lstm layer
# I don't know why the input shape needs to be a tuple
# the return-sequences parameter makes it give you multiple outputs
model.add(LSTM(hidden_dim, 
               input_shape=(None, num_features), 
               return_sequences=True))
model.add(Dropout(0.3))

# add more layers
# I don't really know what adding a layer to an LSTM means, 
#  they're only ever shown with one.

for i in range(layer_num-1):
    model.add(LSTM(hidden_dim, return_sequences=True))

# I don't see why the dense layer is necessary, but Tran says it is.
# and in order to get the dense layer to work, a time distributed layer
#  needs to go between.

model.add(TimeDistributed(Dense(num_features)))

# pick an activation for this layer
model.add(Activation('softmax'))

# pick a loss function and optimization method. 

model.compile(loss="categorical_crossentropy", optimizer="rmsprop")



# ## Heuristic
# 
# This is the part where I get to be creative. 
# 
# 

# In[88]:

def random_pick(model_predictions):
    # Select random character from the best ten
    candidates = np.argpartition(model_predictions[0,-1], -10)[-10:]
    return [np.random.choice(candidates)]
    
def softmax_pick(model_predictions, top_ten = False):
    candidates = np.argpartition(model_predictions[0,-1], -10)[-10:]
    probs = [model_predictions[0,-1, i] for i in candidates]
    probs = softmax(np.asarray(probs), 0.4)
    # print(model_predictions)
    # print(candidates)
    # print(probs)
    if top_ten:
        return [np.random.choice(candidates, p=probs)]
    else:
        return [np.random.choice(model_predictions.shape[2], p=model_predictions[0,-1])]


def beam_search(model_predictions, history, i):
    # Select the character most likely to lead down a good path   
    best_ix = 0
    candidates = np.argpartition(model_predictions[0,-1], -20)[-20:]
    # print(model_predictions.shape) (1, 1, num_features)
    # print(np.argmax(model_predictions[0], 1))
    # print(candidates)
    candidate_predictions = [model_predictions[0, -1, c] for c in candidates]
    # print(candidate_predictions)
    # print(normalize(np.asarray([candidate_predictions])))
    
    h = np.zeros(len(candidates))
    
    for i in range(len(candidates)):
        candidate = candidates[i]
        pred = candidate_predictions[i]
        word = ix_to_char[candidate]
        # print("candidate: {} score: {} word:{}".format(candidate, model_predictions[0,0,candidate], word))

        h[i] = heuristic(history, word, pred)
        # print("adjucted score: ", h[i])
        # if score < best_score:
        #     best_score = score
        #     best_candidate = candidate
    # print("scores: ", h)
    # softmax preserves ranking but squishes values
    # temperature lets you play with how far the values change
    # if > 1 rank 1 is further from rank 2
    # if < 1 rank 1 is closer to rank 2
    h = softmax(h, 0.2) # 0 is 
    # print("softmax scores: ", h)
    return [np.random.choice(candidates, p=h)]
    
    # return [best_candidate]
    # return np.argmax(model_predictions[0], 1)

def heuristic(history, word, prediction):
    # don't give more than two newlines in a row. 
    # maximizing heuristic
    if word == '\n' and len(history) > 1 and history[-2] == '\n':
        return 1000
    
    line = []
    syl_count = 0
    for back_word in history[::-1]:
        if back_word == '\n':
            break
        else:
            syl_count = nsyl(back_word)
            line.insert(0, back_word)
    syl_count += nsyl(word)
    # print("syllable count ", syl_count)
    # return square difference between number 
    # of syllables in the sentence and the average
    # this may cause a preference for long words at the beginning
    # we'll see. 
    
    return prediction * (syl_count - avgsyllables)**-2
    

# copied from http://stackoverflow.com/questions/41902047/how-to-calculate-robust-softmax-function-with-temperature?noredirect=1&lq=1
def softmax(x, tau):
    """ Returns softmax probabilities with temperature tau
        Input:  x -- 1-dimensional array
        Output: s -- 1-dimensional array
    """
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()




# ## Text Generation

# In[32]:

# begin with some random characters and predict the next n characters

def generate_text(model, length, generator="softmax", seed=""):
    if seed == "":
        # generate a number and associated character
        ix = [np.random.randint(num_features)]
        y_char = [ix_to_char[ix[-1]]]
        
        # annoyingly, the big matrix of character sequences is called X
        X = np.zeros((1, length, num_features))
    else: 
        print(seed, end="")
        ix = [char_to_ix[seed[0]]]
        y_char = []
        X = np.zeros((1, length+len(seed), num_features))
        for i in range(len(seed)):
            ix = [char_to_ix[seed[i]]]
            y_char += [ix_to_char[ix[-1]]]
            X[0, i, :][ix[-1]] = 1


    joinchar = '' if characterlstm else ' '
    
    for i in range(len(seed), length):
        # for n characters
        # update the big matrix with the last prediction
        X[0, i, :][ix[-1]] = 1
        # print the last prediction to the command line
        print(ix_to_char[ix[-1]], end=joinchar)
        # get a new prediction
        # instead of taking the max, run beam search 
        if characterlstm:
            if generator == "softmax":
                ix = softmax_pick(model.predict(X[:, :i+1, :]), top_ten=True)
            elif generator == "argmax":
                ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
            else:
                raise NameError("Invalid generator function")

        else:
            ix = beam_search(model.predict(X[:, :i+1, :]), y_char, i)
        
        # convert to character and append to array
        y_char.append(ix_to_char[ix[-1]])
    

    return (joinchar).join(y_char)


# ## Training

# In[15]:

# I don't know if nb is supposed to mean something. This is just a counter

if len(sys.argv) > 1:
    model.load_weights(sys.argv[1])
    print(re.findall(r'epoch(\d+)', sys.argv[1]))
    nb_epoch = int(re.findall(r'epoch(\d+)', sys.argv[1])[0])
    print("Using weights in {}".format(sys.argv[1]))
    print("epoch: {}".format(nb_epoch)) 
    print("new version!")
else:
    print("Using a new model")
    nb_epoch = 0
batch_size = 40
generate_length = 100
epoch_per_gen = 1

while True:
    print("\nEpoch {} generation \n".format(nb_epoch))
    print("\n\n")

    print("\n---------with seed---------\n")
    generate_text(model, generate_length, seed="I will rule", generator="argmax")

    print("\n--------Softmax------------\n")
    # every epoch, show some text examples.
    # I moved this above fitting to make debugging easier. 
    # this is a function defined below.
    # TODO break generate_text and load_model out into another file. 
    generate_text(model, generate_length)
    print("\n---------Argmax------------\n")
    generate_text(model, generate_length, generator="argmax")
    print("\n\n")
    # fit the model for one epoch  
    model.fit(X, y, batch_size=batch_size, verbose=1, nb_epoch=1)
    # increment counter
    nb_epoch += 1
    
    if nb_epoch % 10 == 0:
        # save every tenth epoch group
        print("save epoch # {}".format(nb_epoch))
        generate_text(model, generate_length)
        model.save_weights('checkpoints/checkpoint_{}_epoch{}.hdf5'.format(hidden_dim, nb_epoch))


# In[ ]:



