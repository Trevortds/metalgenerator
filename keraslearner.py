
# coding: utf-8

# ## Initializing stuff

# In[19]:

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import numpy as np

# I'm making this from Trung Tran's LSTM tutorial. 
# I'm going to try to annotate it in my own words so that I can understand what's happening
# And then change it to do what I want. 

data = ['i', ' ', 'h', 'a', 'v', 'e', ' ', 'a', ' ', 'd', 'r', 'e', 'a', 'm']

# features. This will eventually be words
chars = list(set(data))





# ## Preparing data

# In[20]:


# conversion to numbers. This is actually a really clever solution
ix_to_char = {ix:char for ix, char in enumerate(chars)}
char_to_ix = {char:ix for ix, char in enumerate(chars)}

# setting up parameters. 
num_features = len(chars)
# length of the group of words that the lstm will be shown at a time
len_sequence = 6
num_sequences = len(data)//len_sequence


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

# In[23]:

# initialize a sequential network
model = Sequential()

hidden_dim = 500
layer_num = 2

# add an initial lstm layer
# I don't know why the input shape needs to be a tuple
# the return-sequences parameter makes it give you multiple outputs
model.add(LSTM(hidden_dim, input_shape=(None, num_features), return_sequences=True))

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


# ## Training

# In[30]:

# # I don't know if nb is supposed to mean something. This is just a counter

# nb_epoch = 0
# batch_size = 50
# generate_length = 50

# while True:
#     print("\n\n")
#     # fit the model for one epoch
#     model.fit(X, y, batch_size=batch_size, verbose=1, nb_epoch=10)
#     # increment counter
#     nb_epoch += 1
#     # every epoch, show some text examples.
#     # this is a function defined below.
# #     generate_text(model, generate_length)
    
#     if nb_epoch % 10 == 0:
#         # save every tenth epoch
#         print("epoch # {}".format(nb_epoch*10))
#         generate_text(model, generate_length)
#         model.save_weights('checkpoint_{}_epoch{}.hdf5'.format(hidden_dim, nb_epoch))


# ## Text Generation

# In[27]:

# begin with some random characters and predict the next n characters

def generate_text(model, length):
    # generate a number and associated character
    ix = [np.random.randint(num_features)]
    y_char = [ix_to_char[ix[-1]]]
    # annoyingly, the big matrix of character sequences is called X
    X = np.zeros((1, length, num_features))
    
    for i in range(length):
        # for n characters
        # update the big matrix with the last prediction
        X[0, i, :][ix[-1]] = 1
        # print the last prediction to the command line
        print(ix_to_char[ix[-1]], end="")
        # get a new prediction
        # I don't know what most of these arguments are. 
        # I don't understand why the prediction needs to be subscripted
        # I guess the 1 means that you only get one output from argmax? 
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        
        # convert to character and append to array
        y_char.append(ix_to_char[ix[-1]])
    
    return ('').join(y_char)


# In[ ]:

# I don't know if nb is supposed to mean something. This is just a counter

nb_epoch = 0
batch_size = 50
generate_length = 50

while True:
    # print("\n\n") 
    # fit the model for one epoch
    model.fit(X, y, batch_size=batch_size, verbose=0, nb_epoch=100)
    # increment counter
    nb_epoch += 1
    # every epoch, show some text examples.
    # this is a function defined below.
    generate_text(model, generate_length)
    
    if nb_epoch % 10 == 0:
        # save every tenth epoch
        print("epoch # {}".format(nb_epoch*100))
        generate_text(model, generate_length)
        model.save_weights('checkpoint_{}_epoch{}.hdf5'.format(hidden_dim, nb_epoch))

