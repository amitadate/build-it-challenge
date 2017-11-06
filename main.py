import sklearn
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split
import numpy
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf

Tweet = pandas.read_csv("Tweets.csv")

def tweet_to_words(raw_tweet):
    letters_only = re.sub("[^a-zA-Z@]", " ",raw_tweet)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops and not re.match("^[@]", w) and not re.match("flight",w)]
    return( " ".join( meaningful_words ))

#Pre-processing on the tweets
Tweet['clean_tweet']=Tweet['text'].apply(lambda x: tweet_to_words(x))
#Convert sentiment to binary
Tweet['sentiment'] = Tweet['airline_sentiment'].apply(lambda x: 0 if x == 'negative' else 1)

#Join all the words and generate a corpus
all_text = ' '.join(Tweet['clean_tweet'])
words = all_text.split()

# Convert words to integers
from collections import Counter
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

tweet_ints = []
for each in Tweet['clean_tweet']:
    tweet_ints.append([vocab_to_int[word] for word in each.split()])

#Create a list of labels
labels = np.array([0 if each == 'negative' else 1 for each in Tweet['airline_sentiment'][:]])

#Find the number of tweets with zero length after the data pre-processing
tweet_len = Counter([len(x) for x in tweet_ints])
print("Zero-length reviews: {}".format(tweet_len[0]))
print("Maximum review length: {}".format(max(tweet_len)))

#Remove those tweets with zero length and its correspoding label
tweet_idx  = [idx for idx,tweet in enumerate(tweet_ints) if len(tweet) > 0]
labels = labels[tweet_idx]
Tweet = Tweet.ix[tweet_idx]
tweet_ints = [tweet for tweet in tweet_ints if len(tweet) > 0]

seq_len = max(tweet_len)
features = np.zeros((len(tweet_ints), seq_len), dtype=int)
for i, row in enumerate(tweet_ints):
    features[i, -len(row):] = np.array(row)[:seq_len]

split_frac = 0.8
split_idx = int(len(features)*0.8)
train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y = labels[:split_idx], labels[split_idx:]

test_idx = int(len(val_x)*0.5)
val_x, test_x = val_x[:test_idx], val_x[test_idx:]
val_y, test_y = val_y[:test_idx], val_y[test_idx:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

print("Train set: \t\t{}".format(train_y.shape),
      "\nValidation set: \t{}".format(val_y.shape),
      "\nTest set: \t\t{}".format(test_y.shape))

lstm_size = 256
lstm_layers = 1
batch_size = 100
learning_rate = 0.001

# Create input placeholders
n_words = len(vocab_to_int)
# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# Embedding 
embed_size = 300

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)

# LSTM
with graph.as_default():

    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)

    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)

#RNN
with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                             initial_state=initial_state)

#Output - Final output of the RNN layer will be used for sentiment prediction.

with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_, predictions)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#Graph for checking Validation accuracy
with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Pick only full batches of data and return based on the batch_size
def get_batches(x, y, batch_size=100):
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


epochs = 10

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)

        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 0.5,
                    initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

            if iteration % 5 == 0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))

            if iteration % 25 == 0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration += 1
    saver.save(sess, "D:/Projects/Constalytics/checkpoints/twitter_sentiment.ckpt")

# Test

test_acc = []
test_pred = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state= sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
        prediction = tf.cast(tf.round(predictions),tf.int32)
        prediction = sess.run(prediction,feed_dict=feed)
        test_pred.append(prediction)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))

#Use the tweet sentiment predicted for the data in the test set

test_pred_flat = (np.array(test_pred)).flatten()
start_idx = len(train_x) + len(val_x)
end_idx = start_idx + len(test_pred_flat)+1
Tweet.loc[start_idx:end_idx,'predicted_sentiment'] = test_pred_flat

# further can be visualized as per the installed dependencies.
