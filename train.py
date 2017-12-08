import tensorflow as tf
from data.data_set import data_set
from cnn import CNN
from ann import ANN
import os
from progress.bar import Bar

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("train_dir", "data/train",
        "Directory containing the training data")
tf.flags.DEFINE_string("test_dir", "data/test",
        "Directory containing the testing data")
tf.flags.DEFINE_string("checkpoint_path", "data/ckpt",
        "File, file pattern, or directory of checkpoints")
tf.flags.DEFINE_integer("epochs", 80000,
        "Number of training iterations")
tf.flags.DEFINE_integer("save_step", 1000,
        "Number of steps in between checkpoints")
tf.flags.DEFINE_integer("test_steps", 16000,
        "Number of steps to run the test and build accuracy")
tf.flags.DEFINE_boolean("train", True,
        "Flag for training the model")
tf.flags.DEFINE_boolean("test", True,
        "Flag for testing the model")
tf.flags.DEFINE_integer("num_hidden", 3,
        "The number of hidden layers in the input")
tf.flags.DEFINE_integer("N", 32768,
        "The length of the input data (will be padded/trunc)")
tf.flags.DEFINE_integer("hidden_nodes", 256,
        "The number of hidden layer nodes")
tf.flags.DEFINE_integer("num_classes", 3,
        "The number of classes")
tf.flags.DEFINE_integer("batch_size", 5,
        "Batch Size")

def add_filenames_and_labels(fn_arr, lbl_arr, label, dir):
    for file in os.listdir(dir):
        if file.endswith(".wav"):
            fn_arr.append(os.path.join(dir, file))
            lbl_arr.append(label)

train_filenames = []
train_labels = []
add_filenames_and_labels(train_filenames, train_labels, [1,0,0], os.path.join(FLAGS.train_dir, "bee"))
add_filenames_and_labels(train_filenames, train_labels, [0,1,0], os.path.join(FLAGS.train_dir, "cricket"))
add_filenames_and_labels(train_filenames, train_labels, [0,0,1], os.path.join(FLAGS.train_dir, "noise"))

test_filenames= []
test_labels = []
add_filenames_and_labels(test_filenames, test_labels, [1,0,0], os.path.join(FLAGS.test_dir, "bee"))
add_filenames_and_labels(test_filenames, test_labels, [0,1,0], os.path.join(FLAGS.test_dir, "cricket"))
add_filenames_and_labels(test_filenames, test_labels, [0,0,1], os.path.join(FLAGS.test_dir, "noise"))

model = CNN(FLAGS.N, FLAGS.num_classes)
#model = ANN(FLAGS.N, FLAGS.num_hidden, FLAGS.hidden_nodes, FLAGS.num_classes)

train_data = data_set(tf.constant(train_filenames), tf.constant(train_labels), FLAGS.N, FLAGS.batch_size)
test_data = data_set(tf.constant(test_filenames), tf.constant(test_labels), FLAGS.N, FLAGS.batch_size)

def train_model(sess):
    """Train the model on the given training data for the given number of epochs in FLAGS"""
    print('Training Model')
    bar = Bar("Training", max=FLAGS.epochs, suffix='%(percent)d%% - %(eta)ds')
    for i in range(FLAGS.epochs):
        train_audio, train_lbl = sess.run(train_data.next_element)
        if i % FLAGS.save_step == 0:
            train_accuracy = model.accuracy.eval(feed_dict={model.x: train_audio, model.y_: train_lbl, model.keep_prob: 1.0})
            print('\nstep %d, training accuracy %g' % (i, train_accuracy))
            model.saver.save(sess, os.path.join(FLAGS.checkpoint_path, "model.ckpt"), i)
        model.train_step.run(feed_dict={model.x: train_audio, model.y_: train_lbl, model.keep_prob: 0.6})
        bar.next()
    bar.finish()
    save_path = model.saver.save(sess, os.path.join(FLAGS.checkpoint_path, "model.ckpt"), FLAGS.epochs)
    print('Model saved at ' + save_path)

def test_model(sess):
    """Test the model on the given testing data"""
    print('Testing Model')
    bar = Bar("Testing", max=FLAGS.test_steps, suffix='%(percent)d%% - %(eta)ds')
    total_acc = 0
    for i in range(FLAGS.test_steps):
        test_audio, test_labels = sess.run(test_data.next_element)
        acc = model.accuracy.eval(feed_dict={model.x: test_audio, model.y_: test_labels, model.keep_prob: 1.0})
        if i % int(FLAGS.test_steps * .1) == 0:
            print('\ntest accuracy %g' % acc)
        total_acc += acc
        bar.next()
    bar.finish()
    print('Avg. test accuracy %g' % (total_acc / FLAGS.test_steps))

with tf.Session() as sess:
    print('Building Model')
    model.build()
    sess.run(tf.global_variables_initializer())
    print('Trying to restore model')
    try:
        model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_path))
        print('Model restored from checkpoint')
    except:
        print('Couldn\'t load checkpoint from ' + FLAGS.checkpoint_path)

    if FLAGS.train:
        train_model(sess)
    if FLAGS.test:
        test_model(sess)

