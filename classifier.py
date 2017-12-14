from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from cnn import CNN
import os
import librosa

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "data/ckpt",
        "Model checkpoint file or directory containing a "
        "model checkpoint file.")
tf.flags.DEFINE_string("input_files", "",
        "Location/file pattern of audio file(s) to classify.")

def print_audio_class(audio_class, sess):
    argmax = tf.argmax(audio_class, 1)
    #print(img_class)
    if tf.equal(argmax, tf.argmax([[1,0,0]], 1)).eval(session=sess):
        print('Audio is of a bee')
    elif tf.equal(argmax, tf.argmax([[0,1,0]], 1)).eval(session=sess):
        print('Audio is of a cricket')
    else:
        print('Audio is of noise')

model = CNN(32768, 3)

with tf.Session() as sess:
    # 1. Build and Restore Model
    model.build()
    sess.run(tf.global_variables_initializer())
    model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_path))

    filenames = []
    for file_pattern in FLAGS.input_files.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))
    print("Running classification on %d files matching %s" % (
                                          len(filenames), FLAGS.input_files))
    for filename in filenames: 
        # 2. Load and Pre-process Audio 
        audio_arr, sr = librosa.load(filename)
        
        # 3. Classify and Print result
        audio_class = sess.run(model.logits,feed_dict={model.x: [audio_arr[:32768]], model.keep_prob: 1.0})
        print_audio_class(audio_class, sess)

