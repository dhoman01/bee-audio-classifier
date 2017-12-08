import tensorflow as tf
import os
import random
from progress.bar import Bar

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("train_dir", "data/train",
        "The directory containing all of the files")
tf.app.flags.DEFINE_string("test_dir", "data/test",
        "The directory to save the test files")
tf.app.flags.DEFINE_integer("k", 20,
        "Percent of train files to become test files")
tf.app.flags.DEFINE_boolean("do_move", False,
	"Actually move the files via `os.rename`")

def get_filenames(dir):
    fn_arr = []
    for file in os.listdir(dir):
        if file.endswith(".wav"):
            fn_arr.append(file)

    return fn_arr

def get_test_arr(train_arr):
    k = FLAGS.k * .01
    random.shuffle(train_arr)
    return train_arr[:int(len(train_arr) *  k)]

def rename_files(arr, dir):
    bar = Bar('Moving ' + dir, max=len(arr))
    for filename in arr:
        current = os.path.join(os.path.join(FLAGS.train_dir, dir), filename)
        new = os.path.join(os.path.join(FLAGS.test_dir, dir), filename)
        if FLAGS.do_move:
	    os.rename(current, new)
	bar.next()
    bar.finish()

bee_train_filenames = get_filenames(os.path.join(FLAGS.train_dir, "bee"))
cricket_train_filenames = get_filenames(os.path.join(FLAGS.train_dir, "cricket"))
noise_train_filenames = get_filenames(os.path.join(FLAGS.train_dir, "noise"))

bee_test_filenames = get_test_arr(bee_train_filenames)
cricket_test_filenames = get_test_arr(cricket_train_filenames)
noise_test_filenames = get_test_arr(noise_train_filenames)

rename_files(bee_test_filenames, "bee")
rename_files(cricket_test_filenames, "cricket")
rename_files(noise_test_filenames, "noise")

