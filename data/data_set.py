import tensorflow as tf
import librosa
import numpy as np

class data_set(object):
    """Parse and load the training audio files into a tensorflow dataset"""

    # `labels[i]` is the label for the audio in `filenames[i]`
    def __init__(self, filenames, labels, sess, N = 256, batch_size = 25):
        self.N = N
        self.sess = sess
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(self._parse_file)
        dataset = dataset.shuffle(buffer_size=15000)
        batched_dataset = dataset.batch(batch_size)
        batched_dataset = batched_dataset.repeat()

        self.iterator = batched_dataset.make_one_shot_iterator()
        self.next_element = self.iterator.get_next()

    def _parse_file(self, filename, label):
        audio_arr, sr = librosa.load(filename.eval(self.sess))
        audio_arr = audio_arr[:self.N]
        audio_arr = np.pad(audio_arr, (0,self.N), 'constant')
        audio = tf.convert_to_tensor(audio_arr, tf.float32)
        return audio, label
