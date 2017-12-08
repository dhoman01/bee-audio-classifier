import tensorflow as tf
import librosa

class data_set(object):
    """Parse and load the training audio files into a tensorflow dataset"""

    # `labels[i]` is the label for the audio in `filenames[i]`
    def __init__(self, filenames, labels, N = 256, batch_size = 25):
        self.N = N
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(lambda filename, label: tuple(tf.py_func(self._parse_file, [filename, label], [tf.float32, label.dtype])))
        dataset = dataset.shuffle(buffer_size=500)
        batched_dataset = dataset.batch(batch_size)
        batched_dataset = batched_dataset.repeat()

        self.iterator = batched_dataset.make_one_shot_iterator()
        self.next_element = self.iterator.get_next()

    def _parse_file(self, filename, label):
        audio_arr, sr = librosa.load(filename)
        return audio_arr[:self.N], label
