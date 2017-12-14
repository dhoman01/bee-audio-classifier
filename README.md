## Bee Audio Classifier

### Training
Place the training data in your desired directory (default is data/train/[bee|cricket|noise]). Place your testing data into your desired directory (default is data/test/[bee|cricket|noise]) NOTE: To generate test data run `create_test_data.py`. Finally, run ``python3 train.py``` to train the model. Inspect `train.py` to view optional flags.

### Classification
Place the checkpoint into your desired dir (default is data/ckpt/). Run `python3 classifier.py --input_files /path/to/input_file.wav`.

### Accuracy
Running entire test dataset on my PC resulted in 71% accuraccy.
 
