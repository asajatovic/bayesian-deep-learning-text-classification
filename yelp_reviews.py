import csv
import os

from torchtext import data


class YELP(data.Dataset):

    urls = [('https://drive.google.com/uc?export=download&'
             'id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0',
             'yelp_review_full_csv.tar.gz')]
    name = 'yelp'
    dirname = 'yelp_review_full_csv'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, fine_grained=True, **kwargs):
        """Create an Yelp 2015 dataset instance given a path and fields.

        Arguments:
            path: Path to the data file
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            fine_grained: Whether to use 5-class instead of 3-class
                labeling. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('label', label_field), ('text', text_field)]

        def get_label_str(label):
            pre = 'very ' if fine_grained else ''
            return {'1': pre + 'negative', '2': 'negative', '3': 'neutral',
                    '4': 'positive', '5': pre + 'positive', None: None}[label]

        label_field.preprocessing = data.Pipeline(get_label_str)
        with open(os.path.expanduser(path), encoding="utf8") as f:
            reader = csv.reader(f)
            examples = [data.Example.fromlist(row, fields) for row in reader]

        super(YELP, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='.data',
               train='train.csv', test='test.csv', **kwargs):
        """Create dataset objects for splits of the Yelp dataset.


        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: Root dataset storage directory. Default is '.data'.
            train: The filename of the train data. Default: 'train.csv'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'test.csv'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        return super(YELP, cls).splits(
            root=root, text_field=text_field, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.data', vectors=None, **kwargs):
        """Create iterator objects for splits of the Yelp dataset.


        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that contains the Yelp dataset subdirectory.
            vectors: one of the available pretrained vectors or a list with each
                element one of the available pretrained vectors (see Vocab.load_vectors)
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, test), batch_size=batch_size, device=device)
