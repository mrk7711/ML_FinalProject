
def load_data(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.gz'% kind)
    images_path = os.path.join(path,
                               '%s-images.gz'% kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

X_train, y_train = load_data('../Project', kind='train')
X_test, y_test = load_data('../Project', kind='test')

