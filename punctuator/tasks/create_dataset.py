from torchnlp.datasets import iwslt_dataset

from punctuator.src.datasets.utils import make_data
from punctuator.src.path_manager import PathManager


def main():
    train, dev, test = iwslt_dataset(train=True, dev=True, test=True)
    make_data(train, 'en', str(PathManager.RAW / 'train.csv'))
    make_data(dev, 'en', str(PathManager.RAW / 'dev.csv'))
    make_data(test, 'en', str(PathManager.RAW / 'test.csv'))


if __name__ == '__main__':
    main()
