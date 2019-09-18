from sklearn.model_selection import train_test_split

from punctuator.src.datasets import ted_data, write_txt, make_dataset
from punctuator.src.core.path_manager import PathManager


def main():
    series = ted_data()['transcript']

    train = make_dataset(series)

    write_txt(train, PathManager.PROCESSED / 'train.txt')


if __name__ == '__main__':
    main()
