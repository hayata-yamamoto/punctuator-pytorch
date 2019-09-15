from sklearn.model_selection import train_test_split

from punctuator.src.datasets import ted_data, write_txt, make_dataset


def main():
    series = ted_data()['transcript']

    rest, test = train_test_split(series)
    train, val = train_test_split(rest)

    train = make_dataset(train)
    val = make_dataset(val)
    test = make_dataset(test)

    write_txt(train, 'train.txt')
    write_txt(val, 'val.txt')
    write_txt(test, 'test.txt')


if __name__ == '__main__':
    main()
