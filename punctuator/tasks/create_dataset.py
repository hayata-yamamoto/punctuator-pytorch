import json

import pandas as pd
from sklearn.model_selection import train_test_split

from punctuator.src.core.path_manager import PathManager
from punctuator.src.datasets import ted_data, make_records


def main():
    series = ted_data()['transcript']

    train, val = train_test_split(series)
    pd.DataFrame(train).to_csv(PathManager.INTERIM / 'train.csv')
    pd.DataFrame(val).to_csv(PathManager.INTERIM / 'val.csv')

    train = make_records(train)
    val = make_records(val)

    with open(PathManager.PROCESSED / 'train.json', 'w') as f:
        json.dump(train, f)
    with open(PathManager.PROCESSED / 'val.json', 'w') as f:
        json.dump(val, f)


if __name__ == '__main__':
    main()
