from punctuator.src.path_manager import PathManager
import pandas as pd
from collections import Counter
import re


def parse(sent: str) -> pd.Series:
    if not isinstance(sent, str):
        raise TypeError(f"sent must be string, not {type(sent)}")
    if sent == "":
        return pd.Series(Counter())
    cnt = Counter([s.split('###')[1].replace('#', '') for s in sent.split(" ")])
    if cnt['#O'] > 0:
        print(sent)
        print([s.split('###')[1] for s in sent.split(" ")])
    return pd.Series(cnt)


def main() -> None:
    names = ['train.csv', 'dev.csv', 'test.csv']
    for name in names:
        df = pd.read_csv(PathManager.RAW / name).fillna('')
        s = df['0'].apply(parse).fillna(0).sum()
        print('----------------')
        print(f'     {name}     ')
        print('----------------')
        print(df.shape)
        print(s)
        print(s.sum())
        print('\n')


if __name__ == '__main__':
    main()
