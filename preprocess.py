import pandas as pd
import numpy as np
from scipy.sparse import hstack

from lib.log import *
from lib.utils import optimize_dataframe

NROWS = None


def __load_data():
    # Загрузим данные.

    logf('load train...')
    train = optimize_dataframe(pd.read_csv('../data/train.csv', nrows=NROWS))

    logf('load test...')
    test = optimize_dataframe(pd.read_csv('../data/test.csv', nrows=NROWS))


def preprocess(df):
    # ### Определение корректности ФИО
    # Начнём с первой задачи. Сгенерируем признаки на основе tf-idf-преобразования.
    # Используем две версии: по словам (чтобы поймать популярные имена) и по тройкам символов
    # (чтобы поймать опечатки).
    logf('vectorize words...')
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 1))
    words = vectorizer_word.fit_transform(df.fullname)

    logf('vectorize chars...')
    vectorizer_char = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), min_df=5)
    chars = vectorizer_char.fit_transform(df.fullname)

    # Объединим признаки в одну матрицу.
    logf('join features...')

    return hstack([words, chars])


def predict(model, x_test, y_test, s_test, test):
    logf('predict...')
    from sklearn.metrics import recall_score, precision_score, f1_score

    y_predict = model.predict(x_test)
    logf('Recall:', recall_score(y_test, y_predict, average='macro'))
    logf('Precision:', precision_score(y_test, y_predict, average='macro'))
    logf('F1:', f1_score(y_test, y_predict, average='macro'))

    test['target'] = model.predict(s_test)


def spelling(train, test):
    # ### Исправление опечаток
    # Для коррекции опечаток воспользуемся open-source библиотекой
    # ([github](https://github.com/mammothb/symspellpy)). Можно установить через pip.
    import symspellpy
    symspell = symspellpy.SymSpell()

    # Подготовим обучающую выборку для корректора. Добавим туда весь корректный train.
    # На выходе нам нужно отдать файл с частотами слов.
    logf('prepare train data...')
    train.loc[train.target != 1, 'fullname_true'] = train.loc[train.target != 1, 'fullname']
    dicts = [name for person in train.fullname_true for name in person.split(' ')]

    from collections import Counter
    name_freq = Counter(dicts)

    logf('save dictionary...')
    with open('dictionary.txt', 'w') as f:
        for name, freq in name_freq.items():
            f.write('{} {}\n'.format(name, freq))

    # Загрузим словарь в модель.
    logf('load dictionary...')
    symspell.load_dictionary('dictionary.txt', term_index=0, count_index=1)

    # Будем проводить коррекцию по словам.
    def correct(s):
        def correct_word(w):
            tmp = symspell.lookup(w, symspellpy.Verbosity.CLOSEST)
            if len(tmp):
                return tmp[0].term.upper()
            else:
                return w

        return ' '.join([correct_word(word) for word in s.split(' ')])

    logf('spell correction...')
    train_1 = train.loc[train.target == 1].copy()
    train_1['fullname_corrected'] = train_1.fullname.apply(correct)

    np.mean(train_1.fullname_true == train_1.fullname_corrected)

    # Скорректируем тестовую выборку.
    test['fullname_true'] = None
    test.loc[test.target == 1, 'fullname_true'] = test.loc[test.target == 1, 'fullname'].apply(correct)

    # Сохраним итоговый файл.
    logf('save submission...')
    test[['id', 'target', 'fullname_true']].to_csv('submission.csv', index=False)

