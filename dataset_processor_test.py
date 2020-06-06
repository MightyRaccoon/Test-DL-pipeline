import os
from collections import defaultdict

import numpy as np


def test_file():
    source_dir = '/home/mightyracoon/datasets/place_365/data_256'

    train_dir = '/home/mightyracoon/datasets/place_365/train_set'
    val_dir = '/home/mightyracoon/datasets/place_365/val_set'
    test_dir = '/home/mightyracoon/datasets/place_365/test_set'

    source_dir_files_count = sum(map(lambda directory: len(directory[2]), os.walk(source_dir)))

    if os.path.exists(val_dir) and os.path.exists(test_dir):
        copied_files_count = sum(map(lambda directory: len(directory[2]), os.listdir(train_dir))) + \
                             sum(map(lambda directory: len(directory[2]), os.listdir(val_dir))) + \
                             sum(map(lambda directory: len(directory[2]), os.listdir(test_dir)))
    else:
        copied_files_count = sum(map(lambda directory: len(directory[2]), os.listdir(train_dir)))

    assert copied_files_count == source_dir_files_count


def test_split():
    train_ratio = 0.75
    val_ratio = 0.15
    test_ratio = 0.1

    train_dir = '/home/mightyracoon/datasets/place_365/train_set'
    val_dir = '/home/mightyracoon/datasets/place_365/val_set'
    test_dir = '/home/mightyracoon/datasets/place_365/test_set'

    train_size = len(os.listdir(train_dir))
    val_size = len(os.listdir(val_dir))
    test_size = len(os.listdir(test_dir))

    total = train_size + val_size + test_size

    assert round(train_size / total, 2) == train_ratio and \
           round(val_size / total, 2) == val_ratio and \
           round(test_size / total, 2) == test_ratio


def test_stratification_split():
    train_ratio = 0.75
    val_ratio = 0.15
    test_ratio = 0.1

    train_dir = '/home/mightyracoon/datasets/place_365/train_set'
    val_dir = '/home/mightyracoon/datasets/place_365/val_set'
    test_dir = '/home/mightyracoon/datasets/place_365/test_set'

    classes = set()
    train_set = defaultdict(int)
    val_set = defaultdict(int)
    test_set = defaultdict(int)

    for file_name in os.listdir(train_dir):
        class_name = file_name.split('-')[0]
        classes.add(class_name)
        train_set[class_name] += 1

    for file_name in os.listdir(val_dir):
        class_name = file_name.split('-')[0]
        classes.add(class_name)
        val_set[class_name] += 1

    for file_name in os.listdir(test_dir):
        class_name = file_name.split('-')[0]
        classes.add(class_name)
        test_set[class_name] += 1

    for class_name in classes:
        total = train_set[class_name] + val_set[class_name] + test_set[class_name]

        split_error = np.sqrt(
            (round(train_set[class_name] / total, 2) - train_ratio) ** 2 + \
            (round(val_set[class_name] / total, 2) - val_ratio) ** 2 + \
            (round(test_set[class_name] / total, 2) - test_ratio) ** 2
        )

        assert split_error <= 0.1
