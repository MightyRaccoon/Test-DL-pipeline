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
        copied_files_count = sum(map(lambda directory: len(directory[2]), os.walk(train_dir))) + \
                             sum(map(lambda directory: len(directory[2]), os.walk(val_dir))) + \
                             sum(map(lambda directory: len(directory[2]), os.walk(test_dir)))
    else:
        copied_files_count = sum(map(lambda directory: len(directory[2]), os.walk(train_dir)))

    assert copied_files_count == source_dir_files_count


def test_split():
    train_ratio = 0.75
    val_ratio = 0.15
    test_ratio = 0.1

    train_dir = '/home/mightyracoon/datasets/place_365/train_set'
    val_dir = '/home/mightyracoon/datasets/place_365/val_set'
    test_dir = '/home/mightyracoon/datasets/place_365/test_set'

    train_size = sum(map(lambda directory: len(directory[2]), os.walk(train_dir)))
    val_size = sum(map(lambda directory: len(directory[2]), os.walk(val_dir)))
    test_size = sum(map(lambda directory: len(directory[2]), os.walk(test_dir)))

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

    for class_name in os.listdir(train_dir):
        classes.add(class_name)
        train_set[class_name] += len(os.listdir('/'.join((train_dir, class_name))))

    for class_name in os.listdir(val_dir):
        classes.add(class_name)
        val_set[class_name] += len(os.listdir('/'.join((val_dir, class_name))))

    for class_name in os.listdir(test_dir):
        classes.add(class_name)
        test_set[class_name] += len(os.listdir('/'.join((test_dir, class_name))))

    for class_name in classes:
        total = train_set[class_name] + val_set[class_name] + test_set[class_name]

        split_error = np.sqrt(
            (round(train_set[class_name] / total, 2) - train_ratio) ** 2 + \
            (round(val_set[class_name] / total, 2) - val_ratio) ** 2 + \
            (round(test_set[class_name] / total, 2) - test_ratio) ** 2
        ) / 3

        assert split_error <= 0.01
