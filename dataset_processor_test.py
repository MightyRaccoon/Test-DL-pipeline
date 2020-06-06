import os

import pytest


def test_file():

    source_dir = '/home/mightyracoon/datasets/place_365/data_256'

    train_dir = '/home/mightyracoon/datasets/place_365/train'
    val_dir = '/home/mightyracoon/datasets/place_365/val_set'
    test_dir = '/home/mightyracoon/datasets/place_365/test_set'

    source_dir_files_count = sum(map(lambda directory: len(directory[2]), os.walk(source_dir)))

    if os.path.exists(val_dir) and os.path.exists(test_dir):
        copied_files_count = sum(map(lambda directory: len(directory[2]), os.listdir(train_dir))) + \
                             sum(map(lambda directory: len(directory[2]), os.listdir(val_dir))) + \
                             sum(map(lambda directory: len(directory[2]), os.listdir(test_dir)))

    assert copied_files_count == source_dir_files_count


def test_split():

    train_ratio = 0.75
    val_ratio = 0.15
    test_ratio = 0.1

    train_dir = '/home/mightyracoon/datasets/place_365/train'
    val_dir = '/home/mightyracoon/datasets/place_365/val_set'
    test_dir = '/home/mightyracoon/datasets/place_365/test_set'

    train_size = len(os.listdir(train_dir))
    val_size = len(os.listdir(val_dir))
    test_size = len(os.listdir(test_dir))

    total = train_size + val_size + test_size

    assert round(train_size/total, 2) == train_ratio and \
        round(val_size/total, 2) == val_ratio and \
        round(test_size/total, 2) == test_ratio