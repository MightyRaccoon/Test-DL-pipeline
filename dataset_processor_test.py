import os

import pytest




def test_file():

    source_dir = '/home/mightyracoon/datasets/place_365/data_256'
    target_dir = '/home/mightyracoon/datasets/place_365/train'

    source_dir_files_count = sum(map(lambda directory: len(directory[2]), os.walk(source_dir)))
    target_dir_files_count = sum(map(lambda directory: len(directory[2]), os.walk(target_dir)))

    assert target_dir_files_count == source_dir_files_count
