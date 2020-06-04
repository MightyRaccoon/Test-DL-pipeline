import os

import pytest


def search_in_dir(directory: str, directory_counter: int = 0) -> int:
    """
    Count files in directory and its subdirectories
    :param directory:
    :param directory_counter:
    :return:
    """
    content_list = os.listdir(directory)
    count = 0
    for content in content_list:
        source_name = directory + '/' + content
        if os.path.isfile(source_name):
            count += 1
        else:
            count += search_in_dir(source_name, directory_counter)
    return directory_counter + count


def test_file():

    source_dir = '/home/mightyracoon/datasets/place_365/train_async'
    target_dir = '/home/mightyracoon/datasets/place_365/train'

    source_dir_files_count = search_in_dir(source_dir)
    target_dir_files_count = search_in_dir(target_dir)

    assert target_dir_files_count == source_dir_files_count
