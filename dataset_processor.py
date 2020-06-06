"""
Prepare dataset for DL pipeline
"""
import logging
import time
import os
import shutil
import asyncio
from typing import NoReturn
from collections import defaultdict

import click
import numpy as np


async def async_copyfile(source_name: str, destination_name: str) -> NoReturn:
    """
    Some wrapper function for shutil.copyfile with async statement
    :param source_name: Source file for copying process
    :param destination_name: Destination file for copying process
    """
    shutil.copyfile(source_name, destination_name)


async def async_movefile(source_name: str, destination_name: str) -> NoReturn:
    """
    Some wrapper function for shutil.move with async statement
    :param source_name: Source file for copying process
    :param destination_name: Destination file for copying process
    """
    shutil.move(source_name, destination_name)


def files_copy(source_directory: str, destination_directory: str) -> NoReturn:
    """
    Asynchronous files copy from source_directory with its subdirectories to flat directory
    with files renaming using subdirectory name
    :param source_directory: Source directory with files and subdirectories
    :param destination_directory: Destination directory
    :return:
    """
    loop = asyncio.get_event_loop()
    directory_structure = os.walk(source_directory)
    tasks_list = []
    for subdirectory in directory_structure:
        if len(subdirectory[2]) > 0:
            files_names = map(lambda file: subdirectory[0][len(source_directory):].replace('/', '_') + '-' + file, subdirectory[2])
            for source_file, destination_file in zip(subdirectory[2], files_names):
                source_path = subdirectory[0] + '/' + source_file
                destination_path = destination_directory + '/' + destination_file
                tasks_list.append(loop.create_task(async_copyfile(source_path, destination_path)))
    if len(tasks_list) > 0:
        loop.run_until_complete(asyncio.wait(tasks_list))


def train_val_test_split(train_dir: str, val_dir: str, test_dir: str,
                         train_ratio: float, val_ratio: float, test_ratio: float) -> NoReturn:
    """
    Split dataset on 3 parts: train set, validation set and test set.
    Validation and test sets move to its own directories asynchronous.
    :param train_dir:
    :param val_dir:
    :param test_dir:
    :param train_ratio:
    :param val_ratio:
    :param test_ratio:
    :return:
    """
    files = os.listdir(train_dir)
    tasks_list = []
    prob_array = np.random.uniform(size=len(files))
    loop = asyncio.get_event_loop()
    for file_name, prob in zip(files, prob_array):
        if train_ratio < prob <= train_ratio + val_ratio:
            source_file = train_dir + '/' + file_name
            dst_dir = val_dir + '/' + file_name
            tasks_list.append(loop.create_task(async_movefile(source_file, dst_dir)))
        elif train_ratio + val_ratio < prob <= train_ratio + val_ratio + test_ratio:
            source_file = train_dir + '/' + file_name
            dst_dir = test_dir + '/' + file_name
            tasks_list.append(loop.create_task(async_movefile(source_file, dst_dir)))
    if len(tasks_list) > 0:
        loop.run_until_complete(asyncio.wait(tasks_list))


def train_val_test_split_stratified(train_dir: str, val_dir: str, test_dir: str,
                                    train_ratio: float, val_ratio: float, test_ratio: float) -> NoReturn:
    """
    Split dataset on 3 parts: train set, validation set and test set with stratification.
    Validation and test sets move to its own directories asynchronous.
    :param train_dir:
    :param val_dir:
    :param test_dir:
    :param train_ratio:
    :param val_ratio:
    :param test_ratio:
    :return:
    """
    files = os.listdir(train_dir)
    tasks_list = []
    loop = asyncio.get_event_loop()
    class_objects = defaultdict(list)
    for file_name in files:
        class_objects[file_name.split('-')[0]].append(file_name)
    for files in class_objects.values():
        prob_array = np.random.uniform(size=len(files))
        for file_name, prob in zip(files, prob_array):
            if train_ratio < prob <= train_ratio + val_ratio:
                source_file = train_dir + '/' + file_name
                dst_dir = val_dir + '/' + file_name
                tasks_list.append(loop.create_task(async_movefile(source_file, dst_dir)))
            elif train_ratio + val_ratio < prob <= train_ratio + val_ratio + test_ratio:
                source_file = train_dir + '/' + file_name
                dst_dir = test_dir + '/' + file_name
                tasks_list.append(loop.create_task(async_movefile(source_file, dst_dir)))
    if len(tasks_list) > 0:
        loop.run_until_complete(asyncio.wait(tasks_list))


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level='INFO'
)

log = logging.getLogger(__name__)


@click.command()
@click.option('--source-dir', type=str, help='Directory with data')
@click.option('--dst-dir', type=str, help='Destination directory for data')
@click.option('--train-ratio', type=float, help='Train set ratio', default=0.75)
@click.option('--val-ratio', type=float, help='Validation set ratio', default=0.15)
@click.option('--test-ratio', type=float, help='Test set ratio', default=0.1)
@click.option('--use-stratification', type=bool, help='Bool flag stratification usage', default=False)
def main(source_dir, dst_dir, train_ratio, val_ratio, test_ratio, use_stratification):

    log.info('Start')

    if not os.path.exists(dst_dir):
        log.warning('Directory doesn\'t exists')
        os.mkdir(dst_dir)
        log.warning('Directory created')

    if len(os.listdir(dst_dir)) == 0:
        process_start = time.time()
        log.info('Asynchronous copying process start')
        files_copy(source_dir, dst_dir)
        process_end = time.time()
        pics_in_source = sum(map(lambda directory: len(directory[2]), os.walk(source_dir)))
        pics_in_dst = len(os.listdir(dst_dir))
        log.info(f'Copy process finished in {round(process_end - process_start, 2)}s')
        log.info(f'Copied {pics_in_dst}/{pics_in_source}: {100.0 * pics_in_dst / pics_in_source}%')
    else:
        pics_in_source = sum(map(lambda directory: len(directory[2]), os.walk(source_dir)))
        pics_in_dst = len(os.listdir(dst_dir))
        log.info(f'Copied {pics_in_dst}/{pics_in_source}: {100.0 * pics_in_dst / pics_in_source}%')

    if train_ratio + val_ratio + test_ratio != 1.0:
        log.error('Wrong ratio sum')
        log.warning('Default ratios will be used')
        train_ratio = 0.75
        val_ratio = 0.15
        test_ratio = 0.1

    log.info('Creating directories for validation and test sets')
    val_dir = '/'.join(dst_dir.split('/')[:-1]) + '/' + 'val_set'
    test_dir = '/'.join(dst_dir.split('/')[:-1]) + '/' + 'test_set'
    os.mkdir(val_dir)
    os.mkdir(test_dir)

    if use_stratification:
        log.info(f'Stratified Train/Validation/Test split with ratios {train_ratio}/{val_ratio}/{test_ratio} started')
        split_start = time.time()
        train_val_test_split_stratified(train_dir=dst_dir, val_dir=val_dir, test_dir=test_dir,
                                        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        split_end = time.time()
        log.info(f'Split end {round(split_end - split_start, 2)}s')
        log.info(f'Train set size: {len(os.listdir(dst_dir))}')
        log.info(f'Validation set size: {len(os.listdir(val_dir))}')
        log.info(f'Test set size: {len(os.listdir(test_dir))}')
    else:
        log.info(f'Train/Validation/Test split with ratios {train_ratio}/{val_ratio}/{test_ratio} started')
        split_start = time.time()
        train_val_test_split(train_dir=dst_dir, val_dir=val_dir, test_dir=test_dir,
                             train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        split_end = time.time()
        log.info(f'Split end {round(split_end - split_start, 2)}s')
        log.info(f'Train set size: {len(os.listdir(dst_dir))}')
        log.info(f'Validation set size: {len(os.listdir(val_dir))}')
        log.info(f'Test set size: {len(os.listdir(test_dir))}')


if __name__ == '__main__':
    asyncio.run(main())
