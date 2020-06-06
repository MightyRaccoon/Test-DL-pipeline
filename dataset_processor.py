import logging
import time
import os
from typing import NoReturn
import shutil
import asyncio

import click


async def async_copyfile(source_name: str, destination_name: str) -> NoReturn:
    """
    Some wrapper function for shutil.copyfile with async statement
    :param source_name: Source file for copying process
    :param destination_name: Destination file for copying process
    """
    shutil.copyfile(source_name, destination_name)


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


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level='INFO'
)

log = logging.getLogger(__name__)


@click.command()
@click.option('--source-dir', type=str, help='Directory with data')
@click.option('--dst-dir', type=str, help='Destination directory for data')
def main(source_dir, dst_dir):

    log.info('Start')

    if not os.path.exists(dst_dir):
        log.warning('Directory doesn\'t exists')
        os.mkdir(dst_dir)
        log.warning('Directory created')
    if len(os.listdir(dst_dir)) == 0:
        log.info('Asynchronous copying process start')
        process_start = time.time()
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


if __name__ == '__main__':
    asyncio.run(main())