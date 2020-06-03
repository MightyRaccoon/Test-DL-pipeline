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


def search_for_files(source_directory: str, destination_directory: str) -> NoReturn:
    """
    Using some imitation of DFS search for files in source_directory and its subdirectories and
    move ot to destination_directory

    :param source_directory: Directory with files and subdirectories where data is stored
    :param destination_directory: New directory for data without subdirectories. It will contain only data
    """
    content_list = os.listdir(source_directory)
    loop = asyncio.get_event_loop()
    tasks_list = []
    for content in content_list:
        source_name = source_directory + '/' + content
        if os.path.isfile(source_name):
            destination_name = destination_directory + '/' + source_directory.split('/').pop() + '_' + content
            tasks_list.append(loop.create_task(async_copyfile(source_name, destination_name)))
        else:
            search_for_files(source_name, destination_directory)
    if len(tasks_list) > 0:
        loop.run_until_complete(asyncio.wait(tasks_list))


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level='INFO'
)

log = logging.getLogger(__name__)

@click.command()
@click.option('--data-dir', type=str, help='Directory with data')
@click.option('--dst-dir', type=str, help='Destination directory for data')
def main(data_dir, dst_dir):

    log.info('Start')

    if not os.path.exists(dst_dir):
        log.warning('Directory doesn\'t exists')
        os.mkdir(dst_dir)
        log.warning('Directory created')

    log.info('Copy process start')
    process_start = time.time()
    search_for_files(data_dir, dst_dir)

    process_end = time.time()
    log.info(f'Copy process finished in {round(process_end - process_start, 2)}s')


if __name__ == '__main__':
    asyncio.run(main())