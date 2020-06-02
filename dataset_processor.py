import logging
import time
import os
from typing import NoReturn
import shutil

import click


def search_for_files(source_directory: str, destination_directory: str) -> NoReturn:
    """
    Using some imitation of DFS search for files in source_directory and its subdirectories and
    move ot to destination_directory

    :param source_directory: Directory with files and subdirectories where data is stored
    :param destination_directory: New directory for data without subdirectories. It will contain only data
    """
    content_list = os.listdir(source_directory)
    for content in content_list:
        source_name = source_directory + '/' + content
        if os.path.isfile(source_name):
            destination_name = destination_directory + '/' + destination_directory.split('/').pop() + '_' + content
            shutil.copyfile(source_name, destination_name)
        else:
            search_for_files(source_name, destination_directory)


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level='INFO'
)

log = logging.getLogger(__name__)

@click.command()
@click.option('--data-dir', help='Directory with data')
@click.option('--dst-dir', help='Destination directory for data')
def main(data_dir, dst_dir):

    log.info('Start')

    if not os.path.exists(dst_dir):
        log.warning('Directory doesn\'t exists')
        os.mkdir(dst_dir)
        log.warning('Directory created')

    process_start = time.time()
    print(search_for_files(data_dir, dst_dir))

    process_end = time.time()
    log.info(f'Finish in {round(process_end - process_start, 2)}s')


if __name__ == '__main__':
    main()