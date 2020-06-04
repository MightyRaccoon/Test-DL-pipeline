import logging
import time
import os
from typing import NoReturn
import shutil
import asyncio
import multiprocessing

import click


async def async_copyfile(source_name: str, destination_name: str) -> NoReturn:
    """
    Some wrapper function for shutil.copyfile with async statement
    :param source_name: Source file for copying process
    :param destination_name: Destination file for copying process
    """
    shutil.copyfile(source_name, destination_name)

#
# def search_for_files(source_directory: str, destination_directory: str, loop) -> NoReturn:
#     """
#     Using some imitation of DFS search for files in source_directory and its subdirectories and
#     move ot to destination_directory
#
#     :param source_directory: Directory with files and subdirectories where data is stored
#     :param destination_directory: New directory for data without subdirectories. It will contain only data
#     :param loop: Event loop from asyncio. Used for asyncronous files copying
#     """
#     content_list = os.listdir(source_directory)
#     tasks_list = []
#     for content in content_list:
#         source_name = source_directory + '/' + content
#         if os.path.isfile(source_name):
#             destination_name = destination_directory + '/' + source_directory.split('/').pop() + '_' + content
#             tasks_list.append(loop.create_task(async_copyfile(source_name, destination_name)))
#         else:
#             search_for_files(source_name, destination_directory, loop)
#     if len(tasks_list) > 0:
#         loop.run_until_complete(asyncio.wait(tasks_list))


def search_for_files(source_directory: str, destination_directory: str, pool) -> NoReturn:
    content_list = os.listdir(source_directory)
    tasks_list = []
    tasks_list_2 = []
    for content in content_list:
        source_name = source_directory + '/' + content
        if os.path.isfile(source_name):
            destination_name = destination_directory + '/' + source_directory.split('/').pop() + '_' + content
            tasks_list.append((source_name, destination_name))
        else:
            search_for_files(source_name, destination_directory, pool)
    if len(tasks_list) > 0:
        pool.starmap(shutil.copyfile, tasks_list, chunksize=len(pool._pool)//len(tasks_list) + 1)
# def search_for_files(source_directory: str, destination_directory: str) -> NoReturn:
#     """
#     Using some imitation of DFS search for files in source_directory and its subdirectories and
#     move ot to destination_directory
#
#     :param source_directory: Directory with files and subdirectories where data is stored
#     :param destination_directory: New directory for data without subdirectories. It will contain only data
#     """
#     content_list = os.listdir(source_directory)
#     tasks_list = []
#     for content in content_list:
#         source_name = source_directory + '/' + content
#         if os.path.isfile(source_name):
#             destination_name = destination_directory + '/' + source_directory.split('/').pop() + '_' + content
#             shutil.copyfile(source_name, destination_name)
#         else:
#             search_for_files(source_name, destination_directory)


def mass_copy(source_directory: str, destination_directory: str):

    directory_structure = os.walk(source_directory)
    for subdirectory in directory_structure:
        if len(subdirectory[2]) > 0:
            files_names = map(lambda file: subdirectory[0][len(source_directory):].replace('/', '_') + '-' + file, subdirectory[2])
            for source_file, destination_file in zip(subdirectory[2], files_names):
                source_path = subdirectory[0] + '/' + source_file
                destination_path = destination_directory + '/' + destination_file
                shutil.copyfile(source_path, destination_path)



logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level='INFO'
)

log = logging.getLogger(__name__)

@click.command()
@click.option('--data-dir', type=str, help='Directory with data')
@click.option('--dst-dir', type=str, help='Destination directory for data')
@click.option('--retries-timeout', type=int, help='Timeouts duration between event loop state checking', default=1)
def main(data_dir, dst_dir, retries_timeout):

    log.info('Start')

    if not os.path.exists(dst_dir):
        log.warning('Directory doesn\'t exists')
        os.mkdir(dst_dir)
        log.warning('Directory created')

    log.info('Copy process start')
    process_start = time.time()

    #loop = asyncio.get_event_loop()
    # pool = multiprocessing.Pool(processes=14)
    # search_for_files(data_dir, dst_dir, pool)
    # while loop.is_running():
    #     asyncio.wait_for(retries_timeout)
    #     log.warning('Event loop is still running')
    #     log.warning(f'Waiting for {retries_timeout}s until on more check')
    # loop.close()
    # if pool._check_running():
    #     pool.join()
    # pool.close()
    mass_copy(data_dir, dst_dir)
    process_end = time.time()
    log.info(f'Copy process finished in {round(process_end - process_start, 2)}s')


if __name__ == '__main__':
    asyncio.run(main())