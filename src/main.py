import sys

from loguru import logger

if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info(f'hello python{3.8}')
