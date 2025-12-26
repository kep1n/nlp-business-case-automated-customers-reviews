import os
import re
import datetime
from typing import Union
import logging

PROJECT_LOGGER = 'ProjectLogger'
logger = logging.getLogger(PROJECT_LOGGER)


def parse_number(value: str) -> Union[int, float, None]:
    """
    Extracts a numeric value from a string.
    - Returns int if no decimal part exists
    - Returns float if a decimal part exists
    """
    match = re.search(r'-?\d+(?:[.,]\d+)?', value)
    if not match:
        logger.debug(f'No numeric value found, value -> {value}')
        return None

    number_str = match.group()

    # Normalize decimal separator
    if ',' in number_str or '.' in number_str:
        number_str = number_str.replace(',', '.')
        number = float(number_str)
        return int(number) if number.is_integer() else number

    if number_str == '-1':
        return None

    return int(number_str)


def parse_dates(value: str) -> Union[str, None]:
    if value in ('0', ' ', ''):
        return None
    try:
        timestamp = int(value)
        return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
    except OSError:
        logger.warning('{} is not real date. Probably the game is coming soon or date isn\'t set yet')
        return None
    except Exception as e:
        logger.exception(f'{e}; Value: {value}')
