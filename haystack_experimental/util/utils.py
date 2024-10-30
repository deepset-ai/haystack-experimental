# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Union, List


def merge_dicts(d1, d2):
    """
    Merge two dictionaries. If a key exists in both dictionaries, the values are merged according to the rules:

    - If both values are dictionaries, merge them recursively
    - If both values are lists, concatenate them
    - If the types are different, store both values in a list

    :param d1: Dictionary 1
    :param d2: Dictionary 2
    :returns:
        a single dictionary with the merged values from d1 and d2
    """

    merged = {}

    def is_primitive(value):
        return isinstance(value, (int, float, bool, str, type(None)))

    for key in d1.keys():
        if key in d2:
            # both are dictionaries
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                merged[key] = merge_dicts(d1[key], d2[key])
            # both are lists
            elif isinstance(d1[key], list) and isinstance(d2[key], list):
                merged[key] = d1[key] + d2[key]
            # both are of primitive types
            elif is_primitive(d1[key]) and is_primitive(d2[key]):
                merged[key] = [d1[key], d2[key]]
        else: # key only in d1
            merged[key] = d1[key]

    # add remaining keys from d2
    for key in d2.keys():
        if key not in merged:
            merged[key] = d2[key]

    return merged


def expand_page_range(page_range: Union[ List[Union[int,str]], List[int] ] ):
    """
    Takes a list of page numbers and ranges and expands them into a list of page numbers.

    For example, given a page_range=['1-3', '5', '8', '10-12'] the function will return [1, 2, 3, 5, 8, 10, 11, 12]

    :param page_range: List of page numbers and ranges
    :returns:
        An expanded list of page integers

    """
    expanded_page_range = []

    for page in page_range:
        if isinstance(page, int):
            # check if it's a range wrongly passed as an integer expression
            if "-" in str(page):
                msg = "range must be a string in the format 'start-end'"
                raise ValueError(f"Invalid page range: {page} - {msg}")
            expanded_page_range.append(page)

        elif isinstance(page, str) and page.isdigit():
            expanded_page_range.append(int(page))

        elif isinstance(page, str) and "-" in page:
            start, end = page.split("-")
            expanded_page_range.extend(range(int(start), int(end) + 1))

        else:
            msg = "range must be a string in the format 'start-end' or an integer"
            raise ValueError(f"Invalid page range: {page} - {msg}")

    return expanded_page_range