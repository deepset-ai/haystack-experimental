def merge_dicts(d1, d2):
    """
    Merge two dictionaries. If a key exists in both dictionaries, the values are merged according to the following rules:

    - If both values are dictionaries, merge them recursively
    - If both values are lists, concatenate them
    - If the types are different, store both values in a list

    :param d1: Dictionary 1
    :param d2: Dictionary 2
    :return:
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
