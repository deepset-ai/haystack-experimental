
from haystack_experimental.util.utils import merge_dicts

def test_merge_dicts():

    json_a = {"name": "Alice", "age": 30, "hobbies": ["reading", "traveling"], "address": {"city": "Wonderland"}}
    json_b = {"age": 25, "hobbies": ["sports"], "address": {"city": "New York", "zip": "10001"}}

    expected = {
        'name': 'Alice',
        'age': [30, 25],
        'hobbies': ['reading', 'traveling', 'sports'],
        'address': {'city': ['Wonderland', 'New York'], 'zip': '10001'}
    }

    assert expected == merge_dicts(json_a, json_b)


    json_a = {"key": "a_string", "unique_in_a": "test"}
    json_b = {"key": 25, "unique_in_b": ["test"]}
    expected = {{'key': ['a_string', 25], 'unique_in_a': 'test', 'unique_in_b': ['test']}}

    assert expected == merge_dicts(json_a, json_b)