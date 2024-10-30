import pytest

from haystack_experimental.util.utils import merge_dicts, expand_page_range


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
    expected = {'key': ['a_string', 25], 'unique_in_a': 'test', 'unique_in_b': ['test']}
    assert expected == merge_dicts(json_a, json_b)



def test_expand_page_range_valid_input():
    assert expand_page_range([1, 3]) == [1, 3]
    assert expand_page_range(['1-3']) == [1, 2, 3]
    assert expand_page_range(['1-3', 5, 8, '10-12']) == [1,2,3,5,8,10,11,12]

    assert expand_page_range(['1-3', '5', '8', '10-12']) == [1, 2, 3, 5, 8, 10, 11, 12]

    assert expand_page_range(['1-3', 5, 8, '10-12', '15-20', 50]) == [1,2,3,5,8,10,11,12,15,16,17,18,19,20,50]



def test_expand_page_range_invalid_input():

    with pytest.raises(ValueError):
        expand_page_range(['1-3', 'non_digit_string', 8, '10-12', '15-20', '50'])

    with pytest.raises(ValueError):
        expand_page_range([1-3, 5, 8])