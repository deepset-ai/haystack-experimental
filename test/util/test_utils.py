import pytest

from haystack_experimental.util.utils import expand_page_range

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