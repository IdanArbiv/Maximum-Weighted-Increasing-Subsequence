import numpy as np
from itertools import chain, combinations
import time


def weighted_increasing_subsequence(input, index=0, largestA=None, memo={}, results = {}):
    if index >= len(input):
        return 0, results
    if (index, largestA) in memo:
        return memo[(index, largestA)], results
    x, y, w = input[index]
    if largestA is not None and (x <= largestA[0] or y <= largestA[1]):
        result, _ = weighted_increasing_subsequence(input, index + 1, largestA, memo)
    else:
        without_me, _ = weighted_increasing_subsequence(input, index + 1, largestA, memo)
        with_me, _ = weighted_increasing_subsequence(input, index + 1, (x, y), memo)
        with_me = with_me + w
        results[input[index]] = with_me
        result = max(without_me, with_me)
    memo[(index, largestA)] = result
    return result, results


def find_lowest_partial_order(input_list):
    lowest_partial_order = []
    for (x, y, w) in input_list:
        flag = True
        for (x_low,y_low, _) in lowest_partial_order:
            if x_low < x and y_low < y:
                flag = False
                break
        if flag:
            lowest_partial_order.append((x,y,w))
            input_list.remove((x, y, w))
    return lowest_partial_order, input_list


def arbiv_heuristic(input_list, ranked_input):
    tmp_input_list = list(input_list)
    modified_list = list(input_list)
    changed_items = []
    addition = 0
    while len(tmp_input_list) != 0:
        lowest_partial_order, tmp_list = find_lowest_partial_order(tmp_input_list)
        tmp_input_list = tmp_list
        flag = False
        for elem in lowest_partial_order:
            if ranked_input[elem] + addition < maximum_wis:
                (x, y, w) = elem
                changed_items.append((x, y, w + 1))
                modified_list[modified_list.index(elem)] = (x, y, w + 1)
                flag = True
        addition = addition + 2 if flag else addition + 1
    return changed_items, modified_list


def naive_approach(input_list, original_wis):
    input_array = np.array(input_list)
    subsets = list(chain(*map(lambda x: combinations(range(len(input_list)), x), range(0, len(input_list)+1))))
    max_changed = 0
    i = 0
    for subset in subsets:
        subset_array = input_array.copy()
        for index in subset:
            subset_array[index, 2] = 2
        sum_of_third_elements = np.sum(subset_array[:, 2])
        tmp_changed = sum_of_third_elements - len(input_list)
        subset_array = list(map(tuple, subset_array))
        tmp_wis , _ = weighted_increasing_subsequence(subset_array, index=0, largestA=None, memo={}, results = {})
        if tmp_changed == 5 and tmp_wis == 5:
            i = i +1
        if tmp_wis == original_wis:
            if max_changed < tmp_changed:
                max_changed = tmp_changed
    return max_changed



if __name__ == '__main__':
    # input_list = [(4, 4, 1), (2, 3, 1), (5, 6, 1), (1, 1, 1), (7, 8, 1), (8, 9, 1), (10, 10, 1), (3, 2, 1), (6, 7, 1)]
    # input_list = [(4, 4, 1), (3, 3, 1), (2, 2, 1), (1, 1, 1)]
    input_list = [(1, 2, 1), (4, 5, 1), (3, 6, 1), (8, 9, 1), (10, 12, 1), (11, 14, 1), (7, 13, 1), (16, 17, 1), (19, 20, 1), (18, 21, 1), (23, 24, 1), (26, 27, 1), (25, 28, 1), (31, 33, 1)]
    print("Original List: {}".format(input_list))
    maximum_wis, ranked_input = weighted_increasing_subsequence(input_list)
    print("Maximum WIS in original input: {}".format(maximum_wis))
    arbiv_start_time = time.time()
    changed_items, modified_list = arbiv_heuristic(input_list, ranked_input)
    arbiv_end_time = time.time()
    print("The Algorithm changed the following items: {}".format(changed_items))
    print("Modified List: {}".format(modified_list))
    new_maximum_wis, ranked_input = weighted_increasing_subsequence(input_list, index=0, largestA=None, memo={}, results = {})
    if new_maximum_wis == maximum_wis:
        print("The Algorithm Passed Within {:.3f} Seconds! \n"
              "The Algorithm succeeded in changing {} out of {} items".format(arbiv_end_time - arbiv_start_time, len(changed_items), len(input_list)))
    else:
        print("The Algorithm Failed!\n"
              "New Maximum WIS: {} , Original Maximum WIS: {}".format(new_maximum_wis, maximum_wis))

    naive_start_time = time.time()
    print("Naive Approach Changed {} Items".format(naive_approach(input_list, maximum_wis)))
    native_end_time = time.time()
    print("Naive Approach O(N^2 * 2^N) Took {} Second!".format(native_end_time - naive_start_time))