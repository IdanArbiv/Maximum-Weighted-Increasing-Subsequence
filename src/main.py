import numpy as np
from itertools import chain, combinations
import time
import pandas as pd
import sys
import matplotlib.pyplot as plt
sys.setrecursionlimit(100000)
import imageio
import matplotlib.ticker as ticker
import random

# ------------------------------- Main Functions ---------------------------------
def weighted_increasing_subsequence_iter(input):
    n = len(input)
    memo = [0] * n
    results = {}

    for i in range(n - 1, -1, -1):
        x, y, w = input[i]
        memo[i] = w
        results[input[i]] = w
        for j in range(i + 1, n):
            x1, y1, w1 = input[j]
            if x < x1 and y < y1:
                if memo[i] < memo[j] + w:
                    memo[i] = memo[j] + w
                    results[input[i]] = memo[i]

    return max(memo), results


def weighted_increasing_subsequence_recursion(input, index=0, largestA=None, memo={}, results = {}):
    if index >= len(input):
        return 0, results
    if (index, largestA) in memo:
        return memo[(index, largestA)], results
    x, y, w = input[index]
    if largestA is not None and (x <= largestA[0] or y <= largestA[1]):
        result, _ = weighted_increasing_subsequence_recursion(input, index + 1, largestA, memo)
    else:
        without_me, _ = weighted_increasing_subsequence_recursion(input, index + 1, largestA, memo)
        with_me, _ = weighted_increasing_subsequence_recursion(input, index + 1, (x, y), memo)
        with_me = with_me + w
        results[input[index]] = with_me
        result = max(without_me, with_me)
    memo[(index, largestA)] = result
    return result, results


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
        memo = {}
        results = {}
        tmp_wis , _ = weighted_increasing_subsequence_iter(subset_array)
        memo.clear()
        results.clear()
        if tmp_changed == 5 and tmp_wis == 5:
            i = i +1
        if tmp_wis == original_wis:
            if max_changed < tmp_changed:
                max_changed = tmp_changed
    return max_changed


def heuristic_algo(input_list, ranked_input, maximum_wis):
    tmp_input_list = list(input_list)
    modified_list = list(input_list)
    changed_items = []
    addition = 0
    i = 0

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
                # plot_points(input_list, changed_items, i)  # Here we call the plot function.
                i += 1

        addition = addition + 2 if flag else addition + 1
    return changed_items, modified_list


# ------------------------------- Helper Functions ---------------------------------

def execute(input_list):
    # print("Original List: {}".format(input_list))
    maximum_wis, ranked_input = weighted_increasing_subsequence_iter(input_list)
    print("Maximum WIS in original input: {}".format(maximum_wis))
    # print("Ranked Input: " + str(ranked_input))
    arbiv_start_time = time.time()
    changed_items, modified_list = heuristic_algo(input_list, ranked_input, maximum_wis)
    arbiv_end_time = time.time()

    # Finally, compile your saved plots into a gif using imageio.
    # images = []
    # filenames = [f'plot_{i}.png' for i in range(len(changed_items))]
    #
    # for filename in filenames:
    #     images.append(imageio.imread(filename))
    # imageio.mimsave('output.gif', images, duration=0.1, loop=0)

    print("The Algorithm changed the following items: {}".format(changed_items))
    # print("Modified List: {}".format(modified_list))
    new_maximum_wis, ranked_input = weighted_increasing_subsequence_iter(input_list)
    if new_maximum_wis == maximum_wis:
        print("The Algorithm Passed Within {:.3f} Seconds! \n"
              "The Algorithm succeeded in changing {} out of {} items".format(arbiv_end_time - arbiv_start_time,
                                                                              len(changed_items), len(input_list)))
    else:
        print("The Algorithm Failed!\n"
              "New Maximum WIS: {} , Original Maximum WIS: {}".format(new_maximum_wis, maximum_wis))
    # naive_start_time = time.time()
    # print("Naive Approach Changed {} Items".format(naive_approach(input_list, maximum_wis)))
    # native_end_time = time.time()
    # print("Naive Approach O(N^2 * 2^N) Took {} Second!".format(native_end_time - naive_start_time))


def find_lowest_partial_order(input_list):
    lowest_partial_order = []
    for (x, y, w) in input_list:
        flag = True
        for (x_low, y_low, _) in lowest_partial_order:
            if x_low < x and y_low < y:
                flag = False
                break
        if flag:
            lowest_partial_order.append((x, y, w))
            input_list.remove((x, y, w))
    return lowest_partial_order, input_list


def read_xlsx_file(sheet_name, num_of_points):
    df = pd.read_excel('data.xlsx', sheet_name=sheet_name)
    points =  [(df['Column 1'][x], df['Column 2'][x], 1) for x in range(num_of_points)]

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

    # plot_original_points(sheet_name, x_values, y_values)

    return points


# ------------------------------- Functions for plotting and gifs -------------------------------
def plot_original_points(sheet_name, x_values, y_values):
    plt.scatter(x_values, y_values)
    plt.xlabel('Column 1')
    plt.ylabel('Column 2')
    plt.title('Original: ' + sheet_name)
    plt.show()

def plot_points(points, changed_items, i):
    # Clear the previous plot.
    plt.clf()

    # Plot all points in blue.
    for point in points:
        plt.plot(point[0], point[1], 'bo')

    # Overplot the changed points in red.
    for changed_point in changed_items:
        plt.plot(changed_point[0], changed_point[1], 'ro')

    # Save each plot with a different name (sequentially).
    plt.savefig(f'plot_{i}.png')


def remove_third_entry(input_list):
    return [(x, y) for x, y, _ in input_list]


def plot_chain(input_points, chain):
    input_points = remove_third_entry(input_points)

    # Split input points and chain into X and Y coordinates for plotting
    input_x, input_y = zip(*input_points)
    chain_x, chain_y, _ = zip(*chain)

    # Calculate weight of the chain
    weight = sum(w for _, _, w in chain)

    # Create the plot
    plt.scatter(input_x, input_y, color='blue')
    plt.plot(chain_x, chain_y, color='red', linewidth=2)
    plt.title('Points and Heaviest Chain')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Annotate weight of the heaviest chain
    mid_point_index = len(chain) // 2
    plt.annotate(f'Weight: {weight}', xy=(chain_x[mid_point_index], chain_y[mid_point_index]), xytext=(0, 10),
                 textcoords='offset points', ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='black'))

    plt.show()


def weighted_increasing_subsequence_plot_version(input_list):
    n = len(input_list)
    memo = [0] * n
    results = {}
    chains = {}

    for i in range(n - 1, -1, -1):
        x, y, w = input_list[i]
        memo[i] = w
        results[input_list[i]] = w
        chains[input_list[i]] = [input_list[i]]
        for j in range(i + 1, n):
            x1, y1, w1 = input_list[j]
            if x < x1 and y < y1:
                if memo[i] < memo[j] + w:
                    memo[i] = memo[j] + w
                    results[input_list[i]] = memo[i]
                    chains[input_list[i]] = [input_list[i]] + chains[input_list[j]]

    # Find the starting point of the maximum weighted chain
    max_weight_point = max(results, key=results.get)

    return max(memo), results, chains[max_weight_point]


def visualize_weighted_increasing_algorithm(input_list, maximum_wis, ranked_input, max_chain):
    images = []
    chain_images = []
    maximum_wis_list = [i for i in range(maximum_wis, 1, -1)]

    fig, ax = plt.subplots()
    colors = np.arange(len(input_list))

    ind = 0

    # Scatter plot for the points
    # Traverse the ranked_input dictionary
    for i, ((x, y, w), max_chain_weight) in enumerate(ranked_input.items()):

        # Scatter plot for the points
        scatter = ax.scatter([p[0] for p in input_list], [p[1] for p in input_list],
                             c=colors, cmap='viridis', s=[p[2] * 100 for p in input_list], alpha=0.6)

        # Highlight the current point
        ax.scatter([x], [y], color='red', s=w * 150)

        # Annotate the points with their weights and max_chain_weights
        for (x_, y_, w_), max_chain_weight_ in ranked_input.items():
            ax.text(x_, y_, f'w:{w_}, mcw:{max_chain_weight_}', fontsize=8)

        # Set titles, labels and legend
        ax.set_title(f'Step {i + 1}: Point ({x},{y}) with weight {w} and max chain weight {max_chain_weight}\n')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        legend1 = ax.legend(*scatter.legend_elements(num=len(input_list)),
                            loc="upper left", title="Points")
        ax.add_artist(legend1)

        # Save the figure as an image file
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

        # If the point is part of the maximum chain, plot the chain up to this point
        if (x, y, w) in max_chain:
            chain_index = max_chain.index((x, y, w))
            if chain_index > 0:
                ax.set_title(f'Current Maximum Chain Weight: {maximum_wis_list[ind]}')
                ind += 1
                chain_x = [p[0] for p in max_chain[:chain_index + 1]]
                chain_y = [p[1] for p in max_chain[:chain_index + 1]]
                ax.plot(chain_x, chain_y, 'b-')
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                chain_images.append(image)

        # Remove texts and points for next iteration
        ax.clear()

    imageio.mimsave('algorithm_visualization.gif', images + chain_images[::-1], duration=1500, loop=0)
    print("GIF has been saved as 'algorithm_visualization.gif'")


def compare_naive_and_heuristic(input_list):
    n = len(input_list)
    for i in range(1, n + 1):
        subsets = [input_list[:i]]
        for subset in subsets:
            subset_list = list(subset)

            print("Subset Size:", len(subset_list))

            # Run the naive approach
            start_time = time.time()
            original_wis, _ = weighted_increasing_subsequence_iter(subset_list)
            naive_changed = naive_approach(subset_list, original_wis)
            end_time = time.time()
            naive_time = end_time - start_time

            # Run the heuristic algorithm
            start_time = time.time()
            original_wis, ranked_input = weighted_increasing_subsequence_iter(subset_list)
            heuristic_changed, _ = heuristic_algo(subset_list, ranked_input, original_wis)
            end_time = time.time()
            heuristic_time = end_time - start_time

            print("Naive Approach - Number of Items Changed:", naive_changed)
            print("Naive Approach - Execution Time:", naive_time, "seconds")
            print("Heuristic Algorithm - Number of Items Changed:", len(heuristic_changed))
            print("Heuristic Algorithm - Execution Time:", heuristic_time, "seconds")
            print("")


def compare_naive_and_heuristic_graph(input_list):
    n = len(input_list)
    subset_sizes = []
    naive_times = []
    heuristic_times = []
    naive_changed_items = []
    heuristic_changed_items = []

    for i in range(1, n + 1):
        subsets = [input_list[:i]]
        for subset in subsets:
            subset_list = list(subset)
            subset_size = len(subset_list)

            subset_sizes.append(subset_size)

            # Run the naive approach
            start_time = time.time()
            original_wis, _ = weighted_increasing_subsequence_iter(subset_list)
            naive_changed = naive_approach(subset_list, original_wis)
            end_time = time.time()
            naive_time = end_time - start_time

            naive_times.append(naive_time)
            naive_changed_items.append(naive_changed)

            # Run the heuristic algorithm
            start_time = time.time()
            original_wis, ranked_input = weighted_increasing_subsequence_iter(subset_list)
            heuristic_changed, _ = heuristic_algo(subset_list, ranked_input, original_wis)
            end_time = time.time()
            heuristic_time = end_time - start_time

            heuristic_times.append(heuristic_time)
            heuristic_changed_items.append(len(heuristic_changed))

    # Plotting the time comparison graph
    plt.figure()
    plt.plot(subset_sizes, naive_times, label='Naive Approach')
    plt.plot(subset_sizes, heuristic_times, label='Heuristic Algorithm')
    plt.xlabel('Subset Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Time Comparison: Naive Approach vs Heuristic Algorithm')
    plt.legend()
    plt.show()

    # Plotting the item change comparison graph
    plt.figure()
    prev_equal_result = False
    prev_naive_heuristic = False
    for i in range(len(subset_sizes)):
        if naive_changed_items[i] == heuristic_changed_items[i]:
            if not prev_equal_result:
                offset = random.uniform(-0.1, 0.1)  # Random offset for perturbation
                plt.plot(subset_sizes[i] + offset, naive_changed_items[i], 'ro', label='Equal Result')
                prev_equal_result = True
            else:
                plt.plot(subset_sizes[i] + offset, naive_changed_items[i], 'ro')
        else:
            if not prev_naive_heuristic:
                plt.plot(subset_sizes[i], naive_changed_items[i], 'o', label='Naive Approach', color='green')
                plt.plot(subset_sizes[i], heuristic_changed_items[i], 'o', label='Heuristic Algorithm', color='blue')
                prev_naive_heuristic = True
            else:
                plt.plot(subset_sizes[i] + offset, naive_changed_items[i], 'ro', color='green')
                plt.plot(subset_sizes[i] + offset, heuristic_changed_items[i], 'ro', color='blue')

    # Set the y-axis to display only natural numbers
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.xlabel('Subset Size')
    plt.ylabel('Number of Items Changed')
    plt.title('Item Change Comparison: Naive Approach vs Heuristic Algorithm')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # input_list = [(4, 4, 1), (2, 3, 1), (5, 6, 1), (1, 1, 1), (7, 8, 1), (8, 9, 1), (10, 10, 1), (3, 2, 1), (6, 7, 1)]
    # execute(input_list)

    for i in range(1, 11):
        for shape in ['square', 'rhombus']:
            print('\n\nResults Of Arbiv heuristic: ' + shape  + '_' + str(i) +'000_samples')
            input_list_main = read_xlsx_file(shape  + '_' + str(i) +'000_samples', i * 1000)
            execute(input_list_main)


