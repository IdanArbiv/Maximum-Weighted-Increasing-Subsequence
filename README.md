# Maximum-Weighted-Increasing-Subsequence

# Problem Statement:
We have a problem cell, which is defined as a collection (or collections) of points on a plane. Each point in the problem cell is represented as a tuple (x,y), where x and y are coordinates on the plane. The sequence in which the points appear in the cell is important and they should not be sorted.
## Definitions:
1. A point (a,b) is said to be greater than another point (x,y) if and only if both a > x and b > y.
2. A chain is a series of points where each subsequent point in the series is greater than the previous point.
3. If points are assigned weights, then the weight of a chain is the sum of the weights of the points in the chain. By default, each point in a problem cell has a weight of 1.
4. The maximum chain weight initially, denoted as W, is the weight of the heaviest chain that can be formed from the points in the problem cell.

## Task:
Our task is to design an algorithm that can selectively increase the weight of certain points from 1 to 2 while ensuring that the weight of the chain with the maximum weight remains the same as the original W. In other words, after the weights of certain points have been increased, it should still hold that all potential chains formed from the points in the problem cell have a weight less than or equal to W.

## Constraints:
1. It's crucial to note that the sequence of points given in the problem cell should not be sorted. The order of points matters and must be maintained.
2. Chains should strictly follow the sequence of points as given in the input. You can't rearrange the points to form a chain.
3. The number of points in a problem cell is between 1 and 100000.

## Input:
An array of tuples where each tuple represents a point with two coordinates (x,y). For example: [(1,1),(2,2),(3,3)]

## Output:
A list of points with their modified weights that satisfy the condition of the problem. For example, if no weights can be changed while maintaining the maximum chain weight, the output should be the same as the input.

# Research Question:
Given the constraints of a problem cell, how can we effectively develop an algorithm to identify and increase the weight of specific points from 1 to 2, while preserving the maximum chain weight (W) and the sequence of the points in the problem cell?

## Introduction:
The complexity of geometric problems has always held an unparalleled fascination in the field of computational geometry. Among the myriad of these problems, one specific problem has gained significant interest: managing the weights of points in a problem cell on a plane while maintaining a given maximum chain weight. This unique challenge resides in a special area of study where mathematics, computer science, and sequence analysis intersect.
In this context, a problem cell is a collection of ordered points on a plane, each defined by a tuple of two coordinates (x,y). Further adding to the complexity of this problem is the notion of 'chains'—an ordered series of points where each subsequent point is greater than the previous one according to a predefined criterion—and the concept of 'weight'. Each point in a problem cell carries an initial weight, and the weight of a chain is calculated as the sum of the weights of its constituent points.
Our task, and the focus of this research, is to design an effective algorithm capable of selectively increasing the weight of certain points within the cell from 1 to 2, whilst ensuring the maximum chain weight remains constant. This intricately nuanced problem draws on a variety of techniques and principles rooted in graph theory, sequence analysis, and optimization.
In the pursuit of a solution, this research is guided by specific constraints: the sequence of points within the problem cell must not be sorted, chains must follow the original sequence of points, and the number of points in a problem cell lies between 1 and 100000.
By examining this problem and developing a robust algorithm, we hope to contribute significantly to the field of computational geometry. We believe this study will not only address this specific problem but also inspire innovative thinking and methodologies that could be applied to similar problems within this fascinating domain of research.
 
## Algorithms:

1. First, we will present an algorithm that found 2 parameters:
### Input –
![image](https://github.com/IdanArbiv/Maximum-Weighted-Increasing-Subsequence/assets/101040591/61c078f0-0451-448e-a5bf-ca690365a7f1)

### Output - 
W - The maximum weighted chain in the input (The longest chain).
ranked_input - A dictionary contains all the points from the input as keys, where the value of each key is the maximum chain weight starting from that point (not just those on the maximum chain).

### Pseudocode:
![image](https://github.com/IdanArbiv/Maximum-Weighted-Increasing-Subsequence/assets/101040591/7af9ba85-475e-4eab-92e4-bc3c02f0cf1e)
		  	
## Explanation of the Algorithm:
This algorithm finds the maximum weight of a chain (an increasing subsequence) of points and the maximum weight that can be achieved starting from each point, in a given list of points with weights. The sequence of points in the list is important and should be maintained. 

The algorithm works by traversing the input list in reverse order, starting from the end and moving toward the start. For each point, it calculates the maximum sum of the weights of an increasing subsequence that ends at this point. The decision on whether to include a point in a chain is made based on its coordinates and the coordinates of the preceding points in the chain.

In detail, for each point (x,y,w) in the reversed list:

1. The algorithm first initializes the maximum chain weight for this point to be its own weight w. This is done because, at the very least, a chain can be formed with the point itself.

2. Then it looks at all points that come after the current point in the reversed list. For each such point (x1,y1,w1), it checks if (x,y) is less than (x1,y1) (in terms of both x and y coordinates), as per the problem's definition of one point being greater than another.

3. If (x,y) is indeed less than (x1,y1), the algorithm checks if the chain ending at (x1,y1) (whose weight is already calculated and stored in memo[j]) can be extended by adding the point (x,y) to get a chain with a larger weight.

4. If the new chain's weight (memo[j] + w) is larger than the current maximum weight for the point(x,y) (memo[i]), the algorithm updates the maximum weight (memo[i]) and also the entry for (x,y) in the ranked_input dictionary.

5. If (x,y) is not less than (x1,y1), the algorithm proceeds to the next point.
  Once the maximum chain weight for the point (x,y) is finalized (after considering all points that come after it in the reversed list), the algorithm moves to the next point in the reversed list.
  Finally, the algorithm returns the overall maximum chain weight (which is the maximum of all maximum chain weights for individual points) and the dictionary that stores the maximum chain weight for each point in the list.

## Time and Space Complexity Analysis:
The algorithm has a time complexity of O(n^2), where n is the number of points. This is because there are two nested loops: the outer loop runs n times and the inner loop can also run up to n times in the worst case. 

The space complexity of the algorithm is O(n), where n is the number of points. This is because it creates a 'memo' array and a 'results' dictionary, both of size n. The 'memo' array stores the maximum weighted increasing subsequence ending at each point, while the 'results' dictionary stores the maximum weighted increasing subsequence including each point.

## Example:

input_list = [(4,4,1),(2,3,1),(5,6,1),(1,1,1),(7,8,1),(8,9,1),(10,10,1),
(3,2,1),(6,7,1),(9,5,1),(0,0,1),(5,3,1),(2,6,1),(4,9,1),(6,3,1),(1,7,1)]

Our algorithm finds the next chain∶
W=5
ranked_input={ 
    (10,5,1): 1,(4,2,1): 2,(1,5,1): 1,(7,3,1): 2,(5,9,1): 1,(8,4,1): 2,(6,6,1): 1,
    (2,2,1): 3,(9,2,1): 2,(3,5,1): 2,(1,7,1): 2,(6,3,1): 3,(4,9,1): 1,(2,6,1): 2,
    (5,3,1): 3,(0,0,1): 4,(9,5,1): 1,(6,7,1): 1,(3,2,1): 4,(10,10,1): 1,(8,9,1): 2
    (7,8,1): 3,(1,1,1): 5,(5,6,1): 4,(2,3,1): 5,(4,4,1): 5
}

This is a visualization of one of the heaviest chains for this input:
![image](https://github.com/IdanArbiv/Maximum-Weighted-Increasing-Subsequence/assets/101040591/14229b0e-f641-4efd-9a94-53c34b1681f1)
![algorithm_visualization](https://github.com/IdanArbiv/Maximum-Weighted-Increasing-Subsequence/assets/101040591/f9d703d6-2a77-4b22-961c-1d7e0b1b637d)


# Naive approach to solving the main problem-

## Input –
list- ([x_1,y_1,w_1 ],[x_2,y_2,w_2 ],… ,[x_n 〖,y〗_n,w_n ]) s.t w_1=w_2=⋯=w_n=1
W - The maximum weighted chain in the input (The longest chain).

## Output - 
max_changed-  The number of points that for them weights have been increased.

## Pseudocode:
![image](https://github.com/IdanArbiv/Maximum-Weighted-Increasing-Subsequence/assets/101040591/1ca35913-500f-4770-b89b-05b2c261d181)

## Explanation of the Algorithm:
This algorithm applies a naive approach to the problem of maximizing the number of points with a weight of 2 without increasing the weight of the heaviest chain. The algorithm first generates all possible subsets of the input. For each subset, it increases the weight of each point in the subset to 2 and calculates the new total weight and the number of increased weights. It then checks if the heaviest chain in the updated list is lower than W  and if it does and if the number of increased weights is greater than the maximum number of increased weights found so far, it updates max_changed. This algorithm is called "naive" because it tries all possible combinations of points to increase their weight, which can be computationally expensive for large inputs. However, it guarantees finding the optimal solution because it explores the entire solution space.

## Time and Space Complexity Analysis:
The time complexity of this algorithm is O(2^n  * n^2), where n is the length of the input. The 2^n term comes from the generation of all subsets, and the n^2 term comes from the call to the weighted_increasing_subsequence algorithm for each subset.
The space complexity of this algorithm is O(2^n  + n^2), where n is the length of the input.  The 2^n term comes from the storage of all subsets, and the n^2 term comes from the space required by the weighted_increasing_subsequence algorithm. 
In conclusion, this algorithm has a high time and space complexity due to its exhaustive search of the solution space. However, for small inputs, it can be practical and guarantees finding the optimal solution.

# Heuristic approach to solving the main problem-

## Input –
1. list- ([x_1,y_1,w_1 ],[x_2,y_2,w_2 ],… ,[x_n 〖,y〗_n,w_n ]) s.t w_1=w_2=⋯=w_n=1
2. ranked_input - A dictionary where each point is a key and its value is the maximum weight of the chain starting at that point.
3. W - The maximum weighted chain in the input (The longest chain).

## Output - 
changed_items-  A list includes points whose weights have been increased.
modified_list- A list includes all points with their final weights.

## Pseudocode:
![image](https://github.com/IdanArbiv/Maximum-Weighted-Increasing-Subsequence/assets/101040591/8c607d00-b222-41c1-b3f9-d0307640cf02)


## Explanation of the Algorithm:
The algorithm provided above is a heuristic solution aimed at increasing the weight of certain points from 1 to 2 in a set of points while ensuring that the weight of the chain with the maximum weight remains constant.

Let's break down the steps of the algorithm:

1. Initial Setup: The algorithm starts with a given list of points and a dictionary (ranked_input) where each point is a key and its value is the maximum weight of the chain starting at that point. Additionally, it keeps a running total of additional weight assigned to points, called 'addition'.

2. Identification of the lowest partial order: In each round, the algorithm finds a subset of points such that there is no other point in the same subset with both x and y coordinates lower than these points. This subset is termed as the "lowest partial order". These points are essentially those which are not 'greater' than any other point in the same subset, based on the previously mentioned definition of 'greater'.

3. Incrementing weights: For each point in the lowest partial order, the algorithm checks whether adding 'addition' to the chain weight starting from that point would exceed the maximum chain weight, W. If it does not exceed, it increments the weight of the point by 1, adds this point to a separate 'changed_items' list and also increments 'addition' by 2. If none of the points in the lowest partial order meet this condition, 'addition' is simply incremented by 1.

4. Iterative processing: The algorithm repeats the above two steps iteratively until all points have been processed.

5. Output: At the end of this process, it returns two lists. The 'changed_items' list includes points whose weights have been increased and the 'modified_list' includes all points with their final weights.

![output](https://github.com/IdanArbiv/Maximum-Weighted-Increasing-Subsequence/assets/101040591/f037f301-cdb7-496b-ae2b-375ecaf52fd5)

## Time and Space Complexity Analysis:
The time complexity of the heuristic_algo algorithm is mainly determined by the nested loops in both algorithms and the list operations inside them, such as index and remove, which have a time complexity of O(n) in Python. As such, the overall time complexity is approximately O(n^2), where n is the length of the input_list. 
The space complexity is O(n) due to the storage required for the tmp_input_list, modified_list, and changed_items lists, and the space required for lowest_partial_order in the find_lowest_partial_order algorithm.

![image](https://github.com/IdanArbiv/Maximum-Weighted-Increasing-Subsequence/assets/101040591/2e2810ee-2e83-41dc-a2a9-b416b21ba0f3)

## Explain about steps 4 and 5 in the previous page:
In each layer, we go through the rank of each vertex and check if the ranked is less than W if we found any vertex that this condition is met for which we increase its weight by 1. We will do this for every vertex that is in the same layer. If we find such a vertex in a layer, we will add 2 to the ranked of the vertices in the next layer, otherwise, we will only add 1. We will check the condition again and again until we reach the last layer and basically go through all the vertices in the graph.

# Data plotting Examples:
![image](https://github.com/IdanArbiv/Maximum-Weighted-Increasing-Subsequence/assets/101040591/178584d9-e16b-4bd9-8b83-989cfa28b69c)

# Results:
![image](https://github.com/IdanArbiv/Maximum-Weighted-Increasing-Subsequence/assets/101040591/a687fddb-75d9-4c1d-b489-bc09c9a341e5)

# Heuristic Algorithm Results Report:
 In the results obtained from running the heuristic algorithm on various sample datasets (square and rhombus), several key observations can be made:

1. Maximum Weight of the Initial Chain: Each dataset has an initial maximum weight of the chain (W) determined from the original input. The values range from 17 to 44, representing the heaviest chain that can be formed from the initial points.
2. Execution Time: The execution time of the algorithm for each dataset is provided in seconds. The algorithm demonstrates efficient performance across the different sample sizes, with execution times ranging from 0.008 to 0.885 seconds. This showcases the algorithm's ability to process large datasets quickly.
3. Percentage of Items Changed: The algorithm succeeded in changing the weights of certain points from 1 to 2, selectively modifying a subset of the original points. The percentage of items changed varies across the datasets, ranging from approximately 13% to 17%. This demonstrates the algorithm's capability to identify and modify specific points while maintaining the maximum chain weight constraint.
4. Impact of Dataset Size: As the dataset size increases, the number of items successfully changed also increases. This behavior indicates the algorithm's effectiveness in selectively modifying point weights and achieving a desirable outcome, even with larger datasets.
Overall, the heuristic algorithm exhibits promising performance in achieving its goal of selectively increasing point weights while preserving the maximum chain weight constraint. Although it is a heuristic algorithm and may not provide an optimal solution, it showcases efficiency and effectiveness in modifying point weights based on the given constraints.

These results provide valuable insights into the algorithm's behavior and its potential application in scenarios where point weights need to be adjusted while maintaining the overall structure and properties of the chain. Further analysis and experimentation can be conducted to explore its performance on different types of datasets and evaluate its suitability for specific use cases.

## Comparison between Heuristic Algorithm and Naive Approach (optimal solution) on small data sets:
We conducted a comparison between the two algorithms using a smaller input of up to 21 points. The naive algorithm's performance becomes unreasonable for larger inputs, so this restriction was necessary.
We randomly generated 21 points in two-dimensional space when each entry is in range of 0 -30 and executed both algorithms on this set. We compared the running time and the number of points for which the weight was changed from 1 to 2. The results are summarized in two graphs, showcasing the algorithms' running time and the number of successfully modified points. These comparisons provide insights into the relative performance and effectiveness of the algorithms for the given task.

![image](https://github.com/IdanArbiv/Maximum-Weighted-Increasing-Subsequence/assets/101040591/4467a5ff-ee2d-47d8-bac8-aa93e3f097ab)

![image](https://github.com/IdanArbiv/Maximum-Weighted-Increasing-Subsequence/assets/101040591/8212ce6c-3c30-421f-a21a-18bd4c408f48)

We will now analyze the results from the graphs above:

## Test Results:
Both algorithms were tested on varying subset sizes (1-22). Both showed similar outcomes for sizes ≤6, while heuristic algorithm was faster. For larger subset sizes, the naive approach identified more changeable weights, but at the cost of exponentially longer execution times.

## Performance:
With a subset size of 22, naive approach took ~267.32 seconds, while heuristic approach was much faster at ~0.0001 seconds, maintaining its efficiency even for larger subsets.

## Conclusion: 
The heuristic algorithm is more efficient, especially for large subsets, but it may not always provide the optimal solution. This exemplifies a trade-off between optimality and efficiency. For exact optimal solutions, the slower naive approach is suitable, but for larger data sets and efficiency, the heuristic approach is preferable.






