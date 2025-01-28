def hypothesis(example, parameters):
    result = 0
    for i in range(len(example[0])):
        result += example[0][i] * parameters[i]

    return result

# The format of the examples parameter would be:
# examples = [[feature1, feature2, ...], y: int]

def batch_gradient(examples):
    initParameters = [0] * len(examples[0][0])

    maxLimit = 100000
    alpha = 0.001

    for i in range(maxLimit):
        newParameters = initParameters[:]  # Create a copy
        for j in range(len(newParameters)):
            res = 0
            for example in examples:
                h = hypothesis(example, initParameters)
                y = example[1]
                x = example[0][j]
                res += x * (y - h)
            newParameters[j] = initParameters[j] + alpha * res
        initParameters = newParameters
    
    return initParameters

examples = [
    [[2, 3, 7], 3],
    [[4, 2, 8], 7],
    [[9, 8, 5], 5],
    [[1, 5, 3], 9],
    [[2, 3, 5], 4],
    [[3, 2, 4], 8],
    [[5, 2, 3], 6],
    [[4, 1, 2], 7],
    [[2, 4, 3], 10],
    [[7, 5, 3], 6],
    [[1, 3, 7], 4],
    [[4, 2, 4], 8],
    [[8, 5, 6], 5],
    [[2, 9, 4], 11],
    [[3, 2, 2], 6],
     [[2, 3, 7], 3],
    [[4, 2, 8], 7],
    [[9, 8, 5], 5],
    [[1, 5, 3], 9],
    [[2, 3, 5], 4],
    [[3, 2, 4], 8],
    [[5, 2, 3], 6],
    [[4, 1, 2], 7],
    [[2, 4, 3], 10],
    [[7, 5, 3], 6],
    [[1, 3, 7], 4],
    [[4, 2, 4], 8],
    [[8, 5, 6], 5],
    [[2, 9, 4], 11],
    [[3, 2, 2], 6],
]

model = batch_gradient(examples)

def applyBD(testSet):
    sum = 0
    for i in range(len(testSet)):
        sum += testSet[i] * model[i]

    return sum

print(applyBD([3, 2, 1]))
        