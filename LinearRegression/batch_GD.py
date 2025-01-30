def hypothesis(example, parameters):
    return sum(x * p for x, p in zip(example[0], parameters))

# The format of the examples parameter would be:
# examples = [[feature1, feature2, ...], y: int]

def batch_gradient(examples):
    num_features = len(examples[0][0])  # Number of features
    parameters = [0] * num_features  # Initialize parameters to zero

    max_iterations = 100000  # Define inside function
    alpha = 0.0000001

    for iteration in range(max_iterations):
        new_parameters = parameters[:]  # Copy parameters
        
        for j in range(num_features):  # Update each parameter
            gradient_sum = 0
            for example in examples:
                h = hypothesis(example, parameters)  # Prediction
                y = example[1]
                x_j = example[0][j]  # Feature value
                gradient_sum += x_j * (y - h)

            new_parameters[j] = parameters[j] + (alpha/len(examples)) * gradient_sum  # Update rule

        parameters = new_parameters  # Apply new parameters

    return parameters

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
]

model = batch_gradient(examples)
print(model)

def applyBD(testSet):
    sum = 0
    for i in range(len(testSet)):
        sum += testSet[i] * model[i]

    return sum

print(applyBD([3, 2, 2]))
        