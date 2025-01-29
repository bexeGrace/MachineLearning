def hypothesis(example, parameters):
    return sum(x * p for x, p in zip(example[0], parameters))

# The format of the examples parameter would be:
# examples = [[feature1, feature2, ...], y: int]

def batch_gradient(examples):
    num_features = len(examples[0][0])  # Number of features
    parameters = [0] * num_features  # Initialize parameters to zero

    max_iterations = 1000  # Define inside function
    alpha = 0.0000001

    for iteration in range(max_iterations):
        new_parameters = parameters[:]  # Copy parameters
        
        for j in range(num_features):  # Update each parameter
            gradient_sum = 0
            for example in examples:
                h = hypothesis(example, new_parameters)  # Prediction
                y = example[1]  # Actual value
                x_j = example[0][j]  # Feature value
                gradient_sum += (x_j/len(examples)) * (y - h)

            new_parameters[j] = parameters[j] + alpha * gradient_sum  # Update rule

        parameters = new_parameters  # Apply new parameters

    return parameters

examples = [
    [[2104, 3], 400],
    [[1600, 3], 330],
    [[2400, 3], 369],
    [[1416, 2], 232],
    [[3000, 4], 540],
]

model = batch_gradient(examples)
print(model)

def applyBD(testSet):
    sum = 0
    for i in range(len(testSet)):
        sum += testSet[i] * model[i]

    return sum

print(applyBD([1000, 5]))
        