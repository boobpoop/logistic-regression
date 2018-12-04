import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    line_list = open(path, "r").readlines()
    data_list = []
    label_list = []
    for eachline in line_list:
        line_data = eachline.strip().split("\t")
        a_line = []
        for i in range(21):
            a_line.append(float(line_data[i]))
        data_list.append(a_line)
        label_list.append(float(line_data[21]))
    return np.array(data_list), np.array(label_list)

def sigmoid(input_value):
    output_value = 1.0 / (1 + np.exp(-input_value))
    return output_value

def gradient_descent(data_matrix, label_vector):
    m, n = data_matrix.shape
    weights = np.ones((n))
    alpha = 0.001
    batch_size = 29
    for i in range(500):
        for step in range(10):
            start = step * batch_size % m 
            error = label_vector[start: start + batch_size] - sigmoid(data_matrix[start: start + batch_size].dot(weights))
            weights = weights + alpha * data_matrix[start: start + batch_size].transpose().dot(error)
    return weights

def calculate(test_matrix, test_vector, weights):
    error_count = 0
    for i in range(test_matrix.shape[0]):
        predict = sigmoid(sum(test_matrix[i] * weights))
        if int(predict) != test_vector[i]:
            error_count += 1
    error_rate = error_count / float(test_matrix.shape[0])
    return error_rate

def test():
    data_path = "horseColicTraining.txt"
    data_matrix, label_vector = load_data(data_path)
    weights = gradient_descent(data_matrix, label_vector)
    data_path = "horseColicTest.txt"
    test_matrix, test_vector = load_data(data_path)
    error_rate = calculate(test_matrix, test_vector, weights)
    return error_rate

if __name__ == "__main__":
    error_rate = test()
    print(error_rate)

