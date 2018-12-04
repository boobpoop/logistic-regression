import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    line_list = open(path, "r").readlines()
    data_list = []
    label_list = []
    for eachline in line_list:
        line_data = eachline.strip().split("\t")
        data_list.append([1, float(line_data[0]), float(line_data[1])])
        label_list.append(int(line_data[2]))
    return np.array(data_list), np.array(label_list)

def sigmoid(input_value):
    output_value = 1.0 / (1 + np.exp(-input_value))
    return output_value

def gradient_descent(data_matrix, label_vector):
    m, n = data_matrix.shape
    weights = np.ones((n))
    alpha = 0.001
    batch_size = 10
    for i in range(500):
        for step in range(10):
            start = step * batch_size % m 
            error = label_vector[start: start + batch_size] - sigmoid(data_matrix[start: start + batch_size].dot(weights))
            weights = weights + alpha * data_matrix[start: start + batch_size].transpose().dot(error)
    return weights

def visualize(data_matrix, label_vector, weights):
    xcord0 = []
    ycord0 = []
    xcord1 = [] 
    ycord1 = []
    for i in range(data_matrix.shape[0]):
        if label_vector[i] == 0:
            xcord0.append(data_matrix[i][1])
            ycord0.append(data_matrix[i][2])
        else:
            xcord1.append(data_matrix[i][1])
            ycord1.append(data_matrix[i][2])
   
    plt.switch_backend("PDF")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0, ycord0, s = 30, c = "red", marker = "o", alpha = 1, label = "0")
    ax.scatter(xcord1, ycord1, s = 30, c = "blue", marker = "+", alpha = 1, label = "1")
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    plt.legend()
    plt.savefig("_1_2.pdf")
    plt.close()


if __name__ == "__main__":
    data_path = "testSet.txt"
    load_data(data_path)
    data_matrix, label_vector = load_data(data_path)
    weights = gradient_descent(data_matrix, label_vector)
    visualize(data_matrix, label_vector, weights)


