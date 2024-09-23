import random
import math
from math import log10

def relu(x):
    return max(x, 0)

def gradient_clipping(gradient, threshold):
    return max(min(gradient, threshold), -threshold)

class RecurrentNeuralNetwork:
    def __init__(self, weights, features, bias):
        self.weights = weights
        self.features = features
        self.bias = bias
        self.hidden_states = []
        self.outputs = []
        self.position = 0
        self.previous_feature = 0

    def attention_mechanism(self, hidden_states):
        attention_data = []
        previous_delta = 0
        for i in range(len(hidden_states) - 1):
            delta = hidden_states[i+1] - hidden_states[i]
            new_delta = delta ** 2 / (previous_delta + 1e-6)
            previous_delta = new_delta
            attention_data.append(new_delta)
        normalized_data = sum(attention_data)
        return [x / normalized_data for x in attention_data]

    def forget_gate(self, output, feature):
        return relu(sum(self.weights[self.position] * feature) + self.bias[self.position])

    def input_gate(self, feature):
        gate_output = relu(sum(self.weights[self.position] * feature) + self.bias[self.position])
        model_feature = math.tanh(sum(self.weights[self.position] * self.previous_feature) + self.bias[self.position])
        return gate_output, model_feature

    def output_gate(self, feature):
        gate_output = relu(sum(self.weights[self.position] * feature) + self.bias[self.position])
        return gate_output * self.previous_feature

    def forward(self, previous_output=None):
        if self.position >= len(self.features):
            return previous_output
        current_hidden_state = sum(self.weights[self.position] * self.features[self.position]) + self.bias[self.position]
        self.hidden_states.append(current_hidden_state)
        attention_weights = self.attention_mechanism(self.hidden_states)
        weighted_sum = sum(attention_weights[i] * self.hidden_states[i] for i in range(len(attention_weights)))
        self.outputs.append(weighted_sum)
        self.position += 1
        return self.forward(weighted_sum)

    def backward(self, feature):
        if self.position <= 0:
            return feature
        bsum = sum(self.weights[self.position] * self.features[self.position] + self.bias[self.position])
        self.position -= 1
        feature.append(bsum)
        return self.backward(feature)

    def backpropagation_structure(self, target_output, learning_rate):
        for i in range(len(self.outputs)):
            delta = self.outputs[i] - target_output[i]
            gradient = delta / target_output[i]
            gradient = gradient_clipping(gradient, 1.0)  
            self.weights[i] = self.weights[i] - learning_rate * gradient
            self.bias[i] = self.bias[i] - learning_rate * gradient

    def main(self):
        forward_output = self.forward(None)
        backward_features = self.backward([])

class ConvolutionalNeuralNetwork:
    def __init__(self, weights, bias, features):
        self.weights = weights
        self.bias = bias
        self.features = features
        self.position = 0
        self.outputs = []

    def conv2d(self, input, filters, stride, padding):
        input_height = len(input)
        input_width = len(input[0])
        filter_height = len(filters)
        filter_width = len(filters[0])
        output_height = (input_height - filter_height + 2 * padding) // stride + 1
        output_width = (input_width - filter_width + 2 * padding) // stride + 1
        
        padded_input = [[0] * (input_width + 2 * padding) for _ in range(input_height + 2 * padding)]
        for i in range(input_height):
            for j in range(input_width):
                padded_input[i + padding][j + padding] = input[i][j]
        
        output = [[0] * output_width for _ in range(output_height)]
        for i in range(output_height):
            for j in range(output_width):
                for m in range(filter_height):
                    for n in range(filter_width):
                        output[i][j] += padded_input[i * stride + m][j * stride + n] * filters[m][n]
        return output
    
    def pooling(self, input, pool_size, stride):
        output_height = (len(input) - pool_size) // stride + 1
        output_width = (len(input[0]) - pool_size) // stride + 1
        output = [[0] * output_width for _ in range(output_height)]
        
        for i in range(0, len(input) - pool_size + 1, stride):
            for j in range(0, len(input[0]) - pool_size + 1, stride):
                output[i // stride][j // stride] = max(input[i + k][j + l] for k in range(pool_size) for l in range(pool_size))
        
        return output

    def forward(self):
        if self.position < len(self.features):
            self.outputs.append(self.weights * self.features[self.position] + self.bias[self.position])
            self.position += 1
            return self.forward()
        return self.outputs

    def backpropagation(self, target_output, learning_rate):
        for i in range(len(self.outputs)):
            delta = self.outputs[i] - target_output[i]
            gradient = delta / target_output[i]
            gradient = relu(gradient_clipping(gradient, 1))
            self.weights[i] -= learning_rate * gradient
            self.bias[i] -= learning_rate * gradient

def generate_caption(cnn_output, rnn):
    rnn.features = cnn_output
    caption = rnn.forward()
    return caption

def main():
    weights = [[random.random() for _ in range(2)] for _ in range(2)]
    bias = [random.random() for _ in range(2)]
    features = [[random.random() for _ in range(2)] for _ in range(2)]
    cnn = ConvolutionalNeuralNetwork(weights, bias, features)
    rnn = RecurrentNeuralNetwork(weights, features, bias)
    cnn_output = cnn.forward()
    caption = generate_caption(cnn_output, rnn)
    print("Generated Caption:", caption)

# Example usage
if __name__ == "__main__":
    main()
