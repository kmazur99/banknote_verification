import numpy as np
import pandas as pd
import random
from tqdm import tqdm

banknotes = "data_banknote_authentication.csv"

df = pd.read_csv(banknotes)
features = df.values[:, :-1] # take first 4 columns(banknote features)
labels = df.values[:, -1] # take last column(output either 0 or 1)

# Activation functions
def sigmoid(values):
    return 1 / (1 + np.exp(-values))

def relu(values):
    return np.maximum(0, values)

def tanh(values):
    return np.tanh(values)

# Define layer
class Layer:
    def __init__(self, in_features, neurons, activation):
        self.neurons = neurons # Number of neurons for this layer
        self.weights = np.random.randn(in_features, self.neurons) # Calculate random weights between input and output neurons
        self.biases = np.zeros(self.neurons) # Initialize bias for each neuron in this layer (currently 0)
        self.activation = activation # activation function for this layer

    # Forward pass through the layer
    def forward_pass(self, in_data):
        # Apply activation function to (weighted sum of inputs + biases)
        output = self.activation(np.dot(in_data, self.weights) + self.biases)
        return output # Output of current layer

# Define neural network
class Network:
    def __init__(self, in_features):
        self.layers = [] # Store layers
        self.in_features = in_features # Number of input features

    # Add new layer to the neural network
    def add(self, neurons, activation):
        new_layer = Layer(self.in_features, neurons, activation) # initialize new layer
        self.layers.append(new_layer) # add layer to list of layers
        self.in_features = neurons # Update number of input features

    # Forward pass through the entire neural network
    def forward_pass(self, in_data):
        output = in_data
        # Iterate over all layers and apply forward passes
        for layer in self.layers:
            output = layer.forward_pass(output)
        return output # final output of the neural network
    
    def set_weights(self, weights):
        # Unpack weights and set them in the network, assumes weights are in a flat array
        for i in range(len(self.layers)):
            layer_weights = weights[:self.layers[i].weights.size]
            self.layers[i].weights = layer_weights.reshape(self.layers[i].weights.shape)
            weights = weights[self.layers[i].weights.size:]
        
        # Unpack biases and set them in the network, assumes biases are in a flat array
        for i in range(len(self.layers)):
            layer_biases = weights[:self.layers[i].biases.size]
            self.layers[i].biases = layer_biases.reshape(self.layers[i].biases.shape)
            weights = weights[self.layers[i].biases.size:]
    
    def get_accuracy(self, features, labels):
        correct_predictions = 0
        for xi, yi in zip(features, labels):
            prediction = self.forward_pass(xi)
            if prediction == yi:
                correct_predictions += 1
        accuracy = correct_predictions / len(labels)
        return accuracy
    
    def num_parameters(self):
        # Calculate number of parameters in the network
        num_parameters = 0
        for layer in self.layers:
            num_parameters += layer.weights.size + layer.biases.size
        return num_parameters
    
    def print_layers(self):
        # Print weights and biases for each layer
        for i in range(len(self.layers)):
            print(f"\nLayer {i+1} weights: {self.layers[i].weights}")
            print(f"Layer {i+1} biases: {self.layers[i].biases}")
            print()

# Define particle class
class Particle:
    def __init__ (self, dimensions):
        self.position = np.random.uniform(-1, 1, dimensions)
        self.velocity = np.random.uniform(-1, 1, dimensions)
        self.best_position = self.position.copy()
        self.best_score = float('inf')

class PSO:
    def __init__(self, network, num_particles, num_informants):
        self.network = network
        self.particles = [Particle(network.num_parameters()) for i in range(num_particles)]
        self.informants = {p: random.sample(self.particles, num_informants) for p in self.particles}
        self.global_best_position = np.random.uniform(-1, 1, network.num_parameters())
        self.global_best_score = float('inf')
    
    def optimise(self, features, labels, epochs):
        # Inertia weight starts high for exploration and decreases for exploitation
        w_max = 0.9  # Starting inertia weight
        w_min = 0.4  # Ending inertia weight
        
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient

        for epoch in tqdm(range(epochs), desc="Training", unit="epoch", position=0, ncols=100):
            # Linearly decreasing inertia weight
            w = w_max - ((w_max - w_min) * epoch / epochs)

            for particle in self.particles:
                self.network.set_weights(particle.position)
                accuracy = self.network.get_accuracy(features, labels)
                score = 1 - accuracy

                # Update particle personal best
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()
            
            # Update particles
            for particle in self.particles:
                informants_best = min(self.informants[particle], key=lambda p: p.best_score).best_position
                r1, r2 = np.random.rand(), np.random.rand()

                # Update velocity with random factors
                particle.velocity = (w * particle.velocity + 
                                     c1 * r1 * (particle.best_position - particle.position) + 
                                     c2 * r2 * (informants_best - particle.position))

                # Update position
                particle.position += particle.velocity

            ann.print_layers()
            print(f"Epoch {epoch+1} best score: {self.global_best_score}")
            
        self.network.set_weights(self.global_best_position)
    
    def update_particle(self, particle, informants_best_position):
        # Update particle velocity
        particle.velocity = 0.5 * particle.velocity + 0.5 * (particle.best_position - particle.position) + 0.5 * (informants_best_position - particle.position)
        # Update particle position
        particle.position = particle.position + particle.velocity


ann = Network(in_features=4) # specify number of input features

# Add layers, specify number of neurons and activation function used
ann.add(neurons=5, activation=relu)
ann.add(neurons=6, activation=relu)
ann.add(neurons=4, activation=relu)
ann.add(neurons=1, activation=sigmoid)

# Test neural network
def test_ann(ann, features, labels):
    predictions = ann.forward_pass(features)
    
    correct_predictions = 0
    predicted_labels = []
    
    # Classify predicted values
    for prediction in predictions:
        if prediction > 0.5:
            predicted_label = 1
        else:
            predicted_label = 0
        
        predicted_labels.append(predicted_label)
    
    
    for i in range(len(labels)):
        # print predictions
        print(f"Banknote {i}, Predicted label: {predicted_labels[i]}, Actual label: {int(labels[i])}")
        
        # Add to correct count if prediction matches actual label    
        if predicted_labels[i] == labels[i]:
            correct_predictions += 1
    
    # Calculate and print accuracy
    accuracy = (correct_predictions / len(labels))
    print(f"\nAccuracy: {accuracy}")
    print(f"Correct: {correct_predictions}/{len(labels)}")

pso = PSO(ann, num_particles=50, num_informants=8)
pso.optimise(features, labels, epochs=100)

final_accuracy = ann.get_accuracy(features, labels)
print(f"Final Model Accuracy: {final_accuracy * 100:.2f}%")

test_ann(ann, features, labels)
