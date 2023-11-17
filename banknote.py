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
        self.biases = np.zeros(self.neurons) # Initialize bias for each neuron in this layer to 0
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
        for i in range(len(self.layers)):
            layer_weights = weights[:self.layers[i].weights.size] # Load weights and set them in the network
            self.layers[i].weights = layer_weights.reshape(self.layers[i].weights.shape) # Reshape weights to match layer shape
            
    def set_biases(self, biases):
        # Unpack biases and set them in the network
        for i in range(len(self.layers)):
            layer_biases = biases[:self.layers[i].biases.size] # Load biases and set them in the network
            self.layers[i].biases = layer_biases.reshape(self.layers[i].biases.shape) # Reshape biases to match layer shape
    
    def get_accuracy(self, features, labels): # Get current accuracy of the network
        correct_predictions = 0
        for xi, yi in zip(features, labels):
            prediction = self.forward_pass(xi)
            if prediction == yi:
                correct_predictions += 1
        accuracy = correct_predictions / len(labels)
        return accuracy
    
    def get_parameters(self): # Get number of parameters in the network (weights + biases for each layer)
        num_parameters = 0
        for layer in self.layers:
            num_parameters += layer.weights.size + layer.biases.size
        return num_parameters
    
    def print_layers(self):
        # Print weights and biases for each layer
        for i in range(len(self.layers)):
            print(f"\nLayer {i+1} weights: \n{self.layers[i].weights}")
            print(f"\nLayer {i+1} biases: \n{self.layers[i].biases}\n")

# Define particle class
class Particle:
    def __init__ (self, dimensions):
        self.position = np.random.uniform(-1, 1, dimensions) # Initialize random position
        self.velocity = np.random.uniform(-1, 1, dimensions) # Initialize random velocity
        self.best_position = self.position.copy() # Initialize best position to current position
        self.best_score = 0 # Initialize best score to 0

class PSO:
    def __init__(self, network, num_particles, num_informants):
        self.network = network # Initialize neural network
        self.particles = [] # List to store particles
        self.informants = {} # Dictionary to store informants for each particle
        self.num_informants = num_informants # Number of informants for each particle
        self.global_best_position = np.random.uniform(-1, 1, network.get_parameters()) # Initialize global best position to random position
        self.global_best_score = 0 # Initialize global best score to 0
    
        # Initialize particles
        for i in range(num_particles):
            self.particles.append(Particle(network.get_parameters())) # Initialize particle with dimensions == number of parameters in the network

    def train(self, features, labels, epochs, inertia_start, inertia_end, cognitive_coefficient, social_coefficient):
        for epoch in tqdm(range(epochs), desc="Training", unit=" Epoch", ncols=100):
            # Decrease inertia weight linearly from inertia_start to inertia_end to explore more at the start and exploit more at the end
            inertia_weight = inertia_start - (inertia_start - inertia_end) * (epoch / epochs) 

            for particle in self.particles:

                # Assign informants to each particle
                self.informants[particle] = random.sample(self.particles, self.num_informants)

                # Update network weights and biases with particle position
                self.network.set_weights(particle.position)
                self.network.set_biases(particle.position) 

                # Get particle current score based on accuracy
                current_score = self.network.get_accuracy(features, labels)

                # Update particle personal best
                if current_score > particle.best_score:
                    particle.best_score = current_score
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if current_score > self.global_best_score:
                    self.global_best_score = current_score 
                    self.global_best_position = particle.position.copy()
            
                informants_best_position = 0 # Initialize informants best position to 0
                informants_best_score = 0 # Initialize best score to 0

                # Get the informant with the best score and get its best position
                for informant in self.informants[particle]: # Iterate over informants
                    if informant.best_score > informants_best_score: # Update best score and best position if current informant has a better score
                        informants_best_score = informant.best_score 
                        informants_best_position = informant.best_position.copy()

                # Random numbers between 0 and 1
                random1 = np.random.rand()
                random2 = np.random.rand()

                # Update each particle velocity according to the formula using informants best position
                particle.velocity = (inertia_weight * particle.velocity + cognitive_coefficient * random1 * (particle.best_position - particle.position) + social_coefficient * random2 * (informants_best_position - particle.position))

                # Update each particle position by adding the velocity
                particle.position += particle.velocity

            # Print real time info on training
            ann.print_layers()
            print(f"Loss: {1 - self.global_best_score} | Accuracy: {self.global_best_score}")
        
        # Update network weights and biases with global best position
        self.network.set_weights(self.global_best_position)
        self.network.set_biases(self.global_best_position)

# Test neural network
def test_ann(ann, features, labels):
    predictions = ann.forward_pass(features)
    
    correct_predictions = 0
    predicted_labels = []
    
    # Classify predicted values as either 0 or 1 based on threshold of 0.5
    for prediction in predictions:
        if prediction > 0.5:
            predicted_label = 1
        else:
            predicted_label = 0
        
        predicted_labels.append(predicted_label)
    
    for i in range(len(labels)):
        # Add to correct count if prediction matches actual label    
        if predicted_labels[i] == labels[i]:
            correct_predictions += 1
    
    # Calculate and print accuracy
    accuracy = (correct_predictions / len(labels))
    print("\n----------------------")
    print(f"Accuracy: {100 * accuracy:.2f}%") 
    print(f"Correct: {correct_predictions}/{len(labels)}\n")

ann = Network(in_features=4) # Initialize neural network

# Add layers, specify number of neurons and activation function used
ann.add(neurons=4, activation=relu) # Input layer 4 neurons (4 features in)
ann.add(neurons=8, activation=relu)
ann.add(neurons=4, activation=relu)
ann.add(neurons=1, activation=sigmoid) # Output layer 1 neuron (output either 0 or 1)

# Define PSO hyperparameters and train neural network
pso = PSO(ann, num_particles=50, num_informants=5)
pso.train(features, labels, epochs=100, inertia_start=0.9, inertia_end=0.4, cognitive_coefficient=1.5, social_coefficient=1.5) 

# Test neural network
test_ann(ann, features, labels)