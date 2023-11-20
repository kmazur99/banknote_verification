import numpy as np
import pandas as pd
import random
from tqdm import tqdm

df = pd.read_csv("data_banknote_authentication.csv") # Read csv file
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
    
    def set_weights(self, weights): # Set weights according to layer shape and order
        start_index = 0 # Where to start loading weights from
        for layer in self.layers:
            end_index = start_index + layer.weights.size # Calculate end index
            current_layer_weights = weights[start_index:end_index] # Load weights for current layer
            layer.weights = current_layer_weights.reshape(layer.weights.shape) # Reshape weights to match layer shape
            start_index = end_index # Update start index for next layer
            
    def set_biases(self, biases): # Set biases according to layer shape and order
        start_index = 0 # Where to start loading biases from
        for layer in self.layers:
            end_index = start_index + layer.biases.size # Calculate end index
            current_layer_biases = biases[start_index:end_index] # Load biases for current layer
            layer.biases = current_layer_biases.reshape(layer.biases.shape) # Reshape biases to match layer shape
            start_index = end_index # Update start index for next layer
    
    def get_accuracy(self, features, labels): # Get current accuracy of the network
        correct_predictions = 0
        for feature, label in zip(features, labels):
            prediction = self.forward_pass(feature)
            if prediction == label:
                correct_predictions += 1
        return correct_predictions / len(labels) # Return accuracy 
    
    def get_parameters(self): # Get number of parameters in the network (number of weights + number of biases for each layer)
        parameters = 0
        for layer in self.layers:
            parameters += layer.weights.size + layer.biases.size
        return parameters
    
    def print_layers(self):
        # Print weights and biases for each layer
        for i in range(len(self.layers)):
            print(f"\nLayer {i+1} weights: \n{self.layers[i].weights}")
            print(f"\nLayer {i+1} biases: \n{self.layers[i].biases}\n")

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
        for epoch in tqdm(range(epochs), desc="Training", ncols=100):
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
    accuracy = ann.get_accuracy(features, labels)
    print("\n----------------------")
    print(f"Accuracy: {100 * accuracy:.2f}%") 
    print(f"Correct: {int(accuracy * len(labels))}/{len(labels)}\n") # Extract number of correct predictions from accuracy and print it

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