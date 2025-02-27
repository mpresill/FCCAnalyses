import torch
import torch.nn as nn
import torch.optim as optim

class CNN_Model(nn.Module):
    def __init__(self, num_params):
        super(CNN_Model, self).__init__()

        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=1)
        
        # MaxPooling layer (optional but can help reduce dimensionality)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layer to output the prediction
        self.fc1 = nn.Linear(256*int((((((num_params+1)/2)+1)/2)+1)/2), 256)  # Adjust the input dimension based on pooling
        self.fc2 = nn.Linear(256, 1)  # Output layer with 2 units (Signal and Background)

    def forward(self, x):
        # Forward pass through convolution layers with ReLU activations
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Flatten the output from the convolution layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)  # Output layer

        return torch.sigmoid(x)
    
class DNN(nn.Module):
    def __init__(self, num_params):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(num_params, num_params*2)  
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(num_params*2, num_params*4) 
        self.fc3 = nn.Linear(num_params*4, num_params*8)
        self.fc4 = nn.Linear(num_params*8, num_params*16)
        self.fc5 = nn.Linear(num_params*16, 1)

    def forward(self, x):
        x = self.fc1(x) 
        x = self.relu(x)  
        x = self.fc2(x)
        x = self.relu(x)  
        x = self.fc3(x)   
        x = self.relu(x)  
        x = self.fc4(x) 
        x = self.relu(x)  
        x = self.fc5(x)
        
        return torch.sigmoid(x)