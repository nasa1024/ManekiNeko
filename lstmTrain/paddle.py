import paddle
import numpy as np
import pandas as pd
from paddle.io import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def preprocess_data(data, features, target, seq_length, train_size):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, :])
        y.append(scaled_data[i, target])
    X, y = np.array(X), np.array(y)
    
    train_length = int(len(X) * train_size)
    X_train, X_test = X[:train_length], X[train_length:]
    y_train, y_test = y[:train_length], y[train_length:]
    
    return X_train, y_train, X_test, y_test, scaler

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = paddle.to_tensor(X, dtype='float32')
        self.y = paddle.to_tensor(y, dtype='float32')
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BiLSTMModel(paddle.nn.Layer):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(BiLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.maxpool = paddle.nn.MaxPool1D(kernel_size=2)
        self.bidir_lstm1 = paddle.nn.LSTM(input_size, hidden_size, num_layers, direction='bidirectional', dropout=dropout_rate)
        self.bidir_lstm2 = paddle.nn.LSTM(2*hidden_size, hidden_size, num_layers, direction='bidirectional', dropout=dropout_rate)
        self.dropout = paddle.nn.Dropout(dropout_rate)
        self.lstm3 = paddle.nn.LSTM(2*hidden_size, hidden_size, num_layers)
        self.lstm4 = paddle.nn.LSTM(hidden_size, hidden_size, num_layers)
        self.fc = paddle.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.maxpool(x.transpose((0, 2, 1))).transpose((0, 2, 1))
        
        out, (_, _) = self.bidir_lstm1(x)
        out, (_, _) = self.bidir_lstm2(out)
        out = self.dropout(out)
        
        out, (_, _) = self.lstm3(out)
        out, (_, _) = self.lstm4(out)
        
        out = self.fc(out[:, -1, :])
        return out

def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.clear_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with paddle.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
    return running_loss / len(dataloader)

# Set device
paddle.set_device('gpu:0' if paddle.is_compiled_with_cuda() else 'cpu')

# Load and preprocess data
data = pd.read_csv("your_data.csv")
features = [...]  # Your list of features
target = [1, 2, 3, 4]  # Indices of 'open', 'high', 'low', 'close'

seq_length = 60
train_size = 0.8

X_train, y_train, X_test, y_test, scaler = preprocess_data(data, features, target, seq_length, train_size)

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

# Create DataLoader and model
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = len(features)
hidden_size = 128
num_layers = 2
output_size = 4
dropout_rate = 0.3

model = BiLSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate)

# Set loss function, optimizer, and train the model
epochs = 50
lr = 0.001

criterion = paddle.nn.MSELoss()
optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())

for epoch in range(1, epochs + 1):
    train_loss = train(model, train_dataloader, criterion, optimizer)
    test_loss = evaluate(model, test_dataloader, criterion)
    print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
