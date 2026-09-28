import pandas as pd 
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset

#WORKING ON OUR DATASET 
df = pd.read_csv("dataset/mnist_train.csv", header=None)

#print(df.head())

#print(df.shape) -> (60000, 785)

# print(df.columns) Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
#        ...
#        775, 776, 777, 778, 779, 780, 781, 782, 783, 784],
#       dtype='int64', length=785)

#our input and output values 
X = df.iloc[: , 1:]
y = df.iloc[:, 0]


X = torch.tensor(X.values, dtype=torch.float32) / 255
y = torch.tensor(y.values, dtype=torch.long)

#debug purpose 
# print(X.shape)
# print(y.shape)
# print(X.dtype)
# print(y.dtype)

#CLASS for our neural network
class Model(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.layer1 = nn.Linear(num_features, 512)
        self.layer2 = nn.Linear(512, 128)
        self.layer3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)

        return x 

#now lets create our train data loader 
dataset = TensorDataset(X, y)

train_loader = DataLoader(
    dataset, 
    batch_size=64, 
    shuffle = True
)

# X_batch, y_batch = next(iter(train_loader))

# print(X_batch.shape)
# print(y_batch.shape)

model = Model(784)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr = 0.01
)

epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        output = model(X_batch)

        loss = criterion(output, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

test_df = pd.read_csv("dataset/mnist_test.csv", header=None)

X_test = test_df.iloc[:, 1:]
y_test = test_df.iloc[:, 0]

X_test = torch.tensor(X_test.values, dtype=torch.float32) / 255.0
y_test = torch.tensor(y_test.values, dtype=torch.long)

test_dataset = TensorDataset(X_test, y_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

model.eval()

correct = 0
total = 0
import matplotlib.pyplot as plt
with torch.no_grad():

    for X_batch, y_batch in test_loader:

        output = model(X_batch)

        predictions = torch.argmax(output, dim=1)

        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total

print("Test Accuracy:", accuracy)


image = X_test[0]
actual = y_test[0]

model.eval()

with torch.no_grad():
    output = model(image.unsqueeze(0))
    prediction = torch.argmax(output, dim=1)

print("Prediction:", prediction.item())
print("Actual:", actual.item())

plt.imshow(image.reshape(28, 28), cmap="gray")
plt.title(f"Prediction: {prediction.item()} | Actual: {actual.item()}")
plt.axis("off")
plt.show()