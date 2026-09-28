import pandas as pd

df = pd.read_csv(
    "customer_churn_dataset-testing-master.csv",
    encoding="latin1"
)

#remove the customer ID
X = df.drop(columns=['CustomerID', "Churn"])
y = df["Churn"]

#now some of the data is in strings, we need to convert it into passable parameter to ANN

#gender, subscription type and contract length are in string format so we need to do one hot encoding on then
X = pd.get_dummies(
    X,
    columns=[
        "Gender",
        "Subscription Type",
        "Contract Length"
    ]
)
#get dummies convert to true and false 
X = X.astype(float)
y = y.astype(float)

# print(X.dtypes)

#now we need to do test and train split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#scaling on the data 
mean = X_train.mean()
std = X_train.std()

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
#we do not do scaling on test data as it should be unseen and can't be used for training our nueral network 
import torch 
from torch import tensor
X_train_tensor = tensor(X_train.values, dtype=torch.float32)
X_test_tensor = tensor(X_test.values, dtype=torch.float32)
y_train_tensor = tensor(y_train.values, dtype=torch.float32).reshape(-1,1)
y_test_tensor = tensor(y_test.values, dtype=torch.float32).reshape(-1,1)

# print(X_train_tensor.shape)
# print(X_test_tensor.shape)

# print(y_train_tensor.shape)
# print(y_test_tensor.shape)

#CREATING OUR NEURAL NETWORK
import torch.nn as nn


# print("X shape:", X.shape)
# print("y shape:", y.shape)

# print("X_train:", X_train.shape)
# print("y_train:", y_train.shape)

# print("X_test:", X_test.shape)
# print("y_test:", y_test.shape)

class model(nn.Module):

    def __init__(self, features_count):
        super().__init__()

        #layers 
        self.layer1 = nn.Linear(features_count, 10)
        self.layer2 = nn.Linear(10, 8)
        self.layer3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)

        return x

features_count = X_train_tensor.shape[1]
model = model(features_count)

criterion = nn.BCEWithLogitsLoss()


epochs = 100
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(epochs):

    optimizer.zero_grad()

    output = model(X_train_tensor)

    loss = criterion(output, y_train_tensor)

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}"
    )

model.eval()

with torch.no_grad():

    output = model(X_test_tensor)

    probabilities = torch.sigmoid(output)

    predictions = (probabilities >= 0.5).float()

    # print(type(predictions))
    # print(type(y_test_tensor))

    # print(predictions.shape)
    # print(y_test_tensor.shape)

    correct = (predictions == y_test_tensor).sum()

    accuracy = correct / len(y_test_tensor)

    print("Test Accuracy:", accuracy)


from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

y_true = y_test_tensor.numpy()
y_pred = predictions.numpy()

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))