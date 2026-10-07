import torch 
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

#first we need to split the data into training and testing sets. We will use 80% of the data for training and 20% for testing.

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
    ])

dataset = ImageFolder(
    root="./dataset",
    transform=transform
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size ])

train_loader = DataLoader(
    train_dataset,
    batch_size=32,  
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,  
    shuffle=False
)


#OUR CNN NETWORK 

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.conv_layers = nn.Sequential(
            #one convolutional layer 
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16, 128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


m = model()

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    m.parameters(),
    lr = 0.001
)

epochs = 10

for epoch in range(epochs):

    # -------------------------
    # TRAINING
    # -------------------------

    m.train()

    running_loss = 0.0

    for images, labels in train_loader:

        output = m(images)

        labels = labels.float().unsqueeze(1)

        loss = criterion(output, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}], "
        f"Loss: {avg_loss:.4f}"
    )


    # -------------------------
    # TESTING
    # -------------------------

    m.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            output = m(images)

            predictions = (output >= 0.5).float()

            labels = labels.float().unsqueeze(1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    print(f"Test Accuracy: {accuracy:.4f}")