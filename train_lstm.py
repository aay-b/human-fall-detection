import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from skeleton_dataset import SkeletonDataset   # ← matches your file
from lstm_model import ActionLSTM, CLASS_NAMES # ← matches your file

# ====== Hyperparameters ======
sequence_length = 45          # must match play_and_detect.py
input_size = 150              # ActionLSTM expects 150 (25 joints × 3 coords × 2 bodies)
hidden_size = 256
num_layers = 3
num_classes = 2  # 4: walk, fall, stand, jump
batch_size = 32
num_epochs = 20
learning_rate = 0.001

# ====== Dataset ======
dataset = SkeletonDataset(root_dir="labeled_skeletons", sequence_length=sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ====== Model ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActionLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ====== Training loop ======
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# ====== Save trained weights ======
torch.save(model.state_dict(), "action_lstm.pth")
print("✅ Saved weights to action_lstm.pth")
