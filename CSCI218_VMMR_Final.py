# --------------------------------------------------
# CSCI281 Group Project - T01F
# Ying Jie, William, Hugo, Jun Yi
#
# This project implements an image-processing model using a pre-trained ResNet50
# and a custom CNN architecture to classify images from the "Most_Stolen_Cars" dataset.
# The goal is to identify stolen car models by comparing a ResNet50 model with a custom CNN.
#
# Models are trained using GPU CUDA (e.g., Colab T4).
#
# References:
#   - https://github.com/Helias/Car-Model-Recognition?tab=readme-ov-file
#   - https://www.kaggle.com/datasets/abhishektyagi001/vehicle-make-model-recognition-dataset-vmmrdb
#   - http://vmmrdb.cecsresearch.org
#   - https://www.nature.com/articles/s41598-024-51258-6#Sec10
# --------------------------------------------------

# ==================================================
# 1. Import Required Libraries and Modules
# ==================================================
import os                              # For operating system interactions (e.g., file paths)
import random                          # For random number generation
import numpy as np                     # For numerical operations and array handling
import torch                           # PyTorch for deep learning
import torch.nn as nn                  # Neural network modules
import torch.optim as optim            # Optimization algorithms
import torchvision.models as models    # Pre-trained models (e.g., ResNet50)
import torchvision.transforms as transforms  # Image transformations
from torchvision.datasets import ImageFolder   # Dataset loading from folders
from torch.utils.data import DataLoader, random_split  # Data loading and splitting utilities
from torchvision.models import ResNet50_Weights       # Pre-trained weights for ResNet50
from tqdm import tqdm                  # Progress bar for loops
import matplotlib.pyplot as plt        # Plotting library
from matplotlib import pyplot as plt   # (Redundant import kept for consistency)
import seaborn as sns                         # For advanced plotting (if needed)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# ==================================================
# 2. Configuration Parameters and Hyperparameters
# ==================================================
DEBUG = False                # Debug flag (currently not used)
EPOCHS = 20                  # Total number of training epochs for both models
MOMENTUM = 0.9               # Momentum parameter for the SGD optimizer
LEARNING_RATE = 0.01         # Learning rate for the optimizers
BATCH_SIZE = 16              # Batch size used for training and evaluation
THREADS = 0                  # Number of worker threads for DataLoader (0 means main thread)
USE_CUDA = torch.cuda.is_available()  # Flag to determine whether to use GPU (CUDA) or CPU

# ==================================================
# 3. File Paths for Data and Saving Results
# ==================================================
TRAINING_PATH = "train_file.csv"      # Placeholder file path (not used in the code)
VALIDATION_PATH = "test_file.csv"      # Placeholder file path (not used in the code)
TEST_PATH = "test_file.csv"            # Placeholder file path (not used in the code)
RESULTS_PATH = "results"               # Directory for saving results if needed
IMAGES_PATH = "Most_Stolen_Cars"         # Path to the folder containing the image dataset

# Check if the dataset folder exists; if not, raise an error.
if not os.path.exists(IMAGES_PATH):
    raise FileNotFoundError(f"Dataset folder not found: {IMAGES_PATH}")

# ==================================================
# 4. Define Image Transformations for Preprocessing
# ==================================================
# The following transformations will:
#  - Resize images to 224x224 pixels (the standard input size for many pre-trained models)
#  - Convert images to tensors
#  - Normalize the images using ImageNet's mean and standard deviation values
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==================================================
# 5. Load and Split the Dataset
# ==================================================
# Load the dataset using ImageFolder, which expects a directory where each subdirectory
# represents a class. The images are automatically labeled based on the folder names.
full_dataset = ImageFolder(root=IMAGES_PATH, transform=transform)
print("Class Mapping:", full_dataset.class_to_idx)

# Split the dataset into training (80%), validation (10%), and test (10%) sets.
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# ==================================================
# 6. Create DataLoaders for Each Subset
# ==================================================
# DataLoaders help load data in batches and optionally shuffle the dataset.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=THREADS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=THREADS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=THREADS)

# ==================================================
# 7. Device Selection
# ==================================================
# Automatically select the device: GPU (CUDA) if available, else CPU.
device = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {device}")

# ==================================================
# 8. Define the Model Architectures
# ==================================================

# --------------------------
# 8a. Custom ResNet50 Model
# --------------------------
# This model uses a pre-trained ResNet50 and replaces its final fully connected layer to match
# the number of classes in our dataset.
class CustomResNet(nn.Module):
    def __init__(self, num_classes, activation='relu'):
        super(CustomResNet, self).__init__()
        # Load the pre-trained ResNet50 with default weights.
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Select the activation function; default is ReLU.
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU()
        # Replace the final fully connected layer to output "num_classes" predictions.
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Pass the input through the ResNet50 model.
        return self.model(x)

# --------------------------
# 8b. Custom CNN Model
# --------------------------
# This is a simple CNN with three convolutional layers followed by two fully connected layers.
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        # First convolutional layer: from 3 input channels (RGB) to 64 feature maps.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer: from 64 to 128 feature maps.
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # Third convolutional layer: from 128 to 256 feature maps.
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # After three max-pooling layers (each halving the spatial dimensions),
        # an input image of size 224x224 becomes 28x28 (since 224 / 2^3 = 28).
        # The flattened size is: channels * height * width = 256 * 28 * 28.
        self.fc1 = nn.Linear(256 * 28 * 28, 1024)  # First fully connected layer.
        self.fc2 = nn.Linear(1024, num_classes)      # Second fully connected layer for output.
        
        # Max pooling layer with kernel size 2x2.
        self.pool = nn.MaxPool2d(2, 2)
        # ReLU activation function.
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply first conv layer, ReLU activation, then pooling.
        x = self.pool(self.relu(self.conv1(x)))
        # Apply second conv layer, ReLU activation, then pooling.
        x = self.pool(self.relu(self.conv2(x)))
        # Apply third conv layer, ReLU activation, then pooling.
        x = self.pool(self.relu(self.conv3(x)))
        # Flatten the feature maps into a vector.
        x = x.view(-1, 256 * 28 * 28)
        # Pass through the first fully connected layer and apply ReLU.
        x = self.relu(self.fc1(x))
        # Pass through the final fully connected layer to get class scores.
        x = self.fc2(x)
        return x

# --------------------------------------------------
# Function: classify_random_image
# --------------------------------------------------
# This function selects a random image from a given dataset, uses the provided model
# to classify the image, and then displays the image along with its actual label,
# predicted label, and confidence score.
def classify_random_image(model, dataset, device):
    model.eval()
    # Randomly select an index from the dataset.
    idx = random.randint(0, len(dataset) - 1)
    img, label = dataset[idx]
    # Add a batch dimension and move the image to the selected device.
    img_tensor = img.unsqueeze(0).to(device)

    # Get the model's output and compute probabilities using softmax.
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

    # Denormalize the image for proper visualization.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_denormalized = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)

    # Display the image with predicted and actual labels along with the confidence score.
    plt.figure(figsize=(5, 5))
    plt.imshow(img_denormalized.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title(f"Actual: {full_dataset.classes[label]}\n"
              f"Predicted: {full_dataset.classes[predicted_class.item()]}\n"
              f"Confidence Score: {confidence.item() * 100:.2f}%")
    plt.show()

# ==================================================
# 9. Instantiate Models and Define Optimizers
# ==================================================
# Get the number of classes from the dataset.
num_classes = len(full_dataset.classes)

# Create model instances and move them to the appropriate device.
model = CustomResNet(num_classes=num_classes).to(device)    # ResNet-based model.
cnn_model = CustomCNN(num_classes=num_classes).to(device)      # Custom CNN model.

# Define the loss function (CrossEntropyLoss is standard for classification).
criterion = nn.CrossEntropyLoss()

# Define optimizers for both models using Stochastic Gradient Descent (SGD).
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# ==================================================
# 10. Training the ResNet Model
# ==================================================
# Lists to store training and validation losses for each epoch.
train_losses, val_losses = [], []
print("Starting Training...")
for epoch in range(EPOCHS):
    model.train()  # Set model to training mode.
    running_loss = 0.0
    # Create a progress bar for the current epoch.
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()   # Zero out gradients.
        outputs = model(images) # Forward pass.
        loss = criterion(outputs, labels)  # Compute loss.
        loss.backward()         # Backpropagation.
        optimizer.step()        # Update parameters.
        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation phase: Evaluate the model on the validation set.
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# ==================================================
# 11. Training the Custom CNN Model
# ==================================================
# Lists to store CNN model training and validation losses.
cnn_train_losses, cnn_val_losses = [], []
print("Starting CNN Training...")
for epoch in range(EPOCHS):
    cnn_model.train()  # Set CNN model to training mode.
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"CNN Epoch {epoch+1}/{EPOCHS}", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        cnn_optimizer.zero_grad()  # Reset gradients.
        outputs = cnn_model(images)  # Forward pass.
        loss = criterion(outputs, labels)  # Compute loss.
        loss.backward()  # Backpropagation.
        cnn_optimizer.step()  # Update model parameters.
        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    avg_train_loss = running_loss / len(train_loader)
    cnn_train_losses.append(avg_train_loss)

    # Validation phase for the CNN model.
    cnn_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = cnn_model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    cnn_val_losses.append(avg_val_loss)

    print(f"CNN Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# ==================================================
# 12. Save Trained Models
# ==================================================
# Save the trained model parameters (state dictionaries) to files.
torch.save(model.state_dict(), "car_model.pth")
print("ResNet Model Training Complete & Saved!")
torch.save(cnn_model.state_dict(), "cnn_car_model.pth")
print("CNN Model Training Complete & Saved!")

# ==================================================
# 13. Evaluate Validation Accuracy
# ==================================================
# --- ResNet Model Validation Accuracy ---
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Validation Accuracy (ResNet): {100 * correct / total:.2f}%")

# --- Custom CNN Model Validation Accuracy ---
cnn_model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Validation Accuracy (CNN): {100 * correct / total:.2f}%")

# ==================================================
# 14. Plot Training vs. Validation Loss Curves
# ==================================================
# --- Plot for ResNet Model ---
best_epoch = np.argmin(val_losses) + 1  # Best epoch index (human-readable)
best_val_loss = min(val_losses)
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS+1), train_losses, label="Train Loss", color='blue')
plt.plot(range(1, EPOCHS+1), val_losses, label="Validation Loss", color='orange')
plt.scatter(best_epoch, best_val_loss, color="red", zorder=5)
plt.text(best_epoch + 1, best_val_loss, f"Best Epoch: {best_epoch}\nLoss: {best_val_loss:.4f}",
         fontsize=12, color="red", verticalalignment="bottom")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("ResNet Training vs Validation Loss")
plt.tight_layout()
plt.show()

# --- Plot for Custom CNN Model ---
best_cnn_epoch = np.argmin(cnn_val_losses) + 1
best_cnn_val_loss = min(cnn_val_losses)
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS+1), cnn_train_losses, label="Train Loss (CNN)", color='blue')
plt.plot(range(1, EPOCHS+1), cnn_val_losses, label="Validation Loss (CNN)", color='orange')
plt.scatter(best_cnn_epoch, best_cnn_val_loss, color="red", zorder=5)
plt.text(best_cnn_epoch + 1, best_cnn_val_loss, f"Best Epoch: {best_cnn_epoch}\nLoss: {best_cnn_val_loss:.4f}",
         fontsize=12, color="red", verticalalignment="bottom")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("CNN Training vs Validation Loss")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 14.1 Confusion Matrix for ResNet Model
# --------------------------------------------------
# Set the ResNet model to evaluation mode and collect predictions.
model.eval()
y_true_resnet, y_pred_resnet = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true_resnet.extend(labels.cpu().numpy())
        y_pred_resnet.extend(predicted.cpu().numpy())

# Compute the confusion matrix using scikit-learn.
cm_resnet = confusion_matrix(y_true_resnet, y_pred_resnet)

# Plot the confusion matrix using seaborn heatmap.
plt.figure(figsize=(10, 8))
sns.heatmap(cm_resnet, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for ResNet")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# --------------------------------------------------
# Confusion Matrix for ResNet Model with Class Labels
# --------------------------------------------------
model.eval()
y_true_resnet, y_pred_resnet = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true_resnet.extend(labels.cpu().numpy())
        y_pred_resnet.extend(predicted.cpu().numpy())

cm_resnet = confusion_matrix(y_true_resnet, y_pred_resnet)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_resnet, annot=True, fmt="d", cmap="Blues",
            xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
plt.title("Confusion Matrix for ResNet")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

# Identify the most confused class pair for ResNet (ignoring the diagonal)
cm_resnet_no_diag = cm_resnet.copy()
np.fill_diagonal(cm_resnet_no_diag, 0)
max_index = np.unravel_index(np.argmax(cm_resnet_no_diag, axis=None), cm_resnet_no_diag.shape)
most_confused_count = cm_resnet_no_diag[max_index]
most_confused_pair = (full_dataset.classes[max_index[0]], full_dataset.classes[max_index[1]])
print(f"Most confused class pair for ResNet: {most_confused_pair} with {most_confused_count} misclassifications")

# --------------------------------------------------
# Confusion Matrix for Custom CNN Model with Class Labels
# --------------------------------------------------
cnn_model.eval()
y_true_cnn, y_pred_cnn = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs, 1)
        y_true_cnn.extend(labels.cpu().numpy())
        y_pred_cnn.extend(predicted.cpu().numpy())

cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Greens",
            xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
plt.title("Confusion Matrix for Custom CNN")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

# Identify the most confused class pair for Custom CNN (ignoring the diagonal)
cm_cnn_no_diag = cm_cnn.copy()
np.fill_diagonal(cm_cnn_no_diag, 0)
max_index_cnn = np.unravel_index(np.argmax(cm_cnn_no_diag, axis=None), cm_cnn_no_diag.shape)
most_confused_count_cnn = cm_cnn_no_diag[max_index_cnn]
most_confused_pair_cnn = (full_dataset.classes[max_index_cnn[0]], full_dataset.classes[max_index_cnn[1]])
print(f"Most confused class pair for Custom CNN: {most_confused_pair_cnn} with {most_confused_count_cnn} misclassifications")





# ==================================================
# 16. Visualize Correct Predictions with Confidence Scores
# ==================================================
# These mean and standard deviation values are the same as used for normalization.
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Prepare a dictionary to store correctly predicted images by class.
# Here, we assume there are 10 classes (adjust if necessary).
class_images = {i: [] for i in range(10)}

# Loop through the test dataset to classify images and store correct predictions.
for idx in range(len(test_dataset)):
    img, label = test_dataset[idx]
    img_tensor = img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
    # Only store the image if the prediction is correct.
    if predicted_class == label:
        class_images[label].append((img, label, predicted_class, confidence.item()))

# Set up the plot for displaying correct predictions.
plt.figure(figsize=(15, 15))
plt.suptitle("Correct Predictions for Each Class (5 per class) with Confidence Scores", fontsize=20)
plt.subplots_adjust(top=0.80)  # Adjust top margin for the title

# Display 5 images per class (for 5 classes in this example)
for class_idx in range(5):
    for i, (img, label, predicted_class, confidence) in enumerate(class_images[class_idx][:5]):
        # Denormalize the image before displaying.
        img_denormalized = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        plt.subplot(5, 5, class_idx * 5 + i + 1)
        plt.imshow(img_denormalized.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title(f"Pred: {full_dataset.classes[predicted_class.item()]}\n"
                  f"Conf: {confidence*100:.2f}%\nActual: {full_dataset.classes[label]}")
plt.tight_layout()
plt.show()

# ==================================================
# 17. Classify a Random Image using the Helper Function
# ==================================================
# This function call randomly selects an image from the test dataset,
# performs classification using the ResNet model, and displays the result.
classify_random_image(model, test_dataset, device)
