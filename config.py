import os

# Define the save directory path
documents_folder = os.path.expanduser("~/Documents")
save_dir = os.path.join(documents_folder, "plots")

# Create the directory if it does not exist
os.makedirs(save_dir, exist_ok=True)

# File paths
train_csv = 'Train.csv'
test_csv = 'Test.csv'
image_folder_train = 'Train/'
image_folder_test = 'Test/'

# Model Configuration
model_name = 'custom_cnn'  # options: 'resnet18', 'vgg11', 'EfficientNet', 'convnext', 'custom_cnn', 'MicroVGG',
# 'MobileNet'

# Training hyperparameters
batch_size = 16
numClasses = 43
learning_rate = 0.01
num_epochs = 50
warmup_epochs = 3
