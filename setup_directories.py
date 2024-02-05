import os

# Define the base directory for the data
base_dir = "data"

# Define the subdirectories to create within the base directory
sub_dirs = ["ml-1m", "pisa2015"]

# Create the base directory if it doesn't already exist
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Create each subdirectory within the base directory
for sub_dir in sub_dirs:
    path = os.path.join(base_dir, sub_dir)
    if not os.path.exists(path):
        os.makedirs(path)

print("Directories created successfully.")
