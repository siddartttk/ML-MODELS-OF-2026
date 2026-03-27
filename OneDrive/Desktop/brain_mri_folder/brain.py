import torch

# Load the model file
model_path = 'best_brain_mri_model/data.pkl' # Or the exact .pth filename
data = torch.load(model_path, map_location=torch.device('cpu'))

# Check what type of data it is
print(f"Data type: {type(data)}")

# If it's a state_dict (common), print the layer names
if isinstance(data, dict):
    for key in data.keys():
        print(key)
else:
    print(data)