from datasets import load_dataset
import json
import numpy as np
from datasets import Dataset

import matplotlib.pyplot as plt
import io
from PIL import Image

train_dataset = load_dataset("back2classroom/sidewalks", split='train', streaming=False)
val_dataset = load_dataset("back2classroom/sidewalks", split='val', streaming=False)

with open('valid_indices.json', 'r') as f:
    valid_indices = json.load(f)

# Filter out entries based on valid indices
filtered_train_dataset = train_dataset.select(valid_indices)

# Rename columns
filtered_train_dataset = filtered_train_dataset.rename_column('tif', 'image')
filtered_train_dataset = filtered_train_dataset.rename_column('label_tif', 'label')
filtered_train_dataset = filtered_train_dataset.select_columns(['image', 'label'])

print(len(filtered_train_dataset))

#Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]
  return bbox

from torch.utils.data import Dataset

class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    # image = item["image"]
    ground_truth_mask = np.array(item["label"])
    image = np.array(Image.open(io.BytesIO(item["image"])))
    ground_truth_mask = np.array(Image.open(io.BytesIO(item["label"])))

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs
  

with open("preprocessor_config.json", "r") as f:
    preprocessor_config = json.load(f)

# Initialize the processor
from transformers import SamProcessor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base", 
                                        **preprocessor_config)

# Create an instance of the SAMDataset
custom_train_dataset = SAMDataset(dataset=filtered_train_dataset, processor=processor)

# Create a DataLoader instance for the training dataset
from torch.utils.data import DataLoader
train_dataloader = DataLoader(custom_train_dataset, batch_size=2, shuffle=True, drop_last=False)

# Load the model
from transformers import SamModel
model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)


from torch.optim import Adam
import monai
# Initialize the optimizer and the loss function
optimizer = Adam(
    model.mask_decoder.parameters(),
    lr=1e-5,  # Initial learning rate
    weight_decay=0.01,
    betas=(0.9, 0.999),  # Momentum coefficients
    eps=1e-08,  # Term added to the denominator to improve numerical stability
    amsgrad=False  # Use the AMSGrad variant of the optimizer
)
#Try DiceFocalLoss, FocalLoss, DiceCELoss
seg_loss = monai.losses.FocalLoss(gamma=2.0, alpha=None, weight=None, reduction='mean', use_softmax=False)


from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
from monai.metrics import compute_iou

#Training loop
num_epochs = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    epoch_ious = []
    for batch in tqdm(train_dataloader):
      # forward pass
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)

      sam_masks_prob = torch.sigmoid(predicted_masks)
      sam_masks_prob = sam_masks_prob.squeeze()
      sam_masks = (sam_masks_prob > 0.5)

      loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
      ious = compute_iou(sam_masks.unsqueeze(1), ground_truth_masks.unsqueeze(1), ignore_empty=False)
      epoch_ious.append(ious.mean())
      print(f'iou: {ious.mean()}')
      print(f'loss: {loss}')
      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')
    print(f'Mean IOU: {mean(epoch_ious)}')

# Specify the filename
filename = "sam_model.pth"

# Save the model's state dictionary to the current directory
torch.save(model.state_dict(), filename)