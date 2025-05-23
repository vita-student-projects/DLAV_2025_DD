# Deep Learning for Autonomous Vehicles 2025 (CIVIL-459)  
**Théo Houle**  
**Kelan Solomon**  

**Team: Drunk Drivers**

---
## Milestone 3 
For this milestone, we are predicting on real-world images. Since we do not have depth or semantic masks, we reused our Milestone 1 code. Just with the initial code, we achieved the milestone goal. In order to further improve our prediction, we upgraded the Resnet to a Resnet50, and flipped the images and paths to double our dataset size. We also tried different optimisers (SGD and Adam) and schedulers (cosine and reduce lr on plateau). We found that the best combination was Adam with a cosine scheduler at 200 epochs, achieving 1.532 ADE on the validation set. We also tried other augmentations but didn't see any significant improvement. 

## Usage Instructions

### Training

To train the model, run the `milestone3.py` script. It requires the dependencies listed in `requirements.txt`. You can view available parameters by running the script with `--help`.

This script generates:
- a CSV file for the submission
- The best model parameters,
- The last model parameters,
- The metrics for each epoch
- Plots of the loss

These are all saved in the folder `models/model_name` where `model_name` is the name of the model given as input to the script.

### Inference / CSV Generation

The `milestone3.py` script generates a CSV file ready to be submitted.

