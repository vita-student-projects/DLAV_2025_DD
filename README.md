# Deep Learning for Autonomous Vehicles 2025 (CIVIL-459)  
**Théo Houle**  
**Kelan Solomon**  

**Team: Drunk Drivers**

---
## Milestone 2: 
For this milestone, we have the depth and semantic masks as additional data to improve our model. 

### Model Architecture
- #### Image Encoder
Uses a ResNet50 backbone pretrained on ImageNet to extract visual features from camera input. The network removes the classification head and applies adaptive average pooling to obtain a fixed-size feature representation, followed by a fully-connected layer that projects features to 512 dimensions.
- #### History Encoder
Processes the vehicle's past trajectory through a 5-layer MLP architecture (63→512→256→256→128→128) with ReLU activations between layers. This gradually refines the temporal information into a compact 128-dimensional feature vector capturing motion patterns.
- #### Trajectory Decoder
Combines the visual (512-dim) and historical (128-dim) features through concatenation, then processes this 640-dimensional vector through a 3-layer MLP (640→512→256→180) to predict the vehicle's future trajectory for 60 timesteps, each represented by 3 values.

### Auxiliary Decoders
The model supports multi-task learning through two auxiliary decoders that operate on the raw CNN features (2048 channels) before pooling:

- #### Depth Decoder
Contains 5 upsampling blocks that progressively increase spatial resolution while reducing channel dimension (2048→512→256→128→64→32). Each block includes bilinear upsampling, convolution, batch normalisation, and ReLU activation. The final layer produces a single-channel depth map at 200×300 resolution.

- #### Semantic Decoder
It mirrors the architecture of the depth decoder but outputs 15 semantic classes at 200×300 resolution. This enables the model to distinguish between different elements in the driving scene (road, vehicles, pedestrians, etc.).



### Experiments
We tried many different improvements, with varying success. Namely:
* Transformer-based trajectory decoder
* Multi-modal trajectory prediction with a mixture of experts
* Spatial-temporal attention mechanism
* Feature enhancement module (local and global pathways)
* Velocity-aware trajectory prediction with physics-based integration
* Temporal dropout and enhanced regularisation techniques
* Modified learning rate schedules and optimisation algorithms
* Trajectory post-processing
* Using the ground truths as extra channels in the image as a sanity check
* Using a Cosine Schedular with warmup
  
Most of these improvements were only briefly explored, and if there were no immediate improvements, they were dropped.

## Usage Instructions

### Training

To train the model, run the `milestone2.py` script. It requires the dependencies listed in `requirements.txt`. You can view available parameters by running the script with `--help`.

This script generates:
- a CSV file for the submission
- The best model parameters,
- The last model parameters,
- The metrics for each epoch
- Plots of the loss

These are all saved in the folder `models/model_name` where `model_name` is the name of the model given as input to the script.

### Inference / CSV Generation

The `milestone2.py` script generates a CSV file ready to be submitted.

