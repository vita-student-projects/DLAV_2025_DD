# Deep Learning for Autonomous Vehicles 2025 (CIVIL-459)  
**Théo Houle**  
**Kelan Solomon**  

**Team: Drunk Drivers**

---
## Milestone 2: 
In the milestone, we have the depth and semantic masks as additional data to improve our model. 

### Model Architecture

Our new model builds on the previous model from milestone 1. The main difference are the auxiliary depth and semantic map predictions. Let's first review the differences with the phase 1 model.

- **Image Encoder**: Our model for phase 1 uses a ResNet18. We wanted to switch to something bigger but it did not yield any significant improvement and made the training slower so we kept it for phase 2.

- **History Encoder**: Our model for phase 1 did not use an history encoder, it was only flattened. Here, we decided to  **TODO**

- **Ego Encoder**: Phase 1 model was using the last position of the vehicule as Ego position and was encoded using a simple two layers linear model. For phase 2, we decided **TODO**

- **Decoder**: Phase 1 model was using a simple 3-layered MLP. We decide that **TODO**

Let's now take a look and the depth and semantic decoders. 





### Experiments
We tried many different improvements, with varying success. Namely:
- Transformer-based trajectory decoder
- Multi-modal trajectory prediction with a mixture of experts
- Spatial-temporal attention mechanism
- Feature enhancement module (local and global pathways)
- Velocity-aware trajectory prediction with physics-based integration
- Temporal dropout and enhanced regularisation techniques
- Modified learning rate schedules and optimisation algorithms
- RobustTrajectoryPredictor with dynamic input handling
- Trajectory post processing
- Using the ground truths as extra channels in the image as a sanity check
- Using a Cosine Schedular with warmup


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

