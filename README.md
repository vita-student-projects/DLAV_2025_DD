# Deep Learning for Autonomous Vehicles 2025 (CIVIL-459)  
**Théo Houle**  
**Kelan Solomon**  

**Team: Drunk Drivers**

---
## Milestone 2: 
In the milestone, we have the depth and semantic masks as additional data to improve our model. 
## Model Architecture

## Experiments
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




## Milestone 1: Basic End-to-End Planner

We implemented an end-to-end model to predict future trajectories using the following inputs:
- RGB camera image  
- Driving command  
- Vehicle’s motion history (`sdc_history_feature`)  

### Model Architecture

The simplest approach turned out to be the most effective for our use case. We used ResNet-18 to encode the camera image, removing its final layer to extract features. The vehicle’s motion history was not encoded, but rather directly concatenated with the image features. We also added the most recent position as an ego state to represent instantaneous speed.

The combined feature vector was passed through a two-layer linear model, achieving an Average Displacement Error (ADE) of **1.70** in the Kaggle competition.

### Experiments

We explored more complex architectures, including a sequence-to-sequence model using recurrent neural networks. In this setup:
- Position history was encoded using an LSTM.  
- Image features were extracted via a CNN.  
- The LSTM’s final hidden state was concatenated with the image embedding.  
- The resulting vector was passed through a decoder LSTM to predict future trajectories.  

We also experimented with residual connections, adding the image context to various parts of the decoder (input, hidden state, cell state). Despite many variations, we were unable to achieve an ADE below 2. We suspect the model’s depth, combined with limited training data, led to underperformance.

### Alternative Strategy: "Winner-Takes-All"

We tried a "winner-takes-all" approach where the model outputs multiple potential trajectories and is trained using the one with the lowest error. This method did not improve performance, likely due to incorrect implementation, which we were unable to resolve in time.

### Hyperparameter Tuning

We performed a grid search over the following settings:
- **Batch size:** 16, 32, 64  
- **Learning rate:** 0.001, 0.01, 0.1  
- **Loss function:** MSE, Cross-Entropy, Binary Cross-Entropy  
- **History encoder:** None, LSTM, GRU  
- **Ego encoder:** None, two-layer linear model  

All results were saved locally. However, the grid search takes too long to complete, and we couldn’t obtain final results before the milestone deadline.

---

## Usage Instructions

### Training

To train the model, run the `milestone1.py` script. It requires the dependencies listed in `requirements.txt`. You can view available parameters by running the script with `--help`.

### Inference / CSV Generation

To generate a CSV file for the competition:
- Use the `generatecsv.py` script.  
- Specify the path to a trained `.pth` model file.  
- Parameter options are similar to those used during training.

Run with `--help` for detailed usage instructions.
