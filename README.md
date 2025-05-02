# Deep Learning for autonomous vehiculs 2025 (CIVIL-459)
Theo Houle

Kelan Solomon

Team: Drunk Drivers
## Milestone 1: Basic End-to-End Planner
Implement an end-to-end model that predicts future trajectories based on:
- Camera RGB image
- Driving command
- Vehicle’s motion history (sdc_history_feature)
### Structure
The simplest model turned out to be the most effective for our use case. Image features were encoded using ResNet-18 and concatenated with the raw position history. This combined data was then passed through two fully connected layers.

We also experimented with using seq2seq with recurrent neural networks to encode the position data. Specifically, we passed the positions through an LSTM, encoded the image using a CNN, and concatenated the resulting image embedding with the final hidden state of the LSTM. This representation was then decoded using another LSTM. Additionally, we tried incorporating residual connections between the image features and the concatenated data.Many different variations of this were tried, adding the context (image) to cell state, the hidden state and the input, however, we were unable to achieve an ADE below 2 with this method. We believe the model may have been too deep for the available dataset, leading to underperformance due to insufficient training data. 

### How to run/train/infer
