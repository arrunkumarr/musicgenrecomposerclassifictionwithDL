# musicgenrecomposerclassifictionwithDL
This initiative focuses on building a deep learning system capable of recognizing the composer of a musical piece. Leveraging a dataset of MIDI files from four iconic composers—Bach, Beethoven, Chopin, and Mozart—the approach utilizes both Long Short-Term Memory (LSTM) networks and Convolutional Neural Networks (CNNs) to classify composer 

[Input] 
     │
     └─> Expressive feature matrix (timesteps × 3 features: note density, velocity, duration)
          (Shape example: 5 timesteps × 3 features)
     │
     ▼
[Conv1D Layer 1]
     - 64 filters, kernel size=2, activation=ReLU
     - Input: expressive feature matrix
     - Output: feature maps capturing local patterns across timesteps and features
     │
     ▼
[Batch Normalization + MaxPooling]
     - Normalizes activations to stabilize training
     - MaxPooling reduces temporal dimension by factor 2 (downsampling)
     │
     ▼
[Conv1D Layer 2]
     - 32 filters, kernel size=2, activation=ReLU
     - Further extracts patterns at a finer level
     │
     ▼
[Batch Normalization + MaxPooling]
     - Normalizes and downsamples again
     │
     ▼
[LSTM Layer 1]
     - 64 units, returns sequences (output at each timestep)
     - Captures temporal dependencies in extracted CNN features
     - Dropout applied to reduce overfitting
     │
     ▼
[LSTM Layer 2]
     - 32 units, outputs final sequence representation
     │
     ▼
[Dense (Fully Connected) Layer]
     - 64 units, ReLU activation, Dropout
     │
     ▼
[Dense Layer]
     - 32 units, ReLU activation
     │
     ▼
[Output Layer]
     - Dense layer with softmax activation
     - 4 units (for 4 composer classes)
     - Provides class probabilities
     │
     ▼
[Prediction] Composer class with highest softmax probability

