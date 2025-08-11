# Composer Classification Using CNN-LSTM on Expressive Music Features

## Abstract
This project explores the use of deep learning techniques to classify classical music composers based on expressive performance features extracted from MIDI data. A hybrid CNN-LSTM architecture is employed to capture both local temporal motifs and long-range sequential phrasing. The model is trained on a curated dataset of expressive features—note density, velocity, and duration—across multiple composers. The pipeline emphasizes fairness, interpretability, and reproducibility, with modular preprocessing, stratified sampling, and performance evaluation using accuracy, precision, and recall.

## Keywords
Deep Learning, Composer Classification, CNN-LSTM, Expressive Features, MIDI, Music Information Retrieval, Neural Networks, Feature Engineering, Time Series, Keras, TensorFlow

---

##  Introduction Deep learning has revolutionized pattern recognition across domains, including music information retrieval (MIR). Neural networks, particularly convolutional and recurrent architectures, excel at learning hierarchical and temporal representations. In this project, we apply a CNN-LSTM hybrid model to classify composers based on expressive performance features derived from MIDI files. Unlike symbolic pitch-based models, our approach focuses on dynamic and temporal expressivity—capturing stylistic nuances unique to each composer.

---

## Dataset
The dataset consists of MIDI-derived expressive features for multiple classical composers. Each sample represents a 5-time-step segment with 3 features:

- **Note Density**: Number of notes per time slice  
- **Velocity**: Average note velocity (dynamics)  
- **Duration**: Mean note duration  

Each sample is labeled with its corresponding composer. The dataset is imbalanced, with dominant representation from composers like Bach, requiring careful sampling and weighting.

---

## Methodology & Preprocessing

### Label Encoding
```python
composer_to_label = {name: idx for idx, name in enumerate(target_composers)}
label_encoder = LabelEncoder().fit(y_train_composer_labels)
```

### Feature Scaling
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 3)).reshape(-1, 5, 3)
```

### Reshaping
```python
X_train_reshaped = X_train_scaled
X_test_reshaped = X_test.reshape(-1, 5, 3)
```

### Class Weighting
```python
from sklearn.utils.class_weight import compute_class_weight
class_weight_dict = dict(zip(np.unique(y_train), compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
```

---

## Feature Extraction
Expressive features were extracted from MIDI files using custom parsing logic. Each file was segmented into 5 time slices, and the following features were computed:

- **Note Density**: Count of notes per slice  
- **Velocity**: Mean velocity per slice  
- **Duration**: Mean duration per slice  

These features were chosen for their stylistic relevance and interpretability.

---

##  Model Architecture
```python
def build_cnn_lstm_model(timesteps, feature_dim, num_classes):
    input_layer = layers.Input(shape=(timesteps, feature_dim))

    x = layers.Conv1D(64, kernel_size=2, activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(32, kernel_size=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=input_layer, outputs=output)
```

---

##  Training Process
```python
model = build_cnn_lstm_model(timesteps=5, feature_dim=3, num_classes=len(np.unique(y_train)))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train_reshaped, y_train,
    validation_data=(X_test_reshaped, y_test),
    epochs=40,
    batch_size=32,
    class_weight=class_weight_dict,
    verbose=2
)
```

---

##  Evaluation Metrics
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

y_pred = np.argmax(model.predict(X_test_reshaped), axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
```

---

##  References & Libraries

- [TensorFlow & Keras](https://www.tensorflow.org)  
- [Scikit-learn](https://scikit-learn.org)  
- [Pandas](https://pandas.pydata.org)  
- [NumPy](https://numpy.org)  
- MIDI parsing libraries: `mido`, `pretty_midi`  
- [Gradio](https://www.gradio.app)  
- [Hugging Face Spaces](https://huggingface.co/spaces)

---

##  Conclusion
This project demonstrates the effectiveness of CNN-LSTM architectures in capturing expressive musical traits for composer classification. By focusing on note density, velocity, and duration, the model learns stylistic signatures that generalize across time slices. While the baseline model performed better on raw piano roll data, the expressive-feature model offers interpretability and modularity. Future work could explore:

- Attention mechanisms for time-step weighting  
- Data augmentation via expressive perturbation  
- Integration with symbolic pitch features  
- Deployment via Hugging Face or web UI for interactive testing
```

