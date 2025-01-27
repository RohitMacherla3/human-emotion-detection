### Objective
The project aims to detect various human emotions like "angry", "happy", and "sad" based on image data. It utilizes custom convolutional neural networks as a baseline model and incorporates modern CNNs like ResNet and EfficientNet while also exploring the Transformer models in Vision to get the optimal results.

### Deployment

Streamlit Web Application - https://human-emotion-detection.streamlit.app/

<img width="365" alt="image" src="https://github.com/RohitMacherla3/human-emotion-detection-CV/assets/89356811/d0bb916e-545a-4dd1-baf7-6ec22e1817c9">



### Dataset

The dataset is obtained from Kaggle - https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes
It has train and test datasets separated both containing 3 classes - 'angry', 'happy', and 'sad'

Train data has about 6799 files and the train data has 2278 files.

<img width="948" alt="image" src="https://github.com/RohitMacherla3/human-emotion-detection-CV/assets/89356811/59000b56-f0eb-473a-8ed3-a10005a181e5">


### Preprocessing

- Initially, the data was converted into a Tensorflow dataset to be able to work with the neural networks making the data into batches, converting the target variable into categorical values, and shuffling the data to remove any data collection bias.
- Data Augmentation was performed using random rotation, random flip, and random contrast layers of keras to remove location invariance problem.
- Images were rescaled and resized to standard (224,224, 3) sizes.

### Models Used
1. Baseline Convolutional Neural Network
2. ResNet50
3. EfficientNetB4
4. Vision Transformer

### Training and Optimization

All the models were trained for 30 epochs with an initial learning rate of 0.01 (5e-5 for Vision Transformer). Adam was used as the optimizer to train to minimize the categorical cross-entropy loss.
Below are the configurations:
1. BATCH_SIZE:32
2. IM_SIZE: 256
3. LEARNING_RATE: 0.001
4. N_EPOCH: 30
5. N_FILTERS:6
6. KERNEL_SIZE:3
7. N_STRIDES:1
8. POOL_SIZE:2
9. NUM_CLASSES:3

Tensorflow callbacks were used for logging to later visualize on a Tensorboard. 'Early Stopping' and 'Reduce on Plateau' were implemented to stop the training when there was no further improvement for a certain number of epochs.

### Metrics
<img width="616" alt="image" src="https://github.com/RohitMacherla3/human-emotion-detection-CV/assets/89356811/08b7dbeb-fb23-4833-ab90-58891aa8804a">



Model Comparision Plot

<img width="771" alt="image" src="https://github.com/RohitMacherla3/human-emotion-detection-CV/assets/89356811/6e483a42-57d9-4ecd-af0b-7b4466bb9b96">


### Predictions

<img width="948" alt="image" src="https://github.com/RohitMacherla3/human-emotion-detection-CV/assets/89356811/97ac5d99-379e-4456-af09-8904e49d4c7b">


### Conclusion
As one would assume, the transformer model outperforms the CNN models with just training for 10 epochs, and among the CNN models, EfficientNetB4 does a good job compared to ResNet50 with much lesser model complexity.



