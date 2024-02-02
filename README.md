### Objective
The project aims to detect various human emotions like "angry", "happy", and "sad" based on image data. It utilizes custom convolutional neural networks as a baseline model and incorporates the modern CNNs like ResNet and EfficientNet while also exploring the Transformer models in Vision to get the optimal results.

### Dataset

The dataset is obtained from Kaggle - https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes
It has train and test datasets seperated both containing 3 classes - 'angry', 'happy', and 'sad'

Train data has about 6799 files and the train data has 2278 files.

![image](https://github.com/RohitMacherla3/human-emotion-detection/assets/89356811/85a4b73b-efee-46c4-9e46-477a2efd6298)

### Preprocessing

- Initially the data was converted into tensorflow dataset to be able to work with the neural networks making the data into batches, converting the target variable into categorical values, shuffling the data to remove any data collection bias.
- Data Augmentation as performed using random rotation, random flip and random contarst layers of keras to remove location invariance problem.
- Images were rescaled and resized to standard (224,224, 3) sizes.

### Models Used
1. Baseline Convolutional Neural Network
2. ResNet50
3. EfficientNetB4
4. Vision Transformer

### Training and Optimization

All the model were trained for 30 epochs with an initial learning rate of 0.01 (5e-5 for Vision Transformer). Adam was used as the optimizer to train to minimize the categorical cross entropy loss.
Below are the configuration:

CONFIGURATIONS = {
    'BATCH_SIZE':32,
    'IM_SIZE': 256,
    'LEARNING_RATE': 0.001,
    'N_EPOCH': 30,
    'DROPOUT_RATE':0.0,
    'REGULARIZTION_RATE': 0.0,
    'N_FILTERS':6,
    'KERNEL_SIZE':3,
    'N_STRIDES':1,
    'POOL_SIZE':2,
    'N_DENSE_1': 100,
    'N_DENSE_2': 10,
    'NUM_CLASSES':3
}

Tensorflow callbacks were used for logging to later visualize on a tensorboard. Early stopping, reduce on plateau were implemeted to stop the training when there is no further improvement for certain number of epochs.

### Metrics
<img width="511" alt="image" src="https://github.com/RohitMacherla3/human-emotion-detection/assets/89356811/ed76845f-9e9f-4a9e-8564-5a130701311e">

