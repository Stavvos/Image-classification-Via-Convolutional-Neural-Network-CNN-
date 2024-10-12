#the reference used for the codebase is: https://www.youtube.com/watch?v=u3FLVbNn9Os 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Input
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def loadData():
    print("\n\nLoading, splitting, and reshaping the dataset")
    #load the dataset
    dataset = tf.keras.datasets.mnist

    #split the dataset into 80/20 train/test
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    print("\n\nThe shape of the training data before the split is: ", x_train.shape, " and labels: ", y_train.shape)
    print("\n\nThe shape of the testing data before the split is: ", x_test.shape, " and labels: ", y_test.shape)
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 42)

    print("\n\nThe shape of the training data after split is now: ", x_train.shape, " and labels: ", y_train.shape)
    print("\n\nThe shape of the testing data after split is now: ", x_test.shape, " and labels: ", y_test.shape)

    #show the first image from x_train
    plt.imshow(x_train[0])
    plt.title(f"The label for the first traning image is: {y_train[0]}")
    plt.show()

    #show the first image from x_train as a greyscale image
    plt.imshow(x_train[0], cmap = plt.cm.binary)
    plt.title(f"The label for the first training image as greyscale is: {y_train[0]}")
    plt.show()

    #normalise the data. you could do this by dividing the values by 255. 
    x_train = tf.keras.utils.normalize(x_train, axis = 1)
    x_test = tf.keras.utils.normalize(x_test, axis = 1) 

    #inspect what the first image from x_train looks like now after normalization
    plt.imshow(x_train[0], cmap = plt.cm.binary)
    plt.title(f"The label for the first training image after normalisation is: {y_train[0]}")
    plt.show()

    #reshape the dataset
    imageSize = 28
    x_train = np.array(x_train).reshape(-1, imageSize, imageSize, 1)
    x_test = np.array(x_test).reshape(-1, imageSize, imageSize, 1)
    print("\n\nTraining samples dimensions after reshaping:", x_train.shape)
    print("\n\nTesting samples dimensions after reshaping:", x_test.shape)

    return x_train, y_train, x_test, y_test

def augmentData(x_train, y_train, x_test, y_test):
    print("\n\nAugmenting data")
    # Create an ImageDataGenerator object with augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=10,       # Rotate the image by up to 10 degrees
        width_shift_range=0.1,   # Shift the image horizontally by 10%
        height_shift_range=0.1,  # Shift the image vertically by 10%
        zoom_range=0.1,          # Zoom in by up to 10%
        shear_range=0.1,         # Shear the image by 10%
        horizontal_flip=False,   # No horizontal flip for digit images (since MNIST digits are directional)
        fill_mode='nearest'      # Filling in pixels with nearest pixel value
    )

    # Fit the generator to the training data 
    datagen.fit(x_train)
    datagen.fit(x_test)

    #display the first image from x_train and its label
    plt.imshow(x_train[0])
    plt.title(f"The label for the first traning image after augmentation is: {y_train[0]}")
    plt.show()

    #display the first image from x_test and its label
    plt.imshow(x_test[0])
    plt.title(f"The label for the first testing image after augmentation is: {y_test[0]}")
    plt.show()

    print("\n\nThe shape of the training data after augmentation: ", x_train.shape, " and the labels: ", y_train.shape)
    print("\n\nThe shape of the testing data after augmentation: ", x_test.shape, "and the labels: ", y_test.shape, "\n\n")

    dataAugmented = True

    return x_train, x_test

def create_model(x_train):
    #initalise the Convolutional Neural Network
    model = Sequential()

    #first convolutional layer
    model.add(Input(shape = x_train.shape[1:])) #the input layer. make sure the shape of the data is compatable with this layer
    model.add(Conv2D(64, (3,3))) #convolutional layer
    model.add(Activation("relu")) #the activation function
    model.add(MaxPooling2D(pool_size = (2,2)))# max pooling layer

    ##second convolutional layer
    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))

    ##third convolutional layer
    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))

    #fully connected layer 1
    model.add (Flatten()) #need to flatten the data once we're at the fully connected layer. from 2D to 1D
    model.add (Dense(64))
    model.add(Activation("relu"))

    #fully connected layer 2
    model.add(Dense(32))
    model.add(Activation("relu"))
    
    #the final fully connected layer
    model.add(Dense(10)) #this is equal to 10 because we have 10 data classes
    model.add(Activation('softmax'))

    #compile the model and set its loss, optimizer, and metrics
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'])
    return model

def trainModel(x_train, y_train, x_test, y_test):
    #train the model
    print("\n\nTraining the model: ")
    model = create_model(x_train)
    model.fit(x_train, y_train, epochs = 1, validation_split = 0.3)

    #make some predictions and convert from a probability to an actual class label
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis =1)

    #perform cross validation on the model
    print("\n\nPerforming cross validation: ")
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    crossValModel = KerasClassifier(build_fn = create_model(x_train), epochs = 1, validation_split = 0.3)
    crossValScores = cross_val_score(crossValModel, x, y, cv = 5)

    #print the results of the cross validation
    print("\nThe cross validation scores are: ", crossValScores)
    print("The mean accuracy of the cross validation scores is:", crossValScores.mean(), " with a standard deviation of: ", crossValScores.std())

    #print the predictions to the console to check if they are infact the labels 0-9
    print("\nThe predictions after being converted from probabilities to labels are as follows:\n", predictions)

    #make a classification report and print it to the console
    report = classification_report(y_test, predictions, target_names = [str(i) for i in range(10)])
    print("\nThe classification report for the convolutional neural network model is:\n", report)

    #make a confusion matrix
    confusionMatrix = confusion_matrix(y_test, predictions)

    #make a simple performance report
    print("\nA performance report for the CNN model classifying hand written digits:")
    print("Accuracy:",accuracy_score(y_test, predictions))
    print("Precision:",precision_score(y_test, predictions, average = 'weighted'))
    print("Recall:",recall_score(y_test, predictions, average = 'weighted'))
    print("F1-score:",f1_score(y_test, predictions, average = 'weighted'))
    print("\nconfusion matrix:\n", confusionMatrix)

    print("\n\nThe architecture of the CNN model is:\n")
    print(model.summary())

def main():
    x_train, y_train, x_test, y_test = loadData()
    x_train, x_test = augmentData(x_train, y_train, x_test, y_test)
    trainModel(x_train, y_train, x_test, y_test)

main()
