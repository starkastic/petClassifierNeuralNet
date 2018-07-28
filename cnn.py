from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

#build CNN

#initialise CNN
classifier = Sequential()

#step 1 convolution
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

#step 2 pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#another convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#step 3 flatten
classifier.add(Flatten())

#step 4 full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

#compile the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fit CNN
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_Set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')    

classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=5,
                        validation_data=test_set,
                        validation_steps=2000)