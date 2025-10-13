# %%
import tensorflow as tf 
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# %%
Batch_size = 32
Image_size = 256
Channels = 3
Epochs = 25

# %%
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle = True,
    image_size = (Image_size, Image_size),
    batch_size = Batch_size
)

# %%
class_names = dataset.class_names
class_names

# %%
for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())
    # 1 image batch has 32 size which is defined above.

# %%
plt.figure(figsize=(10,10))
for image_batch, labels_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[labels_batch[i]])
        plt.axis("off")

# %%
# 80% ==> training
# 20% ==> test
#     10% ==> validation
#     10% ==> test

# %%
len(dataset) # it give 68 because it divided into 32 batch size.

# %%
train_size = 0.8
len(dataset)*train_size

# %%
train_ds = dataset.take(54)
len(train_ds)

# %%
test_ds = dataset.skip(54)
len(test_ds)

# %%
val_size = 0.1
len(dataset)*val_size

# %%
val_ds = test_ds.take(6)
len(val_ds)

# %%
test_ds = test_ds.skip(6)
len(test_ds)

# %%
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split+val_split+test_split) == 1

    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split*ds_size)
    val_size = int(val_split*ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

# %%
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

# %%
len(dataset), len(train_ds), len(val_ds), len(test_ds)

# %%
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# %%
for image_batch, labels_batch in dataset.take(1):
    print(image_batch[0].numpy()/255)

# %%
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(Image_size, Image_size),
    layers.Rescaling(1.0/255)
])

# %%
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])

# %%
input_size = (Batch_size, Image_size, Image_size, Channels)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = input_size),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])
model.build(input_shape=input_size)

# %%
model.summary()

# %%
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = ['accuracy']
)

# %%
history = model.fit(
    train_ds,
    batch_size = Batch_size,
    validation_data = val_ds,
    verbose = 1,
    epochs = Epochs
)

# %%
scores = model.evaluate(test_ds)
scores

# %%
history.params

# %%
history.history.keys()

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# %%
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(Epochs), acc, label = 'Training Accuracy')
plt.plot(range(Epochs), val_acc, label = 'Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and validation Accuracy')

plt.subplot(1,2,2)
plt.plot(range(Epochs), loss, label = 'Training Loss')
plt.plot(range(Epochs), val_loss, label = 'Validation Loss')
plt.legend(loc='lower right')
plt.title('Training and validation Loss')

plt.show()

# %%
import numpy as np 
for image_batch, labels_batch in test_ds.take(1):
    first_image = image_batch[0].numpy().astype('uint8')
    print("First image to predict")
    plt.imshow(first_image)
    print("Actual Label:-", class_names[labels_batch[0].numpy()])

    batch_prediction = model.predict(image_batch)
    print("Predicted Label:-", class_names[np.argmax(batch_prediction[0])])

# %%
def predict(model, img):
    img_array = tf.keras.utils.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])), 2)

    return predicted_class, confidence

# %%
plt.figure(figsize=(15,15))
for images , labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        predicted_class, confidence = predict(model, images.numpy())

        actual_class = class_names[labels[i]]
        plt.title(f'Actual: {actual_class},\n Predicted: {predicted_class},\n Confidence: {confidence}%')
        plt.axis('off')

plt.show()

# %%
model_version = 1
# Save model in native Keras format for reloading or retraining later
model.save(f"./Models/{model_version}.keras")

# %%
# Export model in TensorFlow SavedModel format for deployment (e.g., FastAPI or TF Serving)
model.export(f"./Models/{model_version}_export")

# %%



