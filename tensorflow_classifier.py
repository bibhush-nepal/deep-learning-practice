from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd

COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

path_train_dataset = tf.keras.utils.get_file("iris_training.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
path_test_dataset = tf.keras.utils.get_file("iris_test.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(path_train_dataset, names= COLUMN_NAMES, header= 0)
test = pd.read_csv(path_test_dataset, names = COLUMN_NAMES, header= 0)

y_train = train.pop('Species')
y_test = test.pop('Species')

def input_function(features, labels, training= True, batch_size= 256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

feature_columns_iris = []
for k in train.keys():
    feature_columns_iris.append(tf.feature_column.numeric_column(key=k))

clf = tf.estimator.DNNClassifier(feature_columns=feature_columns_iris, hidden_units=[30, 10], n_classes=3)
clf.train(lambda : input_function(train, y_train, training= True), steps=5000)

result = clf.evaluate(lambda : input_function(test, y_test, training=False))
print("Accuracy: ", result['accuracy']*100)

def input_pred(features, batch_size= 256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

FEATURES = []
for p in train.columns:
    FEATURES.append(p)

TEST_DATA = {}
for f in FEATURES:
    val = test.loc[15][f]
    TEST_DATA[f] = [float(val)]

predictions = clf.predict(input_fn= lambda: input_pred(TEST_DATA))
for p in predictions:
    cid = p['class_ids'][0]
    prob = p['probabilities'][cid]
    print('Prediction: ', SPECIES[cid],'Accuracy: ', prob*100)