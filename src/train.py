import dataset
import tensorflow_model

import tensorflow as tf
import numpy as np

import os
import gc
gc.collect()

# Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


if __name__ == "__main__":

    # Prepare input data
    print("Current directory path: ", os.getcwd())
    train_path = '../data/All_61326/train_61326/'
    test_path = '../data/All_61326/test_61326/'

    # Train path has the class structure
    classes = os.listdir(train_path)
    num_classes = len(classes)
    print("num of classes:", num_classes)

    #Keeping image size as 128
    img_size = 128
    num_channels = 3

    # We shall load all the training and validation images and labels into memory using openCV and use that during training
    train_data = dataset.read_train_sets(train_path, img_size, classes, validation_size=0.2)
    print("Complete reading input data. Will Now print a snippet of it")
    print("Number of files in Training-set:\t{}".format(len(train_data.train.labels)))
    print("Number of files in Validation-set:\t{}".format(len(train_data.valid.labels)))

    test_data = dataset.load_test(test_path, img_size)
    print("Number of files in test-set:\t{}".format(len(test_data)))

    batch_size = 8
    num_iterations = 3

    #session, y_pred_cls = tensorflow_model.load_tensorflow(train_data, batch_size, num_iterations, img_size)

    #x, y_true = tensorflow_model.input_placeholders(img_size, num_channels, num_classes)
    #y_true_cls = tf.argmax(y_true, dimension=1)

    #prediction = session.run(y_pred_cls, feed_dict={'x': test_data})
    #print(prediction)

    # save the model
    #saver = tf.train.Saver()
    #saver.save(session, '../model/meta_face_model')


    # Test
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('../model/meta_face_model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('../model/'))

        # Place holder for x
        #print([n.name for n in tf.get_default_graph().as_graph_def().node])

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")
        y_pred = graph.get_tensor_by_name("y_pred:0")

        print(x, y_true, y_pred)

        prediction = sess.run(y_pred, feed_dict={x: test_data[:]})
        print(prediction)

        y_pred_cls = [np.argmax(x) for x in prediction]
        #y_pred_cls = np.argmax(prediction)
        print(y_pred_cls)
