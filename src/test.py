import tensorflow as tf
import dataset

saver = tf.train.Saver()
graph = tf.get_default_graph()
x_test, y_labels = dataset.load_test('../data/All_61326/test_61326/', image_size=128)

img_size = 128
num_channels = 3
num_classes = 6

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')


with tf.Session(graph=graph) as session:

    ckpt = tf.train.get_checkpoint_state('C:\\Users\\amehta02\\PycharmProjects\\practice_keras\\model\\checkpoint')
    saver.restore(session, ckpt)

    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    # y_test_images = np.zeros((1, 2))

    feed_dict_testing = {x: x_test, y_true: y_labels}
    result = session.run(y_pred, feed_dict=feed_dict_testing)

    print(result)