import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev = 0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape = shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides= [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

#first layer (convolution and pooling)
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,28,28,1])
l_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
l_pool1 = max_pool_2x2(l_conv1)

#second layer (convolution and pooling)
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
l_conv2 = tf.nn.relu(conv2d(l_pool1, W_conv2) + b_conv2)
l_pool2 = max_pool_2x2(l_conv2)

#third layer (fully connected)
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
l_pool2_flat = tf.reshape(l_pool2, [-1, 7*7*64])
l_fc1 = tf.nn.relu(tf.matmul(l_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32)
l_drop = tf.nn.dropout(l_fc1, keep_prob)

#readout layer (fully connected)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
output = tf.matmul(l_drop, W_fc2)+b_fc2

#calculate accuracy and define the train step
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#define the session and start things up
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0: 
        train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y: batch[1], keep_prob : 1.0}, session = sess)
        print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict = {x: batch[0], y: batch[1], keep_prob : 0.5}, session = sess)

print('test accuracy %g' % accuracy.eval(feed_dict = {x: mnist.test.images,y: mnist.test.labels, keep_prob : 1.0}, session = sess))

