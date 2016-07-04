from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Import MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Define the regression model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define training pieces by defining the cost function and training algorithm
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Initialize the variables
init = tf.initialize_all_variables()

# Launch model in a session
sess = tf.Session()
sess.run(init)

# Train the model
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluate the model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def identify_digit(image):
  print(sess.run(tf.argmax(y,1), feed_dict={x: [image]})[0])

def print_image(image):
  string = ''
  for i in range(0, (28*28)):
    value = image[i]
    if value > 0:
      string += '1'
    else:
      string += '0'
    if i % 28 == 27:
      string += '\n'
  print(string)
