import tensorflow as tf

session =  tf.Session()

hello = tf.constant("Hello World")

print(session.run(hello))

a = tf.constant(22)
b =tf.constant(33)

print('sum a+b ={0}'.format(session.run(a+b)))