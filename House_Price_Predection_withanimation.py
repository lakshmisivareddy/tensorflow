import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

num_house=160
np.random.seed(42)
house_size= np.random.randint(low=1000,high=3500,size=num_house)

house_price = house_size * 100.0+ np.random.randint(low=20000,high=70000,size=num_house)

plt.plot(house_size,house_price,"bx")
plt.xlabel("size")
plt.ylabel("price")
plt.show()

#normalize values to prevent overflow ad underfow 
def normalize(array):
    return (array - array.mean())/array.std()
#defne tranning set 
num_train_samples=math.floor(num_house * 0.7)

#define trannig data
train_house_size    =   np.asarray(house_size[:num_train_samples])
train_price   =   np.asarray(house_price[:num_train_samples:])

train_house_size_norm=normalize(train_house_size)
train_price_norm=   normalize(train_price)

#define test data
test_house_size = np.asarray(house_size[num_train_samples:])
test_price = np.asarray(house_price[num_train_samples:])
test_house_size_norm  = normalize(test_house_size)
test_price_norm = normalize(test_price)

#tensorflow place holders 
tf_house_size = tf.placeholder("float",name="house_size")
tf_price = tf.placeholder("float",name="price")

#tensor variables
tf_size_factor = tf.Variable(np.random.randn(),name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(),name="price_offset")

#tensorflow inference function 
tf_price_pred = tf.add(tf.multiply(tf_size_factor,tf_house_size),tf_price_offset)

#loss function 
tf_cost  =  tf.reduce_sum(tf.pow(tf_price_pred-tf_price,2))/(2*num_train_samples)

#learning rate  
learning_rate =0.1

#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

 #inti
init = tf.global_variables_initializer()
# Launch the graph in the session
with tf.Session() as sess:
    sess.run(init) 

    #display
    display_every=2
    num_tranning_iter=50

    #number of plots
    fit_num_plots = math.floor(num_tranning_iter/display_every)
    fit_size_factor = np.zeros(fit_num_plots)
    fit_price_offsets = np.zeros(fit_num_plots)
    fit_plot_idx=0

    #iterating tranning data 
    for iteration in range(num_tranning_iter):
        #fill all tranning data 
        for(x,y) in zip(train_house_size_norm,train_price_norm):
            sess.run(optimizer,feed_dict={tf_house_size:x,tf_price:y})
        #display currect cost 
        if(iteration+1) % display_every==0:
            c= sess.run(tf_cost,feed_dict={tf_house_size:train_house_size_norm,tf_price:train_price_norm})
            print("itertion # :",'%04d'%(iteration+1),"cost=","{:.9f}".format(c),\
                "size_factor=",sess.run(tf_size_factor),"price_offect=",sess.run(tf_price_offset))
                #fit factor  szie  and price ofset 
            fit_size_factor[fit_plot_idx] = sess.run(tf_size_factor)
            fit_price_offsets[fit_plot_idx] = sess.run(tf_price_offset)
            fit_plot_idx = fit_plot_idx +1
            

    print("optimization finished")
    tranning_cost=sess.run(tf_cost,feed_dict={tf_house_size:train_house_size_norm,tf_price:train_price_norm})
    print("Trained cost=",tranning_cost,"size_factor=",sess.run(tf_size_factor),"price_offect",sess.run(tf_price_offset),'\n')
    
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()
    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    #ploat graph 
    plt.rcParams["figure.figsize"]=(10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size,train_price,'go',label="Tranning data")
    plt.plot(test_house_size,test_price,'mo',label="Testing Data")
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean, (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
             label='Learned Regression')
    plt.legend(loc='upper left')
    plt.show()

    fig, ax = plt.subplots()
    line, = ax.plot(house_size, house_price)
    plt.rcParams["figure.figsize"]=(10,8)
    plt.title("Gradient Decent fitting Regression Line")
    plt.xlabel("Size (sq.ft)")
    plt.ylabel("Price")
    plt.plot(train_house_size,train_price,'go',label="Tranning data")
    plt.plot(test_house_size,test_price,'mo',label="Testing Data")
    
    def animate(i):
        line.set_xdata(train_house_size_norm * train_house_size_std + train_house_size_mean) 
        line.set_ydata((fit_size_factor[i] * train_house_size_norm + fit_price_offsets[i]) * train_price_std + train_price_mean)  # update the data
        return line,
 
     # Init only required for blitting to give a clean slate.
    def initAnim():
        line.set_ydata(np.zeros(shape=house_price.shape[0])) # set y's to 0
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, fit_plot_idx), init_func=initAnim,
                                 interval=1000, blit=True)

    plt.show() 