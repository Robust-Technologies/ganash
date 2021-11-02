""" Install Packages """
!pip3 install imageio
!pip3 install matplotlib
!pip3 install tensorflow==1.14.0
!pip3 install tensorpack
!pip3 install pandas
!pip3 install reedsolo==0.3
!pip3 install gast==0.2.2 


""" Import modules """
import os, sys
import re
import zlib
import time
import string
import random
import pathlib
import imageio
import functools
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from tqdm import tqdm
from reedsolo import RSCodec
from google.colab import drive
from collections import Counter
from warnings import filterwarnings
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope


""" Fixtures """
# Flush and mount drive
drive.flush_and_unmount()
drive.mount("/content/drive")

# Change dir
os.chdir("drive/My Drive/GANASH/Stegno-dev")
print(os.getcwd())
!ls

# RS Codec
rs = RSCodec(250)

# Filter warnings
filterwarnings("ignore")

# TF fixtures
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.reset_default_graph()

# Path and chennels
channels=3
image_path="data/div2k/val/_/"
dataroot=pathlib.Path(image_path)
all_=list(dataroot.glob('*'))
all_image_paths=[str(path) for path in all_]


""" Model Class """
class Model:
    def __init__(self,hiddendim,Batch_size,data_depth):
        self.hiddendim=hiddendim
        self.batch=Batch_size
        self.depth=data_depth
    def layer(self,x,filters):
        return tf.layers.conv2d(inputs=x,filters=filters,kernel_size=3,activation=tf.nn.leaky_relu,padding="same")
    @reuse_variable_scope
    def critic_network(self,x):
        with tf.variable_scope("critic" ):
            filters = self.hiddendim
            a1=tf.layers.conv2d(inputs=x,filters=filters,kernel_size=3,activation=tf.nn.leaky_relu,padding="same")
            b1=tf.layers.batch_normalization(a1,axis=-1)

            a2 = tf.layers.conv2d(inputs=b1,filters=filters,kernel_size=3,activation=tf.nn.leaky_relu,padding="same") 
            b2=tf.layers.batch_normalization(a2,axis=-1)

            a3 = tf.layers.conv2d(inputs=b2,filters=filters,kernel_size=3,activation=tf.nn.leaky_relu,padding="same")
            b3=tf.layers.batch_normalization(a3,axis=-1)

            final=tf.layers.conv2d(inputs=b3,filters=1,kernel_size=3)

            x=tf.reduce_mean(tf.reshape(final,[tf.shape(final)[0],-1]),-1)
            
        return x
    
    
    def encoder_network(self,image,data):
        #data=tf.cast(data,tf.float32)
        with tf.variable_scope("encoder", reuse =tf.AUTO_REUSE):
            filters = self.hiddendim
            s1=tf.layers.conv2d(inputs = image,filters=filters,kernel_size=3,padding="same")
            bb1=tf.layers.batch_normalization(s1,axis=-1)
            lbb1 = tf.nn.leaky_relu(bb1)
            # print(data.shape)
            inter=tf.concat([lbb1,data],axis=3)
            
            s2=tf.layers.conv2d(inputs= inter,filters=filters,kernel_size=3,padding="same")
            b2=tf.layers.batch_normalization(s2,axis=-1)
            lb2 = tf.nn.leaky_relu(b2)
            
            
            s21=tf.layers.conv2d(inputs= lb2,filters=filters,kernel_size=3,padding="same")
            b21=tf.layers.batch_normalization(s21,axis=-1)
            lb21 = tf.nn.leaky_relu(b21)
            
            s3=tf.layers.conv2d(inputs= lb21,filters=filters,kernel_size=3,padding="same")
            b3=tf.layers.batch_normalization(s3,axis=-1)
            lb3 = tf.nn.leaky_relu(b3)
            
            s4=tf.layers.conv2d(inputs=lb3,filters=3,kernel_size=3,activation=tf.tanh,padding="same")
            
        return s4
    def decoder_network(self,stegno):
        with tf.variable_scope("decoder",  reuse =tf.AUTO_REUSE):
            filters = self.hiddendim
            d1= tf.layers.conv2d(inputs = stegno,filters=filters,kernel_size=3,padding="same")
            db1=tf.layers.batch_normalization(d1,axis=-1)
            ldb1 = tf.nn.leaky_relu(db1)
#             #inter=tf.concat([db1,data],axis=3)
            d2=tf.layers.conv2d(inputs = db1,filters=filters,kernel_size=3,padding="same")
            db2=tf.layers.batch_normalization(d2,axis=-1)
            ldb2=  tf.nn.leaky_relu(db2)

            d21=tf.layers.conv2d(inputs= ldb2,filters=filters,kernel_size=3,padding="same")
            db21=tf.layers.batch_normalization(d21,axis=-1)
            ld21 = tf.nn.leaky_relu(db21)

            d3=tf.layers.conv2d(inputs = ldb1,filters=filters,kernel_size=3,padding="same")
            db3=tf.layers.batch_normalization(d3,axis=-1)
            ldb3 = tf.nn.leaky_relu(db3)

            d4=tf.layers.conv2d(inputs=ldb3,filters=self.depth,kernel_size=3,padding="same")
#             print(tf.gradients(d4,d1))
        return d4
    
    def inference(self,cover,y_true, quantize=False):
        N=self.batch
#         y_true = tf.random_uniform([10,10],0,2,dtype=tf.float32)
#         if not y_true:
#         y_true = tf.cast(tf.random_uniform([N,np.int(cover.shape[1]),np.int(cover.shape[2]),1],0,2,dtype=tf.int64),dtype = tf.float32)
          
#         else:
        y_true  = y_true
        # print("Tell me",cover.shape)
        # print("Tell me bits",y_true.shape)

        stegno=self.encoder_network(cover,y_true)
        
        if quantize:
            stegno=(255.0 * (stegno +1.0 )/2.0)
            stegno=2.0*stegno/255.0 -1.0
            
        y_pred=self.decoder_network(stegno)

        return cover,y_true,stegno,y_pred
    
    
    def build_losses(self,cover,y_true,stegno,y_pred):
        
        enc_mse=tf.losses.mean_squared_error(predictions = stegno,labels = cover)
        dec_loss= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_pred,labels = y_true))
#         dec_loss= tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_pred = y_pred,y_true = y_true))
        dec_acc=tf.reduce_sum(tf.to_int32(tf.math.equal(y_pred>=0,y_true>=0.5)))/tf.size(y_true)
        cover_score=tf.reduce_mean(self.critic_network(cover))
        stegno_score=tf.reduce_mean(self.critic_network(stegno))
        #print(stegno_score)
        
        return enc_mse,dec_loss,dec_acc,cover_score,stegno_score


""" Pied piper class """
class pied_piper():
    def __init__(self,image_paths,BATCH_SIZE,coworkers,buffer,channel=3,cropsize=(360,360)):
        self.path=image_paths
        self.channel=channel
        self.size=cropsize
        self.batch=BATCH_SIZE
        self.coworkers=coworkers
        self.buffer=buffer


    def _parse(self,image_path):
        image=tf.read_file(image_path)
        return tf.io.decode_png(image,channels=self.channel)


    def horizontal_flip(self,image):
        return tf.image.random_flip_left_right(image)

    def resize(self,image):
        image=tf.image.resize(images = tf.expand_dims(image,axis=0),size = self.size)
#         image = tf.image.resize_nearest_neighbor(images = tf.expand_dims(image,axis=0),size = self.size)
#         image = tf.image.crop_to_bounding_box(tf.expand_dims(image,axis=0),10,10,10,10)
#         image = tf.image.resize_image_with_crop_or_pad(tf.expand_dims(image,axis=0),self.size[0],self.size[1])
        image = tf.cast(image,dtype=tf.float32)
        image=tf.squeeze(image,axis=0)
        return image

    def norm(self,image):
        image = tf.cast(image,dtype=tf.float32)
        image=image/255.0
        return image

    def pipeline(self,shuffle=None):
        all_image_paths=tf.convert_to_tensor(self.path)
        data=tf.data.Dataset.from_tensor_slices(all_image_paths)

        data=data.map(self._parse,num_parallel_calls=self.coworkers)
        data=data.map(self.horizontal_flip,num_parallel_calls=self.coworkers)
        data=data.map(self.resize,num_parallel_calls=self.coworkers)
        data=data.map(self.norm,num_parallel_calls=self.coworkers)
        if shuffle:
            data=data.shuffle(self.buffer)
        
        data=data.batch(self.batch)
        
        #data=data.repeat(32)
        iterator=tf.data.Iterator.from_structure(data.output_types,data.output_shapes)

#         next_element=iterator.get_next()

        init_op=iterator.make_initializer(data)

        return iterator,init_op

""" Utility Functions """
# Message to bits related fns
def random_char(y):
	return ''.join(random.choice(string.ascii_letters) for x in range(y))

def text_to_bits(text):
    # Convert text to a list of ints in {0, 1}
    return bytearray_to_bits(text_to_bytearray(text))

def bits_to_text(bits):
    # Convert a list of ints in {0, 1} to text
    return bytearray_to_text(bits_to_bytearray(bits))

def bytearray_to_bits(x):
    # Convert bytearray to a list of bits
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def bits_to_bytearray(bits):
    # Convert a list of bits to a bytearray
    ints = []
    for b in range(len(bits) // 8):
#         print(b)
        byte = bits[b * 8:(b + 1) * 8]
        ints.append(int(''.join([str(bit) for bit in byte]), 2))
#     print(ints)
    return bytearray(ints)

def text_to_bytearray(text):
    # Compress and add error correction
    assert isinstance(text, str), "expected a string"
    x = zlib.compress(text.encode("utf-8"))
    x = rs.encode(bytearray(x))
    return x

def bytearray_to_text(x):
    # Apply error correction and decompress
    assert isinstance(x, bytearray), "expected a bytearray"
    try:
        text = rs.decode(x)
        text = zlib.decompress(text)
        return text.decode("utf-8")
    except BaseException:
        return False

def change_bits_to_text(payload):
    candidates = Counter()
    for candidate in bits_to_bytearray(payload).split(b'\x00\x00\x00\x00'):
#             print(candidate)
            candidate = bytearray_to_text(bytearray(candidate))
#             print(candidate)
            if candidate:
                    candidates[candidate] += 1

            # choose most common message
    if len(candidates) == 0:
        raise ValueError('Failed to find message.')

    candidate, count = candidates.most_common(1)[0]
    return candidate
	
def change_text_to_bits(text):

    message = text_to_bits(text) + [0] * 32
    width,height,depth= 360,360,1
    payload = message
    
    while len(payload) < width * height * depth:
        payload += message

    payload = payload[:width * height * depth]
    payload = np.reshape(payload,(1,width,height,depth))
    return payload

# TF related fns
def reuse_variable_scope(func):
    def wrapper(*args,**kwargs):
        scope=tf.get_variable_scope()
        #print(tf.get_default_graph())
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            return func(*args,**kwargs)
    return wrapper

def get_variables(*args):
        a=(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"%s"%i)for i in args)
        return a,args
		
def get_optimizer(lr):
    return tf.train.AdamOptimizer(lr,0.5)


""" __name__ == '__main__' """
batch=8
rem = len(all_image_paths) % batch
all_image_paths = all_image_paths[:-rem]
init=pied_piper(all_image_paths,BATCH_SIZE=batch,coworkers=4,buffer=8)
iterator,init_op=init.pipeline()
i = 0
 
 
Message    = "How you doing ?"        #@param {type:"string"}
vectors = change_text_to_bits(Message)
# Creating batches for messages
a = []
for i in range(batch):
    a.append(vectors[0])
batched_vectors = np.array(a,dtype=np.float32)
 
y_true = tf.convert_to_tensor(batched_vectors)
""" TRAINING CRITIC NETWORK ALONE """
# ----------------------------------------------------------------------------------------------------------------
cover = iterator.get_next() # cover Image
 
model=Model(hiddendim=32,Batch_size=batch,data_depth=1)
 
c,y_t,s,y_p=model.inference(cover = cover,y_true = y_true)
 
 
_,_,_,c_s,s_s=model.build_losses(c,y_t,s,y_p)
 
optim = get_optimizer(0.00001)
 
loss=c_s-s_s
 
 
gradients_critic = optim.compute_gradients(loss=loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"critic"))
 
capped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gradients_critic]
 
critics_train_op = optim.apply_gradients(capped_gvs,name="critic")
#-----------------------------------------------------------------------------------------------------------------------
 
""" TRAINING WHOLE MODEL """
 
#-----------------------------------------------------------------------------------------------------------------------
 
cover_e,y_true_e,steg_e,y_pred_e = model.inference(cover = cover,y_true = y_true,quantize = True)
enc_mse,dec_mse,dec_acc,_,steg_score = model.build_losses(cover_e,y_true_e,steg_e,y_pred_e)
 
model_optim = get_optimizer(0.00001)
model_loss = 100.0 * enc_mse + dec_mse + steg_score
 
total_model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"encoder") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"decoder")
 
gradients=model_optim.compute_gradients(loss = model_loss,var_list = total_model_params)
 
# capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
 
model_train_op = model_optim.apply_gradients(gradients)
 
 
#-----------------------------------------------------------------------------------------------------------------------
""" TRAINING DECODER NETWORK FOR BETTER DECODING"""
#-----------------------------------------------------------------------------------------------------------------------
 
dec_optim = get_optimizer(0.01)
dec_loss = dec_mse*100
dec_gradients = dec_optim.compute_gradients(loss = dec_loss, var_list =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"decoder"))
dec_train_op =  model_optim.apply_gradients(dec_gradients)
 
#-----------------------------------------------------------------------------------------------------------------------
 
 
saver = tf.train.Saver()
num_iterations = 1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(init_op)
    
    #Restore trained variables from disk to RAM.
    saver.restore(sess, "trained_models/model-499.ckpt")
    
    #Encoding and decoding at same time
    s = time.time()
    for i in range(num_iterations):
            cover_e_s,y_true_es,steg_es,y_pred_es = sess.run([cover_e,y_true_e,steg_e,y_pred_e])
    print('\n\n')
    print('-------------------------------------------------------------------------------------------------------')
    print(f"Number of images Encoded and decoded ::::: {len(cover_e_s)*num_iterations}")
    print("-------------------------------------------------------------------------------------------------------")
    print(f"Total time consumed for Message extraction :::: {(time.time()-s)} seconds")
    print("-------------------------------------------------------------------------------------------------------")
 
change_image = 4            #@param {type:"slider", min:0, max:7, step:1}
print("Cover Image")
plt.imshow(cover_e_s[change_image])
plt.title("Cover")
plt.show()
 
pic = (np.clip(steg_es[change_image],0.,1.))
print("Stego Image")
 
plt.imshow(pic)
plt.title("Stego Image")
plt.show()
num = change_image
y_pred = np.asarray(y_pred_es[num] > 0.0,dtype=np.float32)
# arr = confusion_matrix(y_true_es[num],y_pred)
true = np.asarray(y_pred,dtype=int)
true = np.reshape(true,-1)
# givebits()
payload = true
decoded_text = change_bits_to_text(true)
final_string = re.sub("_"," ",decoded_text)
print("Message Decoded : ", final_string)











