import tensorflow as tf
gf = tf.GraphDef()
gf.ParseFromString(open('./forGo/saved_model.pb','rb').read())

#[n.name + '=>' +  n.op for n in gf.node if n.op in ( 'Softmax','Placeholder')]
