from keras.models import load_model
import tensorflow as tf
from keras import backend as K


weight_file_path = "./resnet50_malefemale.h5"
K.set_learning_phase(0)
K.set_image_data_format('channels_last')

model = load_model(weight_file_path)

sess = K.get_session()


builder = tf.saved_model.builder.SavedModelBuilder("export")

builder.add_meta_graph_and_variables(sess, ["tag"])
builder.save()
