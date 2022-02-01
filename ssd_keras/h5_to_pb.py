from keras.models import load_model
import tensorflow as tf
import os
import os.path as osp
from keras import backend as K
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_loss_function.keras_ssd_loss import SSDLoss
#路径参数
input_path = 'train_dir'
weight_file = 'ssd300_pascal_07+12_epoch-33_loss-9.9595_val_loss-8.5557.h5'
weight_file_path = osp.join(input_path,weight_file)
output_graph_name = weight_file[:-3] + '.pb'
#转换函数
def h5_to_pb(h5_model,output_dir,model_name,out_prefix = "output_",log_tensorboard = True):
    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i],out_prefix + str(i + 1))
    sess = K.get_session()
    from tensorflow.python.framework import graph_util,graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)
    graph_io.write_graph(main_graph,output_dir,name = model_name,as_text = False)
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir,model_name),output_dir)
#输出路径
output_dir = osp.join(os.getcwd(),"train_dir")
#加载模型,包括自定义层
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0).compute_loss
h5_model = load_model(weight_file_path,custom_objects={'L2Normalization': L2Normalization,
                                                       'AnchorBoxes': AnchorBoxes,
                                                       'compute_loss': ssd_loss,})
h5_to_pb(h5_model,output_dir = output_dir,model_name = output_graph_name)
print('=======================model saved========================')

