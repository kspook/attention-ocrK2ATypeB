from __future__ import absolute_import

import os
import logging
import sys

import tensorflow as tf

def replacementA2K(c):
    if c=='!': 
        c='하'
    elif c=='#':
        c='나'
    elif c=='$':
        c='우'
    elif c=='%':
        c='리'
    elif c=='&':
        c='국'
    elif c=='(':
        c='민' 
    elif c==')':
        c='신'
    elif c=='*':
        c='한'
    elif c=='+':
        c='기'
    elif c=='.':
        c='업'
    elif c=='?':
        c='농'
    elif c=='@': 
        c='협'
    else: c=c
    return c
class Exporter(object):
    def __init__(self, model):
        self.model = model

    def save(self, path, model_format):
        if model_format == "savedmodel":
            logging.info("Creating a SavedModel.")

            builder = tf.saved_model.builder.SavedModelBuilder(path)
            freezing_graph = self.model.sess.graph
            text=freezing_graph.get_tensor_by_name('prediction:0')
            '''
            if sys.version_info >= (3,):
                #text = text.decode('iso-8859-1')
                text = text.decode('utf-8')
            '''
            new=''
            for c in text:
                c = replacementA2K(c)
                new+=c
            text=new 

            tensor_info_output = tf.saved_model.utils.build_tensor_info(text)            
            builder.add_meta_graph_and_variables(
                self.model.sess,
                ["serve"],
                signature_def_map={
                    'serving_default': tf.saved_model.signature_def_utils.predict_signature_def(
                        {'input': freezing_graph.get_tensor_by_name('input_image_as_bytes:0')},
                        {
                            'output': tensor_info_output,
                            #'output': freezing_graph.get_tensor_by_name('prediction:0'),                            
                            'probability': freezing_graph.get_tensor_by_name('probability:0')
                        }
                    ),
                },
                clear_devices=True)

            builder.save()

            logging.info("Exported SavedModel into %s", path)

        elif model_format == "frozengraph":

            logging.info("Creating a frozen graph.")

            if not os.path.exists(path):
                os.makedirs(path)

            output_graph_def = tf.graph_util.convert_variables_to_constants(
                self.model.sess,
                self.model.sess.graph.as_graph_def(),
                ['prediction', 'probability'],
            )

            with tf.gfile.GFile(path + '/frozen_graph.pb', "wb") as outfile:
                outfile.write(output_graph_def.SerializeToString())

            logging.info("Exported as %s", path + '/frozen_graph.pb')
