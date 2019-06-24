#!C:\Users\sinhan\Anaconda3\envs\py37g
#-*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import re

import tensorflow as tf

from six import b

#    CHARMAP = ['', '', ''] + list('!#$%&()*+-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ?@')
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def replacementK2A(c):
    if c=='하': 
        c='!'
    elif c=='나':
        c='#'
    elif c=='우':
        c='$'
    elif c=='리':
        c='%'
    elif c=='국':
        c='&'
    elif c=='민':
        c='(' 
    elif c=='신':
        c=')'
    elif c=='한':
        c='*'
    elif c=='기':
        c='+'
    elif c=='업':
        c='.'
    elif c=='농':
        c='?'
    elif c=='협': 
        c='@'
    else: c=c
    return c

def generate(annotations_path, output_path, log_step=5000,
             force_uppercase=True, save_filename=False):

    logging.info('Building a dataset from %s.', annotations_path)
    logging.info('Output file: %s', output_path)

    writer = tf.python_io.TFRecordWriter(output_path)
    longest_label = ''
    idx = 0

    with open(annotations_path, 'r', encoding='utf-8') as annotations:
        for idx, line in enumerate(annotations):
            line = line.rstrip('\n')

            # Split the line on the first whitespace character and allow empty values for the label
            # NOTE: this does not allow whitespace in image paths
            line_match = re.match(r'(\S+)\s(.*)', line)
            if line_match is None:
                logging.error('missing filename or label, ignoring line %i: %s', idx+1, line)
                continue
            (img_path, label) = line_match.groups() 
            print(label)
            #label=label.encode('utf-8')        
            with open(img_path, 'rb') as img_file:
                img = img_file.read()    

            new=''
            for c in label:
                c= replacementK2A(c)
                new+=c
            label=new
             
            if force_uppercase:
                label = label.upper()

            if len(label) > len(longest_label):
                longest_label = label

            feature = {}
            feature['image'] = _bytes_feature(img)
            feature['label'] = _bytes_feature(b(label))
            if save_filename:
                feature['comment'] = _bytes_feature(b(img_path))

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())

            if idx % log_step == 0:
                logging.info('Processed %s pairs.', idx+1)
                logging.info('label %s ord(label) %i', label, ord(label))
            logging.info('label %s ord(label) %i', label, ord(label))
    if idx:
        logging.info('Dataset is ready: %i pairs.', idx+1)
        logging.info('Longest label (%i): %s ,  %i', len(longest_label), longest_label, ord(longest_label))

    writer.close()
