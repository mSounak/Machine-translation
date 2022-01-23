import pathlib
import random
import string
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle



@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_string):
    strip_chars = string.punctuation
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")
    
    text = tf.strings.lower(input_string)
    
    text = tf.strings.regex_replace(text, "[%s]" % re.escape(strip_chars), "")

    return text