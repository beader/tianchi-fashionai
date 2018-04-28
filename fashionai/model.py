from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D, Softmax, Reshape, RepeatVector, Embedding
from keras.layers import multiply
from keras.applications import Xception


def build_xception_model(input_shape, num_cat, num_lm):
    input_imgs = Input(shape=input_shape)
    input_cats = Input(shape=(1,), dtype='int32')
    cat_embed = Embedding(num_cat, num_lm)(input_cats)
    cat_embed = Reshape((num_lm,))(cat_embed)
    cat_embed = Softmax()(cat_embed)
    cat_embed = RepeatVector(2)(cat_embed)
    cat_embed = Reshape((num_lm * 2,))(cat_embed)

    x = Xception(weights=None, input_shape=input_shape, include_top=False)(input_imgs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_lm * 2)(x)

    preds = multiply([x, cat_embed])

    model = Model([input_imgs, input_cats], preds)
    return model
