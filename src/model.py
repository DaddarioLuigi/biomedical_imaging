# src/model.py
# 3D U-Net con residui (come nel notebook) + aggiunte anti-overfitting marcate NEW

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv3D, Conv3DTranspose, UpSampling3D, MaxPooling3D,
    BatchNormalization, Activation, Concatenate, Add, Dropout
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


# ---------------------------
# Metriche & loss
# ---------------------------
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def iou_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    intersection = K.sum(y_true * y_pred)
    total = K.sum(y_true) + K.sum(y_pred)
    union = total - intersection
    return (intersection + smooth) / (union + smooth)


# ---------------------------
# Blocchi base (come nel notebook)
# ---------------------------
def conv3_bn_relu(x, filters, reg=0.0):
    x = Conv3D(filters, 3, padding='same',
               kernel_regularizer=l2(reg) if reg > 0 else None)(x)  # NEW: L2 opzionale
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def down_block(x, filters, reg=0.0, p_dropout=0.0):
    """
    Encoder block con residuo downsampled (come nel notebook descritto):
      main:   Conv3x3 -> BN -> ReLU (x2 o x3) -> MaxPool(stride2)
      resid:  Conv1x1(stride2) per matchare canali/spatial
      out:    Add(main_pooled, resid)
      skip:   feature prima del pooling (per le skip connection)
    """
    # conv stack
    x1 = conv3_bn_relu(x, filters, reg)
    x1 = conv3_bn_relu(x1, filters, reg)
    # NEW: dropout opzionale nell'encoder
    if p_dropout > 0:
        x1 = Dropout(p_dropout)(x1)  # NEW
    skip = x1
    main = MaxPooling3D(pool_size=2)(x1)
    resid = Conv3D(filters, 1, strides=2, padding='same',
                   kernel_regularizer=l2(reg) if reg > 0 else None)(x)  # residual downsample
    out = Add()([main, resid])
    return out, skip

def up_block(x, skip, filters, reg=0.0, use_transpose=True, p_dropout=0.0):
    """
    Decoder block con upsample + skip (come nel notebook) + residuo upsampled.
      upsample: Conv3DTranspose o UpSampling3D+Conv1x1 (switch)
      concat con skip
      conv stack
      resid: UpSampling3D + Conv1x1 per matchare
      out: Add(main, resid)
    """
    if use_transpose:
        up = Conv3DTranspose(filters, 2, strides=2, padding='same',
                             kernel_regularizer=l2(reg) if reg > 0 else None)(x)
    else:
        # NEW: alternativa per ridurre checkerboard: UpSampling3D + Conv1x1
        up = UpSampling3D(size=2)(x)  # NEW
        up = Conv3D(filters, 1, padding='same',
                    kernel_regularizer=l2(reg) if reg > 0 else None)(up)  # NEW

    x2 = Concatenate()([up, skip])
    x2 = conv3_bn_relu(x2, filters, reg)
    x2 = conv3_bn_relu(x2, filters, reg)
    # NEW: dropout opzionale nel decoder
    if p_dropout > 0:
        x2 = Dropout(p_dropout)(x2)  # NEW

    resid = UpSampling3D(size=2)(x)
    resid = Conv3D(filters, 1, padding='same',
                   kernel_regularizer=l2(reg) if reg > 0 else None)(resid)
    out = Add()([x2, resid])
    return out


# ---------------------------
# Costruzione U-Net 3D
# ---------------------------
def build_unet_3d(
    input_shape: Tuple[int, int, int, int] = (64, 64, 64, 1),
    base_filters: int = 16,
    reg_l2: float = 1e-5,          # NEW: L2 globale
    p_dropout_enc: float = 0.1,    # NEW: dropout encoder
    p_dropout_dec: float = 0.1,    # NEW: dropout decoder
    p_dropout_bot: float = 0.1,    # NEW: dropout bottleneck
    use_transpose: bool = False    # NEW: default UpSampling3D+Conv più stabile
) -> Model:
    """
    Ricalca l'architettura del notebook (down/up con residui) con:
      - L2, Dropout, switch upsampling (NEW)
    """
    inputs = Input(shape=input_shape)

    # Encoder
    x0 = inputs
    x1, s1 = down_block(x0, base_filters,         reg=reg_l2, p_dropout=p_dropout_enc)
    x2, s2 = down_block(x1, base_filters * 2,     reg=reg_l2, p_dropout=p_dropout_enc)
    x3, s3 = down_block(x2, base_filters * 4,     reg=reg_l2, p_dropout=p_dropout_enc)
    x4, s4 = down_block(x3, base_filters * 8,     reg=reg_l2, p_dropout=p_dropout_enc)

    # Bottleneck
    b  = conv3_bn_relu(x4, base_filters * 16, reg=reg_l2)
    if p_dropout_bot > 0:
        b = Dropout(p_dropout_bot)(b)  # NEW
    b  = conv3_bn_relu(b, base_filters * 16, reg=reg_l2)

    # Decoder
    u3 = up_block(b,  s4, base_filters * 8, reg=reg_l2, use_transpose=use_transpose, p_dropout=p_dropout_dec)
    u2 = up_block(u3, s3, base_filters * 4, reg=reg_l2, use_transpose=use_transpose, p_dropout=p_dropout_dec)
    u1 = up_block(u2, s2, base_filters * 2, reg=reg_l2, use_transpose=use_transpose, p_dropout=p_dropout_dec)
    u0 = up_block(u1, s1, base_filters,     reg=reg_l2, use_transpose=use_transpose, p_dropout=p_dropout_dec)

    # Output (sigmoid per binary mask)
    out = Conv3D(1, 1, padding='same',
                 kernel_regularizer=l2(reg_l2) if reg_l2 > 0 else None)(u0)
    out = Activation('sigmoid')(out)

    model = Model(inputs, out, name="unet3d_residual")

    # Compilazione standard per il laboratorio
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, amsgrad=True),
        loss=dice_loss,                   # conforme al notebook / assignment
        metrics=[dice_coefficient, iou_coefficient]
    )
    return model
