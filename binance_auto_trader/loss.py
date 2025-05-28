from tensorflow.keras import backend as K
import tensorflow as tf
from config import *

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=1)
    return loss

def focal_asymmetric_loss(alpha_long=3.0, alpha_short=3.0, alpha_hold=0.5, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())

        # Trọng số cho từng lớp
        alpha = (
                y_true[:, 0] * alpha_hold +
                y_true[:, 1] * alpha_long +
                y_true[:, 2] * alpha_short
        )

        # Dự đoán sai: trọng số sai sẽ lớn hơn
        # max_pred_class: index của lớp dự đoán có xác suất cao nhất
        pred_class = K.argmax(y_pred, axis=-1)
        true_class = K.argmax(y_true, axis=-1)

        # Mask: 1 nếu sai, 0 nếu đúng
        mistake_mask = K.cast(K.not_equal(pred_class, true_class), dtype='float32')

        cross_entropy = -K.sum(y_true * K.log(y_pred), axis=1)
        return alpha * cross_entropy * (1.0 + gamma * mistake_mask)
    return loss

def risk_aware_loss(penalty_matrix):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -K.sum(y_true * K.log(y_pred), axis=1)

        # Tính penalty dựa trên dự đoán sai
        pred_class = K.argmax(y_pred, axis=1)
        true_class = K.argmax(y_true, axis=1)

        penalties = tf.gather_nd(penalty_matrix, tf.stack([true_class, pred_class], axis=1))
        return cross_entropy * penalties
    return loss

focal = focal_loss(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA)
custom_loss = focal_asymmetric_loss(
    alpha_long=AlPHA_LONG,
    alpha_short=ALPHA_SHORT,
    alpha_hold=ALPHA_HOLD,
    gamma=FOCAL_LOSS_GAMMA
)

penalty_matrix = tf.constant([
    [1.0, 3.0, 3.0],  # Nếu thực tế là HOLD mà dự đoán LONG/SHORT thì phạt cao
    [1.0, 1.0, 2.0],
    [1.0, 2.0, 1.0]
], dtype=tf.float32)

risk_loss = risk_aware_loss(penalty_matrix)

