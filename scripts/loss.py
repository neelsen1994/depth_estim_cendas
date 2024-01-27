import tensorflow as tf

def custom_loss(y_true, y_pred):
    # Define Weights for each Losses
    w_ssim = 1.0
    w_l1 = 0.1
    w_edges = 0.9
    
    # Structural Similarity Index (SSIM) Loss
    ssim_loss = tf.reduce_mean(
        1 - tf.image.ssim(
            y_true, y_pred,
            max_val=640, filter_size=7, k1=0.01**2, k2=0.03**2
        )
    )
    
    # L1 Loss
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Edge Loss
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
    weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))
    
    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y
    edges_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))
    
    # Final Loss
    loss = (ssim_loss * w_ssim) + (l1_loss * w_l1) + (edges_loss * w_edges)
    
    return loss