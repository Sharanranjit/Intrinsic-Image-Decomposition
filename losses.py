import keras.backend as K
import tensorflow as tf

def albedo_loss_function(true_img, y_pred_shading):
    def albedo_inside_loss_function(y_true, y_pred):
    

        y_true_albedo = y_true
        y_pred_albedo = y_pred
        
        #Scale-Invariant Error For ALBEDO
        si_first_term = K.mean(K.l2_normalize(K.log(1+y_true_albedo) - K.log(1+y_pred_albedo)),axis=-1)
        si_second_term = 0.5*K.square(K.mean(K.abs(K.log(1+y_true_albedo) - K.log(1+y_pred_albedo)),axis=-1))
        si_alb = K.abs(si_first_term - si_second_term)

        # Edges for ALBEDO
        dy_true, dx_true = tf.image.image_gradients(y_true_albedo)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred_albedo)
        l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1) #Can use K.square instead of K.abs

        # Structural similarity (SSIM) index for ALBEDO
        ssim_alb = K.clip((1 - tf.image.ssim(y_true_albedo, y_pred_albedo, 1.0)) * 0.5, 0, 1)

        #Reconstruction Loss as I = A.S
        pred_img = tf.multiply(y_pred_albedo, y_pred_shading)    	
        ssim_recons = K.clip((1 - tf.image.ssim(true_img, pred_img, 1.0)) * 0.5, 0, 1)

        # Weights
        w1 = 1.0
        w2 = 1.0
        w3 = 1.0
        w4 = 1.0

        return (w1 * ssim_alb) + (w2 * K.mean(l_edges)) + (w3 * ssim_recons) + (w4 * K.mean(si_alb))

    return albedo_inside_loss_function


def shading_loss_function(true_img, y_pred_albedo):
    def shading_inside_loss_function(y_true, y_pred):
    
        y_true_shading = y_true
        y_pred_shading = y_pred

        #Scale-Invariant Error For SHADING
        si_first_term = K.mean(K.l2_normalize(K.log(1+y_true_shading) - K.log(1+y_pred_shading)),axis=-1)
        si_second_term = 0.5*K.square(K.mean(K.abs(K.log(1+y_true_shading) - K.log(1+y_pred_shading)),axis=-1))
        si_shad = K.abs(si_first_term - si_second_term)

        # Structural similarity (SSIM) index for SHADING
        ssim_shad = K.clip((1 - tf.image.ssim(y_true_shading, y_pred_shading, 1.0)) * 0.5, 0, 1)

        #Reconstruction Loss as I = A.S
        pred_img = tf.multiply(y_pred_albedo, y_pred_shading)
        ssim_recons = K.clip((1 - tf.image.ssim(true_img, pred_img, 1.0)) * 0.5, 0, 1)

        # Weights
        w1 = 1.0
        w2 = 1.0
        w3 = 1.0

        return (w1 * ssim_shad) + (w2 * ssim_recons) + (w3 * K.mean(si_shad))

    return shading_inside_loss_function