import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def predict(model, images, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale 
    if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))

    # Compute predictions, output is 2 images
    # 2 -> albedo & shading
    (albedo, shading) = model.predict(images, batch_size=batch_size)

    # Put in expected range
    return (albedo, shading)

def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = np.clip(np.asarray(Image.open(file), dtype=float) / 255, 0, 1)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)

def display_images(outputs, inputs=None, gt=None):
    
    import skimage
    from skimage.transform import resize

    shape = (outputs[0].shape[1], outputs[0].shape[2], 3)
    batch = outputs[0].shape[0]
    all_images = []

    for i in range(batch):
        imgs = []
        
        if isinstance(inputs, (list, tuple, np.ndarray)):
            x = inputs[i]
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)

        if isinstance(gt, (list, tuple, np.ndarray)):
            x = gt[i]
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)

        imgs.append((outputs[0])[i])
        imgs.append((outputs[1])[i])
        
        img_set = np.vstack(imgs)
        all_images.append(img_set)

    all_images = np.stack(all_images)
    
    return skimage.util.montage(all_images, multichannel=True, fill=(0,0,0))

def save_images(filename, outputs, inputs=None, gt=None):
    montage =  display_images(outputs, inputs)
    im = Image.fromarray(np.uint8(montage*255))
    im.save(filename)

def ssq_error(correct, estimate):
    """Compute the sum-squared-error for an image, where the estimate is
    multiplied by a scalar which minimizes the error. Sums over all pixels
    where mask is True. If the inputs are color, each color channel can be
    rescaled independently."""
    if np.sum(estimate**2) > 1e-5:
        alpha = np.sum(correct * estimate) / np.sum(estimate**2)
    else:
        alpha = 0.
    return np.sum((correct - alpha*estimate) ** 2)

def local_error(correct, estimate, window_size, window_shift):
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    M, N = correct.shape[:2]
    ssq = total = 0.
    for i in range(0, M - window_size + 1, window_shift):
        for j in range(0, N - window_size + 1, window_shift):
            correct_curr = correct[i:i+window_size, j:j+window_size]
            estimate_curr = estimate[i:i+window_size, j:j+window_size]
            ssq += ssq_error(correct_curr, estimate_curr)
            total += np.sum(correct_curr**2)
    assert ~np.isnan(ssq/total)
    return ssq / total

def evaluate(model, batch_size=6, verbose=True, data_zip_file = 'MPI_2.zip'):
    # Evaluation on MPI-Sintel dataset
    from data import extract_zip
    from io import BytesIO
    data = extract_zip(data_zip_file)
    mpi_test = list((row.split(',') for row in (data['MPI_2/data_test.csv']).decode("utf-8").split('\n') if len(row) > 0))
    N = len(mpi_test)

    def compute_errors(gt, pred):

        thresh = np.maximum((gt / pred), (pred / gt))
        
        a1 = (thresh < 1.25 ).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean(np.abs(gt - pred) / gt)
        mse = ssq_error(gt, pred)/(448*1024)
        lmse = local_error(gt, pred, 20, 10)

        return a1, a2, a3, abs_rel, mse, lmse

    alb_scores = np.zeros((6, N)) # six metrics
    shad_scores = np.zeros((6, N))

    bs = batch_size
        
    shape_rgb = (bs, 448, 1024, 3)
    shape_alb = (bs, 448, 1024, 3)
    shape_shad = (bs, 448, 1024, 3)
    
    rgb, alb, shad = np.zeros(shape_rgb), np.zeros(shape_alb), np.zeros(shape_shad)

    for i in range(N//bs):
        index = i*bs
        #Loading Test data
        for j in range(bs):
            sample = mpi_test[index+j]

            x = np.float32(np.clip(np.asarray((Image.open(BytesIO(data[sample[0]]))).resize((1024,448)))/255,0,1))
            y = np.float32(np.clip(np.asarray((Image.open(BytesIO(data[sample[1]]))).resize((1024,448)))/255,0,1))
            z = np.float32(np.clip(np.asarray((Image.open(BytesIO(data[sample[2]]))).resize((1024,448)))/255,0,1))

            rgb[j] = x      #input
            alb[j] = y      #albedo
            shad[j] = z     #shading
        
        # Compute results
        true_alb, true_shad = alb, shad
        (pred_alb, pred_shad) = predict(model, rgb, batch_size=bs)


                
        # Compute errors per image in batch
        for j in range(len(true_alb)):
            alb_errors = compute_errors(true_alb[j], pred_alb[j])
            shad_errors = compute_errors(true_shad[j], pred_shad[j])
            for k in range(len(alb_errors)):
                alb_scores[k][(i*bs)+j] = alb_errors[k]
                shad_scores[k][(i*bs)+j] = shad_errors[k]

    e_alb = alb_scores.mean(axis=1)
    e_shad = shad_scores.mean(axis=1)

    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'mse', 'lmse'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e_alb[0],e_alb[1],e_alb[2],e_alb[3],e_alb[4],e_alb[5]))
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'mse', 'lmse'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e_shad[0],e_shad[1],e_shad[2],e_shad[3],e_shad[4],e_shad[5]))
    
    return e_alb,e_shad

