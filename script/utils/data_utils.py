import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

    
def show_pair(img, mask, idx):
    plt.figure(dpi=100)
    plt.suptitle(f'Sample {idx:04d}')

    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_datasets(train_dataset, valid_dataset, test_dataset):
    plt.figure()
    plt.title('Data split distribution')
    plt.bar(0, len(train_dataset), label='Train')
    plt.bar(1, len(valid_dataset), label='Validation')
    plt.bar(2, len(test_dataset), label='Test')
    plt.ylabel('Number of samples')
    plt.xticks([0,1,2],['Train', 'Validation', 'Test'])
    plt.legend()
    plt.show()

def show_mask(model, dataloader, verbose=True):
    # Take the first batch
    for batch in dataloader:
        sample_batch = batch
        break
        
    # Generat network prediction
    with torch.no_grad():
        y_pred = model(sample_batch['image'].cuda())

    if verbose:
        # Print the shapes of the images, masks, predicted masks
        print('Sample batch \'image \'shape is: {0}\nSample batch \'mask\' shape is: {1}\nPredicted mask shape is: {2}'.format(sample_batch['image'].shape, 
                                                                                                                            sample_batch['mask'].shape,
                                                                                                                        y_pred.shape
                                                                                                                        ))
    
    # Conver Pytorch tensor to numpy array then reverse the preprocessing steps
    img = (sample_batch['image'][0][0].numpy() * 255).astype('uint8')
    msk = (sample_batch['mask'][0].numpy() * 255).astype('uint8')
    img_id = sample_batch['img_id'][0]
    pred_msk_binary = (np.argmax(y_pred.cpu().numpy()[0], axis=0) * 255).astype('uint8')

    # Plot the smaple, ground truth, the prediction probability map, and the final predicted mask
    plt.figure(figsize=(12,9))
    plt.suptitle(f'Test sample Image {img_id}', fontsize=18)

    plt.subplot(2,3,1)
    plt.title('Input Image', fontsize=15)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.title('Ground Truth', fontsize=15)
    plt.imshow(msk, cmap='gray')
    plt.axis('off')

    plt.subplot(2,3,3)
    plt.title('Non-trained Binary Prediction', fontsize=15)
    plt.imshow(pred_msk_binary, cmap='gray')
    plt.axis('off')

    input_overlayed_GT = img.copy()
    input_overlayed_GT = cv2.cvtColor(input_overlayed_GT, cv2.COLOR_GRAY2RGB)
    input_overlayed_GT[msk == 255, :] = [0, 255, 0]
    plt.subplot(2,3,4)
    plt.title('Input Image overlayed with Ground Truth', fontsize=15)
    plt.imshow(input_overlayed_GT)
    plt.axis('off')

    input_overlayed_Pred = img.copy()
    input_overlayed_Pred = cv2.cvtColor(input_overlayed_Pred, cv2.COLOR_GRAY2RGB)
    input_overlayed_Pred[pred_msk_binary == 255, :] = [255, 0, 0]
    plt.subplot(2,3,5)
    plt.title('Input Image overlayed with Prediction', fontsize=15)
    plt.imshow(input_overlayed_Pred)
    plt.axis('off')

    GT_overlayed_prediction = np.zeros_like(img)
    GT_overlayed_prediction = cv2.cvtColor(GT_overlayed_prediction, cv2.COLOR_GRAY2RGB)
    GT_overlayed_prediction[msk == 255, 1] = 255
    GT_overlayed_prediction[pred_msk_binary == 255, 0] = 255
    plt.subplot(2,3,6)
    plt.title('Ground Truth overlayed with Prediction', fontsize=15)
    plt.imshow(GT_overlayed_prediction)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

