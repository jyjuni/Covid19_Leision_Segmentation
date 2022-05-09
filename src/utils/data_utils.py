import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from sklearn.utils import class_weight

def show_mask(model, dataloader):

    # Take the first batch
    for batch in dataloader:
        sample_batch = batch
        break
        
    # Generat network prediction
    with torch.no_grad():
        y_pred = model(sample_batch['image'].cuda())

    # Print the shapes of the images, masks, predicted masks
    print('Sample batch \'image \'shape is: {0}\nSample batch \'mask\' shape is: {1}\nPredicted mask shape is: {2}'.format(sample_batch['image'].shape, 
                                                                                                                        sample_batch['mask'].shape,
                                                                                                                        y_pred.shape
                                                                                                                        ))
    
    # Conver Pytorch tensor to numpy array then reverse the preprocessing steps
    img = (sample_batch['image'][0][0].numpy())
    msk = (sample_batch['mask'][0].numpy() * 255)
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


def get_class_weight(dataset):
    y = dataset[11]["mask"].flatten()
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes =np.unique(y), y=y.numpy())
    class_weights = torch.tensor(class_weights,dtype=torch.float).cuda()
    return class_weights

def show_training(EPOCHS, train_loss, valid_loss, train_dice, valid_dice, train_jaccard, valid_jaccard):
    plt.figure(figsize=(18,8))
    plt.suptitle('Learning Curve', fontsize=18)

    plt.subplot(1,3,1)
    plt.plot(np.arange(EPOCHS)+1, train_loss, '-o', label='Training Loss')
    plt.plot(np.arange(EPOCHS)+1, valid_loss, '-o', label='Validation Loss')
    plt.xticks(np.arange(EPOCHS)+1)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(np.arange(EPOCHS)+1, train_dice, '-o', label='Training DICE score')
    plt.plot(np.arange(EPOCHS)+1, valid_dice, '-o', label='Validation DICE score')
    plt.xticks(np.arange(EPOCHS)+1)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('DICE score', fontsize=15)
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(np.arange(EPOCHS)+1, train_jaccard, '-o', label='Training Jaccard score')
    plt.plot(np.arange(EPOCHS)+1, valid_jaccard, '-o', label='Validation Jaccard score')
    plt.xticks(np.arange(EPOCHS)+1)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Jaccard score', fontsize=15)
    plt.legend()

    plt.tight_layout()
    plt.show()
