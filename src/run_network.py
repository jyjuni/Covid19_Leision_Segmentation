import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from sklearn.metrics import confusion_matrix, accuracy_score


def jaccard_coeff_binary(y_pred, y_true):
        """computes the Jaccard score coeffiencient for binary segmentation"""
        eps = 0.0001
        inter = torch.dot(y_pred.view(-1).float(), y_true.view(-1).float())
        sums = torch.sum(y_pred.float()) + torch.sum(y_true.float())
        jaccard_index = ((inter.float() + eps) / (sums.float() - inter.float() + eps))
        return jaccard_index.cpu().numpy()

def dice_coeff_binary(y_pred, y_true):
        """computes the DICE score coeffiencient for binary segmentation"""
        eps = 0.0001
        inter = torch.dot(y_pred.view(-1).float(), y_true.view(-1).float())
        union = torch.sum(y_pred.float()) + torch.sum(y_true.float())
        return ((2 * inter.float() + eps) / (union.float() + eps)).cpu().numpy()
    
def train_net(net, epochs, train_dataloader, valid_dataloader, optimizer, criteria):
    """training function"""
    trigger_times, patience = 0, 5
    last_loss = float("inf")
    if not os.path.isdir('{0}'.format(net.name)):
        os.mkdir('{0}'.format(net.name))
    
    n_train = len(train_dataloader)
    n_valid = len(valid_dataloader)    
    
    train_loss = list()
    valid_loss = list()
    train_dice = list()
    valid_dice = list()
    train_jaccard = list()
    valid_jaccard = list()
    
    for epoch in range(epochs):
        
        ########################################################### Training ###########################################################
        net.train()
        train_batch_loss = list()
        train_batch_dice = list()
        train_batch_jaccard = list()
        
        for i, batch in enumerate(train_dataloader):

            # Load a batch
            imgs = batch['image'].cuda()
            true_masks = batch['mask'].cuda()

            # estimated mask using current weights
            y_pred = net(imgs)

            # batch-wise loss
            loss = criteria(y_pred, true_masks)
            batch_loss = loss.item()
            train_batch_loss.append(batch_loss)

            # binary mask for DICE score
            pred_binary = torch.argmax(y_pred, axis=1) #axis=1 because the format is [batch, channel, height, width]
            
            # batch-wise DICE score
            batch_dice_score = dice_coeff_binary(pred_binary, true_masks)
            train_batch_dice.append(batch_dice_score)
            
            # batch-wise Jaccard score
            batch_jaccard_score = jaccard_coeff_binary(pred_binary, true_masks)      
            train_batch_jaccard.append(batch_jaccard_score)
            

            # Reset gradient values
            optimizer.zero_grad()

            # Compute the backward losses
            loss.backward()

            # Update the weights
            optimizer.step()
            
            # Print the progress
            print(f'EPOCH {epoch + 1}/{epochs} - Training Batch {i+1}/{n_train} - Loss: {batch_loss:08f}, DICE score: {batch_dice_score:08f}, Jaccard score: {batch_jaccard_score:08f}            ', end='\r')
            

        
        average_training_loss = np.array(train_batch_loss).mean()
        average_training_dice = np.array(train_batch_dice).mean()
        average_training_jaccard = np.array(train_batch_jaccard).mean()
        train_loss.append(average_training_loss)
        train_dice.append(average_training_dice)
        train_jaccard.append(average_training_jaccard)
        
        ########################################################## Validation ##########################################################
        
        net.eval()
        valid_batch_loss = list()
        valid_batch_dice = list()
        valid_batch_jaccard = list()
        
        # evaluation mode (effects some layers such as BN and Dropout)
        with torch.no_grad():
            for i, batch in enumerate(valid_dataloader):

                # Load a batch
                imgs = batch['image'].cuda()
                true_masks = batch['mask'].cuda()

                # estimated mask using current weights
                y_pred = net(imgs)

                # batch-wise loss
                loss = criteria(y_pred, true_masks)
                batch_loss = loss.item()
                valid_batch_loss.append(batch_loss)

                # binary mask for DICE score
                pred_binary = torch.argmax(y_pred, axis=1) #axis=1 because the format is [batch, channel, height, width]

                # batch-wise DICE score
                batch_dice_score = dice_coeff_binary(pred_binary, true_masks)
                valid_batch_dice.append(batch_dice_score)
                
                # batch-wise Jaccard score
                batch_jaccard_score = jaccard_coeff_binary(pred_binary, true_masks)
                valid_batch_jaccard.append(batch_jaccard_score)

                # Print the progress
                print(f'EPOCH {epoch + 1}/{epochs} - Validation Batch {i+1}/{n_valid} - Loss: {batch_loss:08f}, DICE score: {batch_dice_score:08f}, Jaccard score: {batch_jaccard_score:08f}            ', end='\r')

        average_validation_loss = np.array(valid_batch_loss).mean()
        average_validation_dice = np.array(valid_batch_dice).mean()
        average_validation_jaccard = np.array(valid_batch_jaccard).mean()
        valid_loss.append(average_validation_loss)
        valid_dice.append(average_validation_dice)
        valid_jaccard.append(average_validation_jaccard)
        
        # Early stopping
        if average_validation_loss > last_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                return train_loss, train_dice, train_jaccard, valid_loss, valid_dice, valid_jaccard

        else:
            print('trigger times: 0')
            trigger_times = 0
        
        last_loss = average_validation_loss
        
        print(f'EPOCH {epoch + 1}/{epochs} - Training Loss: {average_training_loss:08f}, Training DICE score: {average_training_dice:08f}, Training Jaccard score: {average_training_jaccard:08f}, Validation Loss: {average_validation_loss:08f}, Validation DICE score: {average_validation_dice:08f}, Validation Jaccard score: {average_validation_jaccard:08f}')

        # Saving Checkpoints
        torch.save(net.state_dict(), f'{net.name}/epoch_{epoch+1:03}.pth')
    
    return train_loss, train_dice, train_jaccard, valid_loss, valid_dice, valid_jaccard

def test_net(net, test_dataloader, loss_function, save_dir):
    """
    compute the overall performance of the model on all test samples and save the prediction masks
    This function takes the following arguments:
    1. `net`: The model we want to test.
    2. `test_dataloader`: The `DataLoader` for the test set.
    3. `loss_function`: The loss function to calculate the loss.

    This function returns:
    1. `test_loss`: The average test loss.
    2. `test_dice`: The average test DICE score.
    3. `test_jaccard`: The average test Jaccard score.
    4. `test_accuracy`: The overall accuracy of the model.
    5. `test_CM`: The normalized confusion matrix of the model.
    """

    # Create the pred_mask folder
    save_path = os.path.join(save_dir, "pred_mask")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    net.eval()
    
    n_test = len(test_dataloader)
    test_batch_loss = list()
    test_batch_dice = list()
    test_batch_jaccard = list()
    test_batch_accuray = list()
    test_batch_CM = list()

    # This part is almost the same as the validation loop in `train_net` function. 
    # The difference is that we will calculate the accuracy and confusion matrix per each batch and save the predicted images.
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):

            # Load a batch and pass it to the GPU
            imgs = batch['image'].cuda()
            true_masks = batch['mask'].cuda()
            img_ids = batch['img_id'].numpy().astype('int')

            # Produce the estimated mask using current weights
            y_pred = net(imgs)

            # Compute the loss for this batch and append it to the epoch loss
            loss = loss_function(y_pred, true_masks)
            batch_loss = loss.item()
            test_batch_loss.append(batch_loss)

            # Make the binary mask to compute the DICE score. Since the y_pred is a Pytoch tensor, we use `torch.argmax()` instead of `np.argmax()`.
            # the axis must be 1 instead of 0 because the format is [batch, channel, height, width]
            pred_binary = torch.argmax(y_pred, axis=1)

            # Compute the DICE score for this batch and append it to the epoch dice
            batch_dice_score = dice_coeff_binary(pred_binary, true_masks)
            test_batch_dice.append(batch_dice_score)
            
            # Compute the Jaccard score for this batch and append it to the epoch dice
            batch_jaccard_score = jaccard_coeff_binary(pred_binary, true_masks)
            test_batch_jaccard.append(batch_jaccard_score)
            
            # Save the predicted masks
            for idx, pred_msk in enumerate(pred_binary):
                cv2.imwrite(os.path.join(save_path, f'pred_mask_{img_ids[idx]:04}.png'), pred_msk.cpu().numpy())
            
            # Vectorize the true mask and predicted mask for this batch
            vectorize_true_masks = true_masks.view(-1).cpu().numpy()
            vectorize_pred_masks = pred_binary.view(-1).cpu().numpy()
            
            # Compute the accuracy for this batch and append to the overall list
            batch_accuracy = accuracy_score(vectorize_true_masks, vectorize_pred_masks)
            test_batch_accuray.append(batch_accuracy)
            
            # Compute the normalized confusion matrix for this batch and append to the overall list
            batch_CM = confusion_matrix(vectorize_true_masks, vectorize_pred_masks, normalize='true', labels=[0, 1])
            test_batch_CM.append(batch_CM)

            # Print the progress
            print(f'Test Batch {i+1}/{n_test} - Loss: {batch_loss}, DICE score: {batch_dice_score}, Jaccard score: {batch_jaccard_score}, Accuracy: {batch_accuracy}', end='\r')

    test_loss = np.array(test_batch_loss).mean()
    test_dice = np.array(test_batch_dice).mean()
    test_jaccard = np.array(test_batch_jaccard).mean()
    test_accuracy = np.array(test_batch_accuray).mean()
    test_CM = np.array(test_batch_CM).mean(axis=0)
    
    return test_loss, test_dice, test_jaccard, test_accuracy, test_CM



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
