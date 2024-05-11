import torch
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm
from data import Stenosis_Dataset
from UNet import UNet
from utils import current_time, save_model, load_model, visualize_eval, visualize_train, log_output
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import os   


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0], device=device))
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

print("Loading Train Set...")
train_set = Stenosis_Dataset(mode="train")
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
print("Loading Validation Set...")
val_set = Stenosis_Dataset(mode="val")
val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4, drop_last=False)
print("Finished Loading Dataset. Training Begins.")

"""
drop_last is an argument in PyTorch's DataLoader that determines how to handle the 
last batch of data if the dataset size is not divisible by the batch size.
"""

if __name__=="__main__":
    print("Preparing Directories")
    # prepare for logging
    train_batch_ids=[]
    train_losses=[]
    eval_ids=[]
    eval_score=[]
    # prepare for visualization
    start_time=current_time()
    os.makedirs('curves/'+start_time+'/eval', exist_ok=True)
    os.makedirs('curves/'+start_time+'/train', exist_ok=True)
    os.makedirs('log', exist_ok=True)
    best_model_state_dict=dict()
    # prepre for saving models
    os.makedirs('models/'+start_time, exist_ok=True)
    
    print("Training Starts.")
    # Prevent the submodules from starting a multiple process prematurely
    multiprocessing.freeze_support()
    for epoch in tqdm(range(10)):
        print(f"EPOCH: {epoch}/10")

        # Train the model, single pass through the entire training set
        print(f"--START MODEL TRAINING")
        model.train()
        for batch_idx, (inputs, masks) in tqdm(enumerate(train_loader)):
            inputs, masks = inputs.to(device), masks.to(device)

            predicted_masks = model(inputs)
            loss = criterion(predicted_masks, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if batch_idx ==1:
            #     break
            if batch_idx % 25 == 0:
                print(f"batch_idx: {batch_idx}, train_loss: {loss.item()}")
                train_batch_ids.append(epoch*(len(train_loader))+batch_idx)
                train_losses.append(loss.item())
        lr_scheduler.step()
        # visulize training curve in this epoch
        visualize_train(start_time,epoch, train_batch_ids, train_losses)

        # Evaluate the model.
        model.eval()
        print(f"--START MODEL EVALUATION")
        with torch.no_grad():
            total_samples, total_f1_score = 0, 0
            for batch_idx, (inputs, masks) in tqdm(enumerate(val_loader)):
                # move to gpu.
                inputs, masks = inputs.to(device), masks.to(device)

                # the predicted masks are floating point tensors in [0,1]
                predicted_masks = model(inputs)

                # decision-making to convert soft decision in [0,1] to hard decision in {0,1}
                predicted_masks = predicted_masks[:, 1, :, :] > predicted_masks[:, 0, :, :]

                # evaluate the prediction by precision and recall.
                tp = (masks * predicted_masks).sum()         # actually one, and you take it just as one.
                fp = ((1 - masks) * predicted_masks).sum()   # actually zero, but you take it as one.
                fn = (masks * ~predicted_masks).sum()        # actually one, but you take it as zero.
                f1 = tp / (tp + 0.5 * (fp + fn))

                total_samples += inputs.size(0)
                total_f1_score += f1 * inputs.size(0)

            average_f1_score=(total_f1_score / total_samples).item()
            print(f"----evaluation f1 score={average_f1_score}")

            eval_ids.append(epoch)
            eval_score.append(average_f1_score)

        # log
        print("lOGGING...")
        log_output(start_time, epoch, lr_scheduler, train_batch_ids, train_losses,eval_score)

        # save the model
        print("SAVING MODEL")
        save_model(model, start_time, epoch_id=epoch)

    # visualize evaluation result across epochs
    print("VISUALIZE EVLUATION RESULT OF ALL THE EPOCHS.")
    visualize_eval(start_time, eval_ids, eval_score,)
