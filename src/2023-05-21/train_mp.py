import torch
from tqdm import tqdm as tqdm
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


from utils import current_time, save_model, load_hyper_param, load_model, visualize_eval, visualize_train, log_eval,load_set, make_all_dirs

from loss import FocalLoss
from model import UNet_vanilla, ResUNet

def train_mp(rank, world_size):
    # distributed training.
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyper parameters
    (learning_rate,num_epochs, loss_type, model_class)=load_hyper_param(os.path.join('config','config.yaml'))

    # model class
    if model_class=="UNet_vanilla":
        model = UNet_vanilla().to(device)
    elif model_class=="ResUNet":
        model=ResUNet().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # loss function
    if loss_type=="CE":
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0], device=device))
    elif loss_type=="focal":
        criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')  # You can adjust alpha and gamma as needed
    
    # optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # load datasets.
    train_loader=load_set("train")
    val_loader=load_set("val")

    # prepare for logging
    train_batch_ids=[]
    train_losses=[]
    eval_ids=[]
    eval_score=[]
    start_time=make_all_dirs()
    
    print("Training Starts.")
    # Prevent the submodules from starting a multiple process prematurely
    multiprocessing.freeze_support()
    for epoch in tqdm(range(num_epochs)):
        print(f"EPOCH: {epoch}/{num_epochs}")

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
            
            if batch_idx % 25 == 0:
                print(f"batch_idx: {batch_idx}, train_loss: {loss.item()}")
                train_batch_ids.append(epoch*(len(train_loader))+batch_idx)
                train_losses.append(loss.item())
                
                with open("./log/"+start_time+"/full_loss_realtime.txt", "a") as f_intime:
                    current_step=epoch*(len(train_loader))+batch_idx 
                    f_intime.write(f"{current_step}"+"\t"+f"{loss.item()}"+"\n")
                    
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
            
        # save evaluation result
        log_eval(start_time, epoch, lr_scheduler, train_batch_ids, eval_score)
        
        # save the model
        print("SAVING MODEL")
        save_model(model, start_time, epoch_id=epoch)

    # visualize evaluation result across epochs
    print("VISUALIZE EVLUATION RESULT OF ALL THE EPOCHS.")
    visualize_eval(start_time, eval_ids, eval_score)

    dist.destroy_process_group()

if __name__ == '__main__':
    train_loader=load_set("train")
    val_loader=load_set("val")
    world_size = 3
    mp.spawn(train_mp, args=(world_size,), nprocs=world_size, join=True)
