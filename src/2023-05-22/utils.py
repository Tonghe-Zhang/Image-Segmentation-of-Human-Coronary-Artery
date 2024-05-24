import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
from data import Stenosis_Dataset
from torch.utils.data import DataLoader

def load_hyper_param(hyper_param_file_name:str)->tuple:
    ''''
    given hyper parameter file name, return the params.   
    '''
    # print(f"hyper_param_file_name={hyper_param_file_name}")
    with open(hyper_param_file_name, 'r') as file:
        hyper_param = yaml.safe_load(file)
    lr=hyper_param['train']['learning_rate']
    K=hyper_param['train']['num_epochs']
    return (lr,K)

def load_set(mode:str)->tuple:
    """
    mode=='train', 'val', 'test'
    """
    print(f"Loading {mode} set...")
    dataset = Stenosis_Dataset(mode=mode)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    print(f"Finished loading {mode} set")

    return loader

def make_all_dirs():
# prepare for visualization
    start_time=current_time()
    os.makedirs('curves/'+start_time+'/eval', exist_ok=True)
    os.makedirs('curves/'+start_time+'/train', exist_ok=True)
    os.makedirs('curves/'+start_time+'/test', exist_ok=True)
    os.makedirs('log/'+start_time, exist_ok=True)
    best_model_state_dict=dict()
    # prepre for saving models
    os.makedirs('models/'+start_time, exist_ok=True)
    return start_time   
    
 







def current_time()->str:
    import datetime
    # Get the current date and time
    now = datetime.datetime.now()
    # Format the date and time as a string
    formatted_date_time = now.strftime("%Y-%m-%d-%H-%M")
    return formatted_date_time

def save_model(model, start_time:str, epoch_id=0):
    torch.save(model.state_dict(), os.path.join('models', start_time, str(epoch_id) + '.pt'))

def load_model(model, start_time, epoch_id=0):
    model.load_state_dict(torch.load(os.path.join('models', start_time, str(epoch_id) + '.pt')))

def visualize_train(start_time, epoch, train_batch_ids, train_losses):
    plt.plot(train_batch_ids, train_losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(f"Training Curve in the First {epoch} Epochs")
    plt.savefig(f"./curves/"+start_time+f"/train/{epoch}.png")
    plt.close()

def visualize_eval(start_time,eval_ids, eval_score):
    plt.plot(eval_ids, eval_score)
    plt.xlabel("epoch")
    plt.ylabel("F1 score")
    best_model_epoch=eval_ids[np.argmax(np.array(eval_score))]
    plt.axvline(best_model_epoch,c='red')
    plt.title(f"Evaluation Result in Different Epochs"+"\n"+f"Best Model is in Epoch {best_model_epoch}")
    plt.savefig(f"./curves/"+start_time+f"/eval/.png")
    plt.close()

def log_eval(start_time, epoch, lr_scheduler, train_batch_ids, eval_score):
    """"
    output evaluation results 
    after each epoch
    """
    # for coarse information about both training and evaluation
    log_file_name="./log/"+start_time+"/eval.txt"
    with open(log_file_name, "a") as f:
        """
        in each row the numpy array is
        epoch   learning rate   evaluation f1 score of this epoch
        """
        f.write(f"{float(epoch)}"+"\t"+f"{lr_scheduler.get_last_lr()[0]}"+"\t"+f"{eval_score[-1]}"+"\n")
    # fine-grained loss curve
    # full_loss=np.array([train_batch_ids, train_losses]).transpose()
    # with open("./log/"+start_time+"/full_loss.txt", "ab") as f:
    #    np.savetxt(f, full_loss)

def calculate_f1_score(predicted_masks,masks):
    # evaluate the prediction by precision and recall.
    tp = (masks * predicted_masks).sum()         # actually one, and you take it just as one.
    fp = ((1 - masks) * predicted_masks).sum()   # actually zero, but you take it as one.
    fn = (masks * ~predicted_masks).sum()        # actually one, but you take it as zero.
    f1 = tp / (tp + 0.5 * (fp + fn))
    return f1