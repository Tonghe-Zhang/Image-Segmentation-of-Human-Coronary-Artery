import torch
import os
import matplotlib.pyplot as plt
import numpy as np

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

def log_output(start_time, epoch, lr_scheduler, train_batch_ids, train_losses,eval_score):
    """"
    output training and evaluation results 
    after each epoch
    """
    # for coarse information about both training and evaluation
    with open("./log/"+start_time+"train_eval.txt", "a") as f:
        f.write(f"\n epoch: {epoch}\t")
        f.write(f"lr: {lr_scheduler.get_last_lr()[0]}\t")
        f.write(f"train_loss: {train_losses[-1]}\t")
        f.write(f"eval_score: {eval_score[-1]}\n")
        f.close()
    # fine-grained loss curve
    full_loss=np.array([train_batch_ids, train_losses]).transpose()
    with open("./log/"+start_time+"full_loss.txt", "ab") as f:
        np.savetxt(f, full_loss)
    