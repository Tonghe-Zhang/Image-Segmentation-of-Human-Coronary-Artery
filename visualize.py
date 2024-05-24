import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt

def visualize(start_time:str):
    # read loss and eval results
    train_loss=np.loadtxt(os.path.join('log',start_time,'full_loss_realtime.txt'))
    eval_scores=np.loadtxt(os.path.join('log',start_time,'eval.txt'))

    # unpack loss curve data
    steps = train_loss[:, 0]   # batch ids
    losses= train_loss[:, 1] # loss at each iteration

    # unpack f1 score data
    epochs=eval_scores[:,0]
    f1scores=eval_scores[:,2]
    f1score_std=np.sqrt(eval_scores[:,1])

    step_per_epoch=max(steps)//max(epochs)

    # create a figure with two subplots and separate y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    # plot loss curve
    loss_curve, =ax1.plot(steps,losses,label='Train Loss', color='tab:blue')
    ax1.set_ylabel('Loss ')
    ax1.set_xlabel('step')

    # plot F1 score
    scores_l = (f1scores - f1score_std)*100
    scores_h = (f1scores + f1score_std)*100
    best_epoch=np.argmax(f1scores)
    best_eval_score=f1scores[best_epoch]
    eval_optim=ax2.axvline(best_epoch*step_per_epoch, color='darkblue', linestyle='--', label=f'Highest F1 Score {best_eval_score*100:.3f} % '+'\n'+f'at Epoch {best_epoch}')
    ax2.axhline(best_eval_score*100, color='darkblue', linestyle='--')
    eval_curve, =ax2.plot(epochs*step_per_epoch, f1scores*100, color='darkblue',label='Ealution Score')
    ax2.fill_between(epochs*step_per_epoch, scores_l, scores_h, where=scores_h > scores_l, interpolate=True, alpha=0.25, color='tab:blue')
    ax2.set_ylabel('F1 score /%')
    ax2.set_xlabel('Epoch')

    # save figure
    lines = [loss_curve, eval_optim, eval_curve]
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc='center right',bbox_to_anchor=(0.95,0.3))
    plt.title("Training and Evaluation Result")
    plt.tight_layout()
    plt.savefig(f"./curves/"+start_time+f"/train-eval/.png")
    plt.show()



