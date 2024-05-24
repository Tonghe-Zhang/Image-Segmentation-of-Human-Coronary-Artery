import torch
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm
from data import Stenosis_Dataset
from model import UNet_vanilla
from utils import current_time, load_model, calculate_f1_score
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
from loss import FocalLoss

# hyper parameters
num_epochs=10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load the test set
print("Loading Test Set...")
test_set = Stenosis_Dataset(mode="test")
batch_size=4
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
test_scores=[]
print(f"Finished Loading Dataset of length{len(test_set)}. Training Begins.")

# load the best model on validation set
model_time="2024-05-11-21-49"
model_id=6
model_directory=f"./models/{model_time}/{model_id}.pt"
print(f"Loading Saved Model in ./model/{model_time}/{model_id}/.pt")
model=UNet_vanilla()
load_model(model, model_time, model_id)
model=model.to(device)
print(f"Finished Loading Saved Model. Testing starts.")


if __name__=="__main__":
    # Prevent the submodules from starting a multiple process prematurely
    multiprocessing.freeze_support()
    # test the model.
    with torch.no_grad():
        total_samples, total_f1_score = 0, 0
        for batch_idx, (inputs, masks) in tqdm(enumerate(test_loader)):
            # move to gpu.
            inputs, masks = inputs.to(device), masks.to(device)

            # the predicted masks are floating point tensors in [0,1]
            predicted_masks = model(inputs)

            # decision-making to convert soft decision in [0,1] to hard decision in {0,1}
            predicted_masks = predicted_masks[:, 1, :, :] > predicted_masks[:, 0, :, :]

            tested_f1=calculate_f1_score(predicted_masks,masks)

            total_samples += inputs.size(0)
            total_f1_score += tested_f1 * inputs.size(0)

            test_scores.append(tested_f1.item())
        average_f1_score=(total_f1_score / total_samples).item()
        print(f"----Tested Average f1 Score={average_f1_score}")

    # plot the test results.
    os.makedirs('curves/'+model_time+'/test', exist_ok=True)
    plt.plot(test_scores)
    plt.xlabel("batch")
    plt.ylabel("F1 score")
    plt.title("Best Model's F1 Score on the Test Set"+"\n"+f"Mean={np.mean(test_scores)}, variance={np.var(test_scores)}")
    plt.savefig("./curves/"+model_time+"/test/f1-all"+current_time())
    plt.show()
    
    # log the test results.
    with open("./log/"+model_time+"/test_f1.txt", "w") as f_test:
        np.savetxt("./log/"+model_time+"/test_f1.txt", np.array(test_scores))