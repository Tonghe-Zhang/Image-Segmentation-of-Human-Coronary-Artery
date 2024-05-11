import matplotlib.pyplot as plt

# Data parsed into separate lists
epochs = list(range(10))
lr = [0.0009755282581475768, 0.0009045084971874736, 0.0007938926261462366, 0.0006545084971874737, 0.0005,
      0.00034549150281252633, 0.0002061073738537635, 9.549150281252634e-05, 2.4471741852423235e-05, 0.0]
train_loss = [0.17131786048412323, 0.14891070127487183, 0.13007839024066925, 0.07344530522823334, 0.09236391633749008,
              0.08528542518615723, 0.06237294524908066, 0.06523670256137848, 0.0785488486289978, 0.049613069742918015]
eval_score = [0.18173861503601074, 0.23270094394683838, 0.24646219611167908, 0.29538288712501526, 0.3153317868709564,
              0.32043200731277466, 0.35055315494537354, 0.38471975922584534, 0.38291218876838684, 0.37385839223861694]

# Create a figure and a set of subplots
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot training loss on the primary y-axis
ax1.plot(epochs, train_loss, 'r-', label='Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax1.set_title('Training Loss and Evaluation Score by Epoch')

# Create a second y-axis for eval_score
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(epochs, eval_score, 'b-', label='Evaluation Score')
ax2.set_ylabel('Evaluation Score', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# # Operation on the secondary subplot for learning rate
# ax3 = plt.axes([0.15, 0.1, 0.75, 0.3])  # Control the position of the second subplot
# ax3.plot(epochs, lr, 'g-', label='Learning Rate')
# ax3.set_xlabel('Epoch')
# ax3.set_ylabel('Learning Rate', color='g')
# ax3.set_title('Learning Rate by Epoch')

# Show legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
# ax3.legend(loc='upper left')

# Prevent overlapping and layout issues
fig.tight_layout()

# Save the figure to a file and display it
plt.savefig('result.png')
plt.show()


def read_full_loss_curve(file_path):
    import numpy as np
    # Read the entire file to plot
    # Reading the data back from the file
    loaded_data = np.loadtxt(file_path)

    # Extract the first column (batch IDs) and the second column (training losses)
    x = loaded_data[:, 0]   # batch ids
    y = loaded_data[:, 1]   # losses
    # Plotting the data
    
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, c='blue', label='Training Loss per Batch ID')
    plt.plot(x, y, 'r--')  # Optional: Adds a red dashed line connecting the points
    plt.title("Training Loss Over Batches")
    plt.xlabel("Batch ID")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()



