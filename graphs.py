import ast
import matplotlib.pyplot as plt

file_path = 'losses/small_char.txt'
with open(file_path, 'r') as file:
    training_losses = ast.literal_eval(file.readline().strip())
    test_losses = ast.literal_eval(file.readline().strip())

epochs = range(1, len(training_losses) + 1)
plt.plot(epochs, training_losses, label='Training Loss', marker='o')
plt.plot(epochs, test_losses, label='Test Loss', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()