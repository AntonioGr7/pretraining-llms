import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the text file
file_path = 'log/log.txt'

# Read the data from the text file
data = []
with open(file_path, 'r') as file:
    for line in file:
        parts = line.split()
        step = int(parts[0])
        category = parts[1]
        if category not in ['train','val']:
            continue
        loss = float(parts[2])
        data.append((step, category, loss))

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data, columns=['Step', 'Category', 'Loss'])

# Separate the data into training and validation losses
train_df = df[df['Category'] == 'train']
val_df = df[df['Category'] == 'val']

# Plot the training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(train_df['Step'], train_df['Loss'], label='Train Loss')
plt.plot(val_df['Step'], val_df['Loss'], label='Validation Loss', linestyle='--')


plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('log/train_vs_val_loss.png')

plt.show()