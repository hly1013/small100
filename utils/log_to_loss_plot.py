"""
use txt file in `/log` -> make loss plot

result examples:
in `/log`,
log_1.txt -> log_1.png
log_2.txt -> log_2.png
"""

# read log.txt
with open('../log/log_1.txt', 'r') as f:
    lines = f.readlines()
lines[0] = lines[0].split('KST')[1]

# get loss, batch_idx
loss_lst = []
batch_idx_lst = []
for line in lines:
    if line.startswith('loss:'):
        loss = float(line.split('loss: ')[1].rstrip())
        loss_lst.append(loss)
    elif line.startswith('batch_idx:'):
        batch_idx = int(line.split('batch_idx: ')[1].rstrip())
        batch_idx_lst.append(batch_idx)

assert len(loss_lst) == len(batch_idx_lst)

# draw graph
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(batch_idx_lst, loss_lst)
ax.set(xlabel='batch idx', ylabel='loss', title='loss graph')

png_name = 'test'
fig.savefig(f'../log/{png_name}.png')

plt.show()



