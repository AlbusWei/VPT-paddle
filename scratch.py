"""
Unofficial code for VPT(Visual Prompt Tuning) paper of arxiv 2203.12119

A toy Tuning process that demostrates the code

the code is based on timm

"""
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import numpy as np
from .GetPromptModel import build_promptmodel


def setup_seed(seed):  # setting up the random seed
    import random
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(42)
batch_size = 2
edge_size = 384
data = paddle.randn((1, 3, edge_size, edge_size))
labels = paddle.ones(batch_size).long()  # long ones

model = build_promptmodel(num_classes=3, edge_size=edge_size, model_idx='ViT', patch_size=16,
                          Prompt_Token_num=10, VPT_type="Deep")  # VPT_type = "Shallow"
# test for updating
prompt_state_dict = model.obtain_prompt()
model.load_prompt(prompt_state_dict)

optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

preds = model.predict_batch([data])  # (1, class_number)
# print('before Tuning model output：', preds)

# check backwarding tokens
for param in model.parameters():
    if param.requires_grad:
        print(param.shape)

for i in range(3):
    print('epoch:', i)
    optimizer.zero_grad()
    outputs = model.predict_batch([data])
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

preds = model.predict_batch([data])  # (1, class_number)
print('After Tuning model output：', preds)
