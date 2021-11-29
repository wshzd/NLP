import torch
print(torch.__version__)                # 查看torch版本
print(torch.version.cuda)               # 查看cuda版本
print(torch.backends.cudnn.version())   # 查看cudnn版本
print(torch.cuda.is_available())        # 查看前cuda版本下，torch能否使用gpu
