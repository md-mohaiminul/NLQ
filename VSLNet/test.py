import torch
import torch.nn as nn

# from mmaction.apis import init_recognizer, inference_recognizer
# from mmaction.models import build_model
#
# config_file = '/playpen-storage/mmiemon/lvu_state_space/Video-Swin-Transformer/configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py'
# # # download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '/playpen-storage/mmiemon/lvu_state_space/Video-Swin-Transformer/checkpoints/swin_base_patch244_window877_kinetics600_22k.pth'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = init_recognizer(config_file, checkpoint_file, device=device)
#
# avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1)).to(device)
#
# frames = torch.rand([1, 3, 16, 224, 224])
# f = model.extract_feat(frames.to(device))
# print(f.shape)
# f =avg_pool(f).view(-1)
# print(f.shape)

all_features = torch.rand([0,1024])
x = torch.rand([1, 1024])
all_features = torch.cat((all_features, x), 0)
print(all_features.shape)

torch.save(all_features, 'file.pt')
all_features = torch.load('file.pt')
print(all_features.shape)
