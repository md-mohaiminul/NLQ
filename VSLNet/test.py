import clip
import torch
import torch.nn as nn

x = torch.tensor([[0.1, 1.1, -0.1, float("nan")], [0.1, 1.1, -0.1, float("nan")]])
x = x.clamp(0,1)
x[x != x] = 0
print(x)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model, preprocess = clip.load('ViT-L/14')
#
# #image = torch.rand([1, 3, 224, 224]).to(device)
# text = clip.tokenize(["a diagram of a dog"]).to(device)
#
# with torch.no_grad():
#     #image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#
# #print(image_features.shape)
# print(text_features.shape)
# query_affine = nn.Linear(768, 128).cuda()
#
# x = model.token_embedding(text).type(model.dtype)  # [batch_size, n_ctx, d_model]
# x = x + model.positional_embedding.type(model.dtype)
# x = x.permute(1, 0, 2)  # NLD -> LND
# x = model.transformer(x)
# x = x.permute(1, 0, 2)  # LND -> NLD
# x = model.ln_final(x).type(torch.float32)
# x = query_affine(x)
#
# print(x.shape)

# x.shape = [batch_size, n_ctx, transformer.width]
# take features from the eot embedding (eot_token is the highest number in each sequence)
# x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ model.text_projection
#
# print(x.shape)

# print(text_features-x)
# print(torch.sum(text_features-x))