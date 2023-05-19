import torch
from thop import profile

"""
CRNN
    video   37392876    0.032438272
    wifi    73660       0.004306224
    both    37466524    0.76392424
ViFi
    video   37392876    0.032438272
    wifi    456810      0.731486224
    both    37849418    0.76392424
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use CPU or GPU
# model = torch.load('../best_models/crnn_best_both.pt', map_location=device)
model = torch.load('../archived/ViFi/occlusion/board/crop/both/best_models/crnn_best_both.pt', map_location=device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

input_tensor_video = torch.randn(1, 32, 1, 64, 64).to(device)
input_tensor_csi = torch.randn(1, 3, 30, 500).to(device)
flops, params = profile(model, inputs=(input_tensor_video, input_tensor_csi))
gflops = flops / 1e9
print(f"GFLOPs: {gflops}")