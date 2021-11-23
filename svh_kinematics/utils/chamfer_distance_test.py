import torch
from chamfer_distance import ChamferDistance
import time
import kaolin
device = 'cuda:0'
chamfer_dist = ChamferDistance()#.to(device)

p1 = torch.rand([256, 1500, 3]).to(device)
p2 = torch.rand([256, 500, 3]).to(device)

s = time.time()
dist1, dist2, idx1, idx2 = chamfer_dist(p1, p2)
loss = (torch.mean(dist1)) + (torch.mean(dist2))

torch.cuda.synchronize()
print(f"Time: {time.time() - s} seconds")
print(f"Loss: {loss}")

# distance = kaolin.metrics.pointcloud.chamfer_distance(p1, p2)
# print(distance)