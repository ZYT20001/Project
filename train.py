import os
import json
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from collections import defaultdict

from lcd.dataset import CrossTripletDataset
from lcd.models import *
from lcd.losses import *

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="path to the json config file")
parser.add_argument("--logdir", help="path to the log directory")
args = parser.parse_args()

config = args.config
logdir = args.logdir
args = json.load(open(config))

if not os.path.exists(logdir):
    os.mkdir(logdir)

fname = os.path.join(logdir, "config.json")
with open(fname, "w") as fp:
    json.dump(args, fp, indent=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CrossTripletDataset(args["root"], split="train")
loader = data.DataLoader(
    dataset,
    batch_size=args["batch_size"],
    num_workers=args["num_workers"],
    pin_memory=True,
    shuffle=True,
)

patchnet = PatchNetAutoencoder(embedding_size=128, normalize=True)
pointnet = PointNetAutoencoder(
    embedding_size=128,
    input_channels=6,
    output_channels=6,
    normalize=True,
)

# 加载预训练模型
model_path = '/Users/zezhang/Desktop/lcd-master/logs/LCD-D128/model.pth'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
patchnet.load_state_dict(checkpoint['patchnet'])
pointnet.load_state_dict(checkpoint['pointnet'])

# 冻结PatchNetAutoencoder的编码器部分
for name, param in patchnet.named_parameters():
    if "encoder.conv1" in name or "encoder.conv2" in name or "encoder.conv3" in name or "encoder.conv4" in name:
        param.requires_grad = False
    if "encoder.bn1" in name or "encoder.bn2" in name or "encoder.bn3" in name or "encoder.bn4" in name:
        param.requires_grad = False

# 冻结PointNetAutoencoder的编码器部分
for name, param in pointnet.named_parameters():
    if "encoder.stn1" in name or "encoder.stn2" in name or "encoder.mlp1" in name:
        param.requires_grad = False

patchnet.to(device)
pointnet.to(device)

# 只训练未冻结的参数
parameters = list(filter(lambda p: p.requires_grad, patchnet.parameters())) + \
             list(filter(lambda p: p.requires_grad, pointnet.parameters()))
optimizer = optim.SGD(
    parameters,
    lr=args["learning_rate"],
    momentum=args["momentum"],
    weight_decay=args["weight_decay"],
)

criterion = {
    "mse": MSELoss(),
    "chamfer": ChamferLoss(args["output_channels"]),
    "triplet": HardTripletLoss(args["margin"], args["hardest"]),
}
criterion["mse"].to(device)
criterion["chamfer"].to(device)
criterion["triplet"].to(device)

best_loss = np.Inf
loss_threshold = 1e5 

for epoch in range(args["epochs"]):
    start = datetime.datetime.now()
    scalars = defaultdict(list)
    valid_batches = 0  # 记录有效的 batch 数量

    for i, batch in enumerate(loader):
        x = [x.to(device).float() for x in batch]
        y0, z0 = pointnet(x[0])
        y1, z1 = patchnet(x[1])

        loss_r = 0
        loss_d = 0
        loss_r += args["alpha"] * criterion["mse"](x[1], y1)
        loss_r += args["beta"] * criterion["chamfer"](x[0], y0)
        loss_d += args["gamma"] * criterion["triplet"](z0, z1)
        loss = loss_d + loss_r

        if loss.item() > loss_threshold:
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scalars["loss"].append(loss)
        scalars["loss_d"].append(loss_d)
        scalars["loss_r"].append(loss_r)
        valid_batches += 1  

        now = datetime.datetime.now()
        log = "{} | Batch [{:04d}/{:04d}] | loss: {:.4f} |"
        log = log.format(now.strftime("%c"), valid_batches, len(loader), loss.item())
        print(log)

    # Summary after each epoch
    summary = {}
    now = datetime.datetime.now()
    duration = (now - start).total_seconds()
    log = "> {} | Epoch [{:04d}/{:04d}] | duration: {:.1f}s |"
    log = log.format(now.strftime("%c"), epoch, args["epochs"], duration)
    for m, v in scalars.items():
        summary[m] = torch.stack(v).mean()
        log += " {}: {:.4f} |".format(m, summary[m].item())

    fname = os.path.join(logdir, "checkpoint_{:04d}.pth".format(epoch))
    print("> Saving model to {}...".format(fname))
    model = {"pointnet": pointnet.state_dict(), "patchnet": patchnet.state_dict()}
    torch.save(model, fname)

    if summary["loss"] < best_loss:
        best_loss = summary["loss"]
        fname = os.path.join(logdir, "model.pth")
        print("> Saving model to {}...".format(fname))
        model = {"pointnet": pointnet.state_dict(), "patchnet": patchnet.state_dict()}
        torch.save(model, fname)
    log += " best: {:.4f} |".format(best_loss)

    fname = os.path.join(logdir, "train.log")
    with open(fname, "a") as fp:
        fp.write(log + "\n")

    print(log)
    print("--------------------------------------------------------------------------")
