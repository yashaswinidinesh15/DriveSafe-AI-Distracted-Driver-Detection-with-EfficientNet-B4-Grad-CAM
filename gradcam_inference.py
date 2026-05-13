"""
DriveSafe-AI — Grad-CAM Inference
Usage: python src/gradcam_inference.py --image path/to/img.jpg --checkpoint checkpoints/best_efficientnet_b4.pt
"""
import argparse, os
import numpy as np
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt, matplotlib.gridspec as gs
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from train import DriveSafeNet, CLASS_NAMES, NUM_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN   = np.array([0.485,0.456,0.406])
STD    = np.array([0.229,0.224,0.225])

def preprocess(path):
    tfm = A.Compose([A.Resize(224,224), A.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225]), ToTensorV2()])
    img = np.array(Image.open(path).convert("RGB"))
    return tfm(image=img)["image"], np.clip(img/255.,0,1)

def gradcam(model, tensor, cls=None):
    model.eval()
    inp = tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)
    out = model(inp)
    if cls is None: cls = out.argmax(1).item()
    model.zero_grad(); out[0,cls].backward()
    grads = model.gradients; acts = model.activations
    if grads is None: return None, cls, out
    w   = grads.mean(dim=[2,3], keepdim=True)
    cam = F.relu((w*acts).sum(1, keepdim=True))
    cam = F.interpolate(cam,(224,224),mode="bilinear",align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam-cam.min())/(cam.max()-cam.min()+1e-8)
    return cam, cls, out

def overlay(orig, cam):
    h = np.uint8(plt.get_cmap("jet")(cam)[:,:,:3]*255)
    return cv2.addWeighted(np.uint8(orig*255),0.55,h,0.45,0)/255.

def run(img_path, ckpt, out_path="results/gradcam.png"):
    os.makedirs("results", exist_ok=True)
    model = DriveSafeNet(NUM_CLASSES, pretrained=False).to(DEVICE)
    ck    = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(ck["model"] if "model" in ck else ck)

    tensor, orig = preprocess(img_path)
    cam, pred, logits = gradcam(model, tensor)
    probs = F.softmax(logits,1).squeeze().detach().cpu().numpy()
    short = ["Safe","Txt-R","Ph-R","Txt-L","Ph-L","Radio","Drink","Reach","Makeup","Talk"]

    fig  = plt.figure(figsize=(18,5)); fig.patch.set_facecolor("#0d1117")
    grid = gs.GridSpec(1,4,figure=fig,wspace=.05)
    for i,(title,img,cm) in enumerate(zip(
        ["Original","Heatmap","Overlay"],
        [orig, cam if cam is not None else orig, overlay(orig,cam) if cam is not None else orig],
        [None,"jet",None]
    )):
        ax=fig.add_subplot(grid[i]); ax.imshow(img,cmap=cm); ax.axis("off")
        ax.set_title(title,color="white",fontsize=11,pad=6)

    ax4=fig.add_subplot(grid[3]); ax4.set_facecolor("#161b22")
    colors=["#4ade80" if i==pred else "#3b82f6" for i in range(10)]
    ax4.barh(short[::-1],probs[::-1]*100,color=colors[::-1],edgecolor="#0d1117",height=.65)
    ax4.set_xlim(0,115); ax4.set_xlabel("Confidence (%)",color="white",fontsize=9)
    ax4.set_title("Class Probabilities",color="white",fontsize=11,pad=6)
    ax4.tick_params(colors="white",labelsize=8)
    for s in ax4.spines.values(): s.set_edgecolor("#30363d")

    fig.suptitle(f"DriveSafe-AI  ▸  {CLASS_NAMES[pred]}  ({probs[pred]*100:.1f}%)",
                 color="white",fontsize=14,fontweight="bold",y=1.02)
    plt.savefig(out_path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")
    print(f"Prediction: {CLASS_NAMES[pred]} — {probs[pred]*100:.1f}%")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image",      required=True)
    p.add_argument("--checkpoint", default="checkpoints/best_efficientnet_b4.pt")
    p.add_argument("--out",        default="results/gradcam.png")
    args = p.parse_args()
    run(args.image, args.checkpoint, args.out)
