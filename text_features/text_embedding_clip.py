import sys
import os
proj_dir = os.path.abspath(os.getcwd())
print("proj_dir: ", proj_dir)
sys.path.append(proj_dir)

import torch
from src.classnames import *

from clip import clip
from src.imagenet_templates import IMAGENET_TEMPLATES

def load_clip_to_cpu():
    # backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone_name = 'ViT-L/14'
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    print("clip_model_path:", model_path)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def build_clip_model(class_names, file_name):
    classnames = class_names

    clip_model = load_clip_to_cpu()
    clip_model.cuda()

    for params in clip_model.parameters():
        params.requires_grad_(False)

    templates = IMAGENET_TEMPLATES

    num_temp = len(templates)
    print(f"Prompt ensembling (n={num_temp})")

    all_text_features = []
    for i, temp in enumerate(templates):
        if len(classnames)==200: # for CUB-200
            prompts = [temp.format(c.split('.')[-1].replace("_", " ")) for c in classnames]
        #elif len(classnames)==100: # for AirCraft
        #    prompts = [temp.format("aircraft of type " + c.replace("_", " ")) for c in classnames]
        else:
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        text_features = clip_model.encode_text(prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # print("debug size", text_features.size())
        all_text_features.append(text_features)
    all_text_features = torch.stack(tuple(all_text_features),dim=1)
    print(all_text_features.size())
    all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)

    text_features = all_text_features
    torch.save(all_text_features.cpu(), file_name)
    print('text embedding '+file_name + ' is saved.')

    return

def main():
    # -----------------------------------------------------------------------
    # Setting up here.
    classnames = CLASSES_AirCraft # Set classnames
    save_name = 'aircraft_clipL.pt'
    # -----------------------------------------------------------------------
    
    build_clip_model(class_names=classnames, file_name=save_name)
    return

if __name__ == '__main__':
    main()
