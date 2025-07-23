import os
import torch
from lxml import etree
from clip.utils import parse_xml_to_dict, scoremap2bbox
from clip.clip_text import class_names, new_class_names, new_class_names_cod, class_names_coco, new_class_names_coco
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from pytorch_grad_cam.utils.image import scale_cam_image
import cv2
import numpy as np


class ClipOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]




def generate_clip_fts(image, model, require_all_fts=True):
    model = model.cuda()

    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    h, w = image.shape[-2], image.shape[-1]
    image = image.cuda()
    
    image_features_all, attn_weight_list = model.encode_image(image, h, w, require_all_fts=require_all_fts)

    # print(len(image_features_all))
        
    return image_features_all, attn_weight_list


def generate_trans_mat(aff_mask, attn_weight, grayscale_cam):
    aff_mask = aff_mask.view(1,grayscale_cam.shape[0] * grayscale_cam.shape[1])
    aff_mat = attn_weight

    trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

    for _ in range(2):
        trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
    trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

    for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)

    trans_mat = trans_mat * aff_mask
    
    return trans_mat


def compute_trans_mat(attn_weight):
    aff_mat = attn_weight

    trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

    for _ in range(2):
        trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
    trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

    for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)

    trans_mat = trans_mat

    return trans_mat


def generate_trans_mat_seg(aff_mask, attn_weight, grayscale_cam):
    aff_mask = aff_mask.view(1,grayscale_cam.shape[0] * grayscale_cam.shape[1])
    aff_mat = attn_weight

    trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

    for _ in range(2):
        trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
    trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

    for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)

    trans_mat = trans_mat * aff_mask
    
    return trans_mat

################################################################
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def apply_canny_edge_detection(grayscale_img):
    # 使用Canny边缘检测
    edges = cv2.Canny(grayscale_img.astype(np.uint8), 100, 200)
    return edges

def enhance_grayscale_with_edges(grayscale_cam):
    # 将灰度图转换为numpy格式
    grayscale_img = grayscale_cam.squeeze() # 假设输入是 (1, 20, 20)
    
    # 应用Canny边缘检测
    edge_map = apply_canny_edge_detection(grayscale_img)
    
    # 可以将边缘图与灰度图进行加权融合
    alpha = 0.5
    enhanced_image = grayscale_img + alpha * edge_map
    
    # 将增强后的图像转换为tensor格式，准备送入模型（保持相同尺寸）
    enhanced_image = np.expand_dims(enhanced_image, axis=0)  # (1, 1, 20, 20)
    
    return enhanced_image
#########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
import numpy as np
import torch
import cv2

from PIL import Image
import numpy as np
import cv2

def enhance_cam_with_image_edge_numpy(image_pil: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    使用图片的边缘信息增强 Grad-CAM（边缘更连续完整）

    Args:
        image_pil (PIL.Image.Image): 原始图像（通过 Image.open 打开）
        cam (np.ndarray): 原始 CAM，形状为 (1, 32, 32)，类型为 float32
        alpha (float): 边缘融合权重，默认 0.5

    Returns:
        np.ndarray: 增强后的 CAM，形状为 (1, 32, 32)，类型为 float32
    """
    assert isinstance(cam, np.ndarray), "CAM must be a NumPy array"
    # assert cam.shape == (1, 32, 32), "CAM must be of shape (1, 32, 32)"
    assert cam.dtype == np.float32, "CAM must be float32"

    _, h, w = cam.shape

    # Step 1: 图像转灰度
    gray_img = image_pil.convert("L")
    gray_np = np.array(gray_img)  # shape=(H, W), dtype=uint8

    # Step 2: 提取边缘
    edges = cv2.Canny(gray_np, threshold1=100, threshold2=200)  # uint8 [0,255]
    edges = edges.astype(np.float32) / 255.0  # 归一化到 [0, 1]

    # Step 3: Resize 边缘图为 (32, 32)
    edge_resized = cv2.resize(edges, (w, h), interpolation=cv2.INTER_LINEAR)

    # Step 4: 归一化边缘图（可选）
    if edge_resized.max() > edge_resized.min():
        edge_resized = (edge_resized - edge_resized.min()) / (edge_resized.max() - edge_resized.min() + 1e-8)

    # Step 5: 融合 CAM
    enhanced_cam = cam[0] + alpha * edge_resized  # shape: (32, 32)
    enhanced_cam = (enhanced_cam - enhanced_cam.min()) / (enhanced_cam.max() - enhanced_cam.min() + 1e-8)

    # Step 6: 返回 numpy float32 格式
    return enhanced_cam[np.newaxis, :, :].astype(np.float32)  # shape: (1, 32, 32)

########################################################################################################################
def perform_single_voc_cam(img_path, image, image_features, attn_weight_list, seg_attn, bg_text_features,
                       fg_text_features, cam,  mode='train', require_seg_trans=False):
    

    bg_text_features = bg_text_features.cuda()
    fg_text_features = fg_text_features.cuda()

    ori_image = Image.open(img_path)
    # print("@@@@@@@@@@@@@@@@@@@@",ori_image.size)
    ori_height, ori_width = np.asarray(ori_image).shape[:2]
    # print(ori_height, ori_width)
    label_id_list = np.unique(ori_image)
    label_id_list = (label_id_list - 1).tolist()
    if 255 in label_id_list:
        label_id_list.remove(255)
    if 254 in label_id_list:
        label_id_list.remove(254)
    

    label_list = []
    for lid in label_id_list:
        label_list.append(new_class_names[int(lid)])
        # label_list.append(new_class_names_cod[int(lid)])
    label_id_list = [int(lid) for lid in label_id_list]

    image = image.unsqueeze(0)
    h, w = image.shape[-2], image.shape[-1]

    highres_cam_to_save = []
    keys = []

    cam_refined_list = []

    bg_features_temp = bg_text_features.cuda()  # [bg_id_for_each_image[im_idx]].to(device_id)
    fg_features_temp = fg_text_features[label_id_list].cuda()
    text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
    input_tensor = [image_features, text_features_temp.cuda(), h, w]

    for idx, label in enumerate(label_list):
        # label_index = new_class_names_cod.index(label)
        label_index = new_class_names.index(label)
        keys.append(label_index)
        targets = [ClipOutputTarget(label_list.index(label))]
        grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
                                                                targets=targets,
                                                                target_size=None)  # (ori_width, ori_height))
        # print(grayscale_cam.dtype)
    
################################################################
        # print("@@@@@@@@@@@@@@@@@@@@@@@@",fts_all.dtype,grayscale_cam.dtype)

        # grayscale_cam = enhance_grayscale_with_edges(grayscale_cam)

        grayscale_cam=enhance_cam_with_image_edge_numpy(ori_image,grayscale_cam)

    

        # print(grayscale_cam.shape,grayscale_cam.dtype)
#############################################################

        grayscale_cam = grayscale_cam[0, :]


        grayscale_cam_highres = cv2.resize(grayscale_cam, (w, h))

        highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))

        if idx == 0:
            if require_seg_trans == True:
                attn_weight = torch.cat([attn_weight_list, attn_weight_last], dim=0)
                attn_weight = attn_weight[:, 1:, 1:][-6:] #-8

                # attn_diff = torch.abs(seg_attn - attn_weight)
                attn_diff = seg_attn - attn_weight
                attn_diff = torch.sum(attn_diff.flatten(1), dim=1)
                diff_th = torch.mean(attn_diff)

                attn_mask = torch.zeros_like(attn_diff)
                attn_mask[attn_diff <= diff_th] = 1

                attn_mask = attn_mask.reshape(-1, 1, 1)
                attn_mask = attn_mask.expand_as(attn_weight)
                attn_weight = torch.sum(attn_mask*attn_weight, dim=0) / (torch.sum(attn_mask, dim=0)+1e-5)

                attn_weight = attn_weight.detach()
                attn_weight = attn_weight * seg_attn.squeeze(0).detach()
            else:
                attn_weight = torch.cat([attn_weight_list, attn_weight_last], dim=0)
                attn_weight = attn_weight[:, 1:, 1:][-8:]
                attn_weight = torch.mean(attn_weight, dim=0)  # (1, hw, hw)
                attn_weight = attn_weight.detach()
            _trans_mat = compute_trans_mat(attn_weight)
        _trans_mat = _trans_mat.float()

        box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
        aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1])).cuda()
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            aff_mask[y0_:y1_, x0_:x1_] = 1

        aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
        trans_mat = _trans_mat*aff_mask

        cam_to_refine = torch.FloatTensor(grayscale_cam).cuda()
        cam_to_refine = cam_to_refine.view(-1, 1)

        cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h // 16, w // 16)
        cam_refined_list.append(cam_refined)

    if mode == 'train':
        return cam_refined_list, keys, w, h
    else:
        return cam_refined_list, keys, ori_width, ori_height




def generate_cam_label(cam_refined_list, keys, w, h):
    refined_cam_to_save = []
    refined_cam_all_scales = []
    for cam_refined in cam_refined_list:
        cam_refined = cam_refined.cpu().numpy().astype(np.float32)
        cam_refined_highres = scale_cam_image([cam_refined], (w, h))[0]
        refined_cam_to_save.append(torch.tensor(cam_refined_highres))

    keys = torch.tensor(keys)

    refined_cam_all_scales.append(torch.stack(refined_cam_to_save,dim=0))

    refined_cam_all_scales = refined_cam_all_scales[0]
    
    return {'keys': keys.numpy(), 'refined_cam':refined_cam_all_scales}




def perform_single_coco_cam(img_path, image, image_features, attn_weight_list, seg_attn, bg_text_features,
                        fg_text_features, cam, mode='train', require_all_fts=True, require_seg_trans=False):
    bg_text_features = bg_text_features.cuda()
    fg_text_features = fg_text_features.cuda()

    ori_image = Image.open(img_path)
    ori_height, ori_width = np.asarray(ori_image).shape[:2]
    label_id_list = np.unique(ori_image)
    label_id_list = (label_id_list-1).tolist()
    if 255 in label_id_list:
        label_id_list.remove(255)
    if 254 in label_id_list:
        label_id_list.remove(254)

    label_list = []
    for lid in label_id_list:
        label_list.append(new_class_names_coco[int(lid)])
    label_id_list = [int(lid) for lid in label_id_list]

    image = image.unsqueeze(0)
    h, w = image.shape[-2], image.shape[-1]

    highres_cam_to_save = []
    keys = []

    cam_refined_list = []

    bg_features_temp = bg_text_features.cuda()  # [bg_id_for_each_image[im_idx]].to(device_id)
    fg_features_temp = fg_text_features[label_id_list].cuda()
    text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
    input_tensor = [image_features, text_features_temp.cuda(), h, w]

    for idx, label in enumerate(label_list):
        label_index = new_class_names_coco.index(label)
        keys.append(label_index)
        targets = [ClipOutputTarget(label_list.index(label))]
        grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
                                                                targets=targets,
                                                                target_size=None)  # (ori_width, ori_height))

        grayscale_cam = grayscale_cam[0, :]

        grayscale_cam_highres = cv2.resize(grayscale_cam, (w, h))
        highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))

        # if idx == 0:
        #     attn_weight = torch.cat([attn_weight_list, attn_weight_last], dim=0)
        #     attn_weight = attn_weight[:, 1:, 1:][-8:]
        #     attn_weight = torch.mean(attn_weight, dim=0)  # (1, hw, hw)
        #     attn_weight = attn_weight.detach()
        #     if require_seg_trans == True:
        #         attn_weight = attn_weight * seg_attn.squeeze(0).detach()
        if idx == 0:
            if require_seg_trans == True:
                attn_weight = torch.cat([attn_weight_list, attn_weight_last], dim=0)
                attn_weight = attn_weight[:, 1:, 1:][-10:]  # -8

                # attn_diff = torch.abs(seg_attn - attn_weight)
                attn_diff = seg_attn - attn_weight
                attn_diff = torch.sum(attn_diff.flatten(1), dim=1)
                diff_th = torch.mean(attn_diff)

                attn_mask = torch.zeros_like(attn_diff)
                attn_mask[attn_diff <= diff_th] = 1

                attn_mask = attn_mask.reshape(-1, 1, 1)
                attn_mask = attn_mask.expand_as(attn_weight)
                attn_weight = torch.sum(attn_mask * attn_weight, dim=0) / (torch.sum(attn_mask, dim=0) + 1e-5)

                attn_weight = attn_weight.detach()
                attn_weight = attn_weight * seg_attn.squeeze(0).detach()
            else:
                attn_weight = torch.cat([attn_weight_list, attn_weight_last], dim=0)
                attn_weight = attn_weight[:, 1:, 1:][-8:]
                attn_weight = torch.mean(attn_weight, dim=0)  # (1, hw, hw)
                attn_weight = attn_weight.detach()
            _trans_mat = compute_trans_mat(attn_weight)
        _trans_mat = _trans_mat.float()

        box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.7, multi_contour_eval=True)
        aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1]))
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            aff_mask[y0_:y1_, x0_:x1_] = 1

        aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
        trans_mat = _trans_mat.cuda() * aff_mask.cuda()

        cam_to_refine = torch.FloatTensor(grayscale_cam).cuda()
        cam_to_refine = cam_to_refine.view(-1, 1)

        cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h // 16, w // 16)
        cam_refined_list.append(cam_refined)

    if mode == 'train':
        return cam_refined_list, keys, w, h
    else:
        return cam_refined_list, keys, ori_width, ori_height


