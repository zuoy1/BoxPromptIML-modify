import os
import cv2
import numpy as np
import torch
import warnings
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils.data_val import get_loader, test_dataset
import torch.nn.functional as F
from datetime import datetime
from PIL import Image
from boxpromptiml import BoxPromptIML

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()


def pil_to_save(array, name, target_size=None, save_path='./tmp/'):
    array = array * 255
    pil_image = Image.fromarray(array.astype(np.uint8))

    if target_size is not None:
        if isinstance(target_size, torch.Tensor):
            target_size = tuple(target_size.squeeze().cpu().numpy()[::-1])
        # resize 
        pil_image = pil_image.resize(target_size, Image.BILINEAR)

    real_path = os.path.join(save_path, name)
    pil_image.save(real_path)

def metric(premask, groundtruth):
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    if np.sum(cross) + np.sum(union) == 0:
        iou = 1
    return f1, iou

def get_masks(test_loader, model, save_path, device, datesetname):
    """
    validation function
    """
    global best_mae, best_epoch
    # datesetname = save_path.rsplit('/')[-2]
    # print(datesetname)

    model.eval()
    with torch.no_grad():
        # for i in range(test_loader.size):
        for i in tqdm(range(test_loader.size), desc=f'{datesetname} generate masks'):
            image, _, gt, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.to(device)

            res = model(image)
            res = F.interpolate(res, size=gt.shape, mode='bilinear')
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            res[res > 0.5] = 1
            res[res <= 0.5] = 0

            pil_to_save(res, name, save_path=save_path)



def get_gt_path(file, path_gt):
    if "NC16" in path_gt:
        return path_gt + file
    elif "C1" in path_gt:
        return path_gt + file[:-4] + '_gt.png'
    elif "Coverage" in path_gt:
        return path_gt + file[:-5] + 'forged.tif'
    elif "Columbia" in path_gt:
        return path_gt + file[:-4] + '_edgemask.png'
    elif "Imd" in path_gt:
        return path_gt + file[:-4] + '_mask.png'
    elif "Korus" in path_gt:
        return path_gt + file[:-4] + '.PNG'
    elif "Coco" in path_gt or "In-the-Wild" in path_gt:
        return path_gt + file[:-4] + '.png'
    elif "DSO" in path_gt:
        return path_gt + file[:-4] + '_gt.png'
    else:
        return path_gt + file


def process_single_image(args):
    file, path_pre, path_gt = args
    try:
        pre = cv2.imread(path_pre + '/' + file)
        gt_path = get_gt_path(file, path_gt)
        gt = cv2.imread(gt_path)

        if pre is None or gt is None:
            return None

        H, W, C = pre.shape
        Hg, Wg, C = gt.shape
        if H != Hg or W != Wg:
            gt = cv2.resize(gt, (W, H))
            gt[gt > 127] = 255
            gt[gt <= 127] = 0

        # AUC
        auc = None
        if np.max(gt) != np.min(gt):
            auc = roc_auc_score((gt.reshape(H * W * C) / 255).astype('int'),
                                pre.reshape(H * W * C) / 255.)

        pre[pre > 127] = 255
        pre[pre <= 127] = 0
        f1, iou = metric(pre / 255, gt / 255)
        return (auc, f1, iou)
    except Exception as e:
        print(f"[Error] {file}: {e}")
        return None


def evaluate(path_pre, path_gt, dataset_name, record_txt, epoch, model_name=''):
    flist = sorted(os.listdir(path_pre))
    args_list = [(f, path_pre, path_gt) for f in flist]

    with Pool(processes=min(32, cpu_count())) as pool:
        results = list(tqdm(pool.imap(process_single_image, args_list), total=len(args_list), desc=f'epoch:{epoch} dataset:{dataset_name}'))

    aucs, f1s, ious = [], [], []
    num_valid = 0
    for r in results:
        if r is not None:
            auc, f1, iou = r
            if auc is not None:
                aucs.append(auc)
            f1s.append(f1)
            ious.append(iou)
            num_valid += 1

    print(f"{dataset_name}, epoch:{epoch}")
    print('Evaluation: AUC: %5.4f, F1: %5.4f, IOU: %5.4f' % (
        np.mean(aucs), np.mean(f1s), np.mean(ious)))
    print(f'Evaled {num_valid} pictures')

    with open(record_txt, "a") as f:
        f.write(f'{datetime.now()} {model_name} {dataset_name} epoch:{epoch}')
        f.write(' AUC: %5.4f, F1: %5.4f, IOU: %5.4f\n' %
                (np.mean(aucs), np.mean(f1s), np.mean(ious)))
    return np.mean(aucs), np.mean(f1s), np.mean(ious)

if __name__ == "__main__":

    device = torch.device('cuda:0')
    model = BoxPromptIML()
    checkpoints_path='./checkpoints/checkpoint-20.pth'
    checkpoints = torch.load(checkpoints_path, map_location= device, weights_only=False)
    model.load_state_dict(checkpoints)

    model = model.to(device)
    model.eval()

    epoch = checkpoints_path.rsplit('/')[-1].rsplit('-')[-1]
    epoch = epoch.split('.')[0]
    epoch = int(epoch)

    model_name = checkpoints_path.rsplit('/')[-2]

    image_root = "your test image root dir"
    gt_root = "your test image gt root dir"

    test_datasets = ['C1', 'Coverage', 'Columbia', 'NC16', 'Imd', 'Korus', 'In-the-Wild', 'DSO', 'CocoGlide']
  
    save_path = os.path.join('./masks/', model_name, f'epoch_{epoch}')
    record_txt = "./test_data.txt"

    f1_in = []
    f1_out = []
    

    for i in range(len(test_datasets)):
        dataset = test_datasets[i]
        recent_image_root = os.path.join(image_root, dataset, 'Tp/')
        recent_gt_root = os.path.join(gt_root, dataset, 'Gt/')

        test_loader = test_dataset(image_root=recent_image_root,
                            gt_root=recent_gt_root,
                            testsize=224)
    
        recent_save_path = os.path.join(save_path, dataset)
        os.makedirs(recent_save_path, exist_ok=True)
        get_masks(test_loader, model=model, save_path=recent_save_path, device=device, datesetname=dataset)

        aucs, f1s, ious = evaluate(recent_save_path, recent_gt_root, dataset, record_txt, epoch, model_name=model_name)

        if i < 4:
            f1_in.append(f1s)
        else:
            f1_out.append(f1s)

    with open(record_txt, "a") as f:
        f.write(f'f1_in:{np.mean(f1_in):.4f} f1_out:{np.mean(f1_out):.4f} \n')
    