import os
import cv2
import numpy as np
import torch
import warnings
from pathlib import Path
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

def get_masks(test_loader, model, save_path, device, datesetname, probability_path=None):
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

            probability = res.copy()
            res[res > 0.5] = 1
            res[res <= 0.5] = 0

            # Keep the binary mask used for the paper's fixed-threshold F1.
            pil_to_save(res, name, save_path=save_path)
            # Save the continuous score separately so AUC is not computed from
            # an already thresholded mask.
            if probability_path is not None:
                os.makedirs(probability_path, exist_ok=True)
                pil_to_save(probability, name, save_path=probability_path)



def get_gt_path(file, path_gt):
    path_gt = os.fspath(path_gt)
    if "NC16" in path_gt:
        return os.path.join(path_gt, file)
    elif "C1" in path_gt:
        return os.path.join(path_gt, Path(file).stem + '_gt.png')
    elif "Coverage" in path_gt:
        stem = Path(file).stem
        return os.path.join(path_gt, stem[:-1] + 'forged.tif')
    elif "Columbia" in path_gt:
        return os.path.join(path_gt, Path(file).stem + '_edgemask.png')
    elif "Imd" in path_gt:
        return os.path.join(path_gt, Path(file).stem + '_mask.png')
    elif "Korus" in path_gt:
        return os.path.join(path_gt, Path(file).stem + '.PNG')
    elif "Coco" in path_gt or "In-the-Wild" in path_gt:
        return os.path.join(path_gt, Path(file).stem + '.png')
    elif "DSO" in path_gt:
        return os.path.join(path_gt, Path(file).stem + '_gt.png')
    else:
        return os.path.join(path_gt, file)


def process_single_image(args):
    file, path_pre, path_gt = args[:3]
    path_probability = args[3] if len(args) > 3 else None
    try:
        pre = cv2.imread(os.path.join(path_pre, file), cv2.IMREAD_GRAYSCALE)
        gt_path = get_gt_path(file, path_gt)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if pre is None or gt is None:
            return None

        H, W = pre.shape
        Hg, Wg = gt.shape
        if (H, W) != (Hg, Wg):
            raise ValueError(f'Prediction/GT size mismatch: {file}: {(H, W)} vs {(Hg, Wg)}')

        # AUC uses the continuous score map when available. F1/IoU use the
        # saved binary mask at the paper's fixed threshold of 0.5.
        probability = None
        if path_probability is not None:
            probability_path = os.path.join(path_probability, file)
            probability_image = cv2.imread(probability_path, cv2.IMREAD_GRAYSCALE)
            if probability_image is None or probability_image.shape != (H, W):
                raise ValueError(f'Missing or mismatched score map: {probability_path}')
            probability = probability_image.astype(np.float32) / 255.0

        auc = None
        if probability is not None and np.max(gt) != np.min(gt):
            auc = roc_auc_score((gt.reshape(-1) / 255).astype('int'), probability.reshape(-1))

        pre[pre > 127] = 255
        pre[pre <= 127] = 0
        f1, iou = metric(pre / 255, gt / 255)
        return (auc, f1, iou)
    except Exception as e:
        print(f"[Error] {file}: {e}")
        return None


def evaluate(path_pre, path_gt, dataset_name, record_txt, epoch, model_name='', path_probability=None):
    flist = sorted(f for f in os.listdir(path_pre) if os.path.isfile(os.path.join(path_pre, f)))
    args_list = [(f, path_pre, path_gt, path_probability) for f in flist]

    # Large NIST masks make many simultaneous AUC sorts memory intensive.
    with Pool(processes=min(2, cpu_count())) as pool:
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

    if not flist or num_valid != len(flist):
        raise RuntimeError(f'{dataset_name}: only {num_valid}/{len(flist)} valid predictions; refusing partial metrics')

    print(f"{dataset_name}, epoch:{epoch}")
    auc_mean = np.mean(aucs) if aucs else float('nan')
    print('Evaluation: AUC: %5.4f, F1: %5.4f, IOU: %5.4f' % (
        auc_mean, np.mean(f1s), np.mean(ious)))
    print(f'Evaled {num_valid} pictures')

    with open(record_txt, "a") as f:
        f.write(f'{datetime.now()} {model_name} {dataset_name} epoch:{epoch}')
        f.write(' AUC: %5.4f, F1: %5.4f, IOU: %5.4f\n' %
                (auc_mean, np.mean(f1s), np.mean(ious)))
    return auc_mean, np.mean(f1s), np.mean(ious)


def validate_dataset_layout(dataset, image_root, gt_root, expected_count):
    """Fail before inference if any image/GT pair is missing or misordered."""
    image_root = os.path.abspath(image_root)
    gt_root = os.path.abspath(gt_root)
    image_files = sorted(
        f for f in os.listdir(image_root)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')))
    gt_files = sorted(
        f for f in os.listdir(gt_root)
        if f.lower().endswith(('.png', '.tif', '.bmp', '.jpg', '.jpeg')))
    if len(image_files) != expected_count or len(gt_files) != expected_count:
        raise RuntimeError(
            f'{dataset}: expected {expected_count} images and GTs, '
            f'found {len(image_files)} images and {len(gt_files)} GTs')

    expected_gt = {}
    for image_name in image_files:
        output_name = Path(image_name).stem + '.png' if image_name.lower().endswith('.jpg') else image_name
        expected_gt[image_name] = os.path.abspath(get_gt_path(output_name, gt_root))
        if not os.path.isfile(expected_gt[image_name]):
            raise RuntimeError(f'{dataset}: missing GT for {image_name}: {expected_gt[image_name]}')
        with Image.open(os.path.join(image_root, image_name)) as image, Image.open(expected_gt[image_name]) as gt:
            image.load(); gt.load()
            if image.size != gt.size:
                raise RuntimeError(f'{dataset}: size mismatch {image_name}: {image.size} vs {gt.size}')
            values = np.unique(np.asarray(gt.convert('L')))
            if not np.array_equal(values, [0, 255]):
                raise RuntimeError(f'{dataset}: nonbinary or empty GT for {image_name}: {values}')

    actual_gt = {os.path.abspath(os.path.join(gt_root, f)) for f in gt_files}
    expected_gt_set = set(expected_gt.values())
    if actual_gt != expected_gt_set:
        raise RuntimeError(f'{dataset}: GT directory contains unexpected or missing files')

    loader = test_dataset(image_root + os.sep, gt_root + os.sep, 224)
    if loader.size != expected_count or len(loader.gts) != expected_count:
        raise RuntimeError(f'{dataset}: actual loader does not recognize all files')
    for image_path, gt_path in zip(loader.images, loader.gts):
        image_name = os.path.basename(image_path)
        output_name = Path(image_name).stem + '.png' if image_name.lower().endswith('.jpg') else image_name
        if os.path.abspath(gt_path) != expected_gt[image_name]:
            raise RuntimeError(f'{dataset}: loader pairing mismatch for {image_name}')
    return loader

if __name__ == "__main__":

    # Use the first GPU when available, while keeping evaluation runnable on CPU.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BoxPromptIML()
    checkpoints_path='/hy-tmp/runs/boxprompt_formal_20261016_122447/checkpoints/checkpoint-20.pth'
    checkpoints = torch.load(checkpoints_path, map_location= device, weights_only=False)
    model.load_state_dict(checkpoints)

    model = model.to(device)
    model.eval()

    epoch = checkpoints_path.rsplit('/')[-1].rsplit('-')[-1]
    epoch = epoch.split('.')[0]
    epoch = int(epoch)

    model_name = os.path.basename(os.path.dirname(os.path.dirname(checkpoints_path)))

    # Paper-scale local split.  Each dataset contains Tp/ and Gt/ subdirectories.
    image_root = '/hy-tmp/datasets/processed/boxprompt_paper_scale_20261016/test'
    gt_root = image_root

    # All four in-distribution datasets, in the paper's table order.
    test_datasets = ['NC16', 'C1', 'Columbia', 'Coverage']
    print('IND evaluation: NC16 (150), C1 (920), Columbia (50), Coverage (30). '
          'f1_in is the unweighted mean of the four dataset F1 scores.')
    print('Local reconstructed split: NIST16 has shared source images across train/test; '
          'see the data audit report before comparing with the paper.')
  
    run_path = Path('/hy-tmp/runs') / ('boxprompt_eval_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run_path.mkdir(exist_ok=False)
    save_path = str(run_path / 'masks')
    record_txt = str(run_path / 'metrics.txt')
    print(f'Evaluation outputs: {run_path}')

    expected_counts = {'NC16': 150, 'C1': 920, 'Columbia': 50, 'Coverage': 30}
    loaders = {}
    for dataset in test_datasets:
        loaders[dataset] = validate_dataset_layout(
            dataset, os.path.join(image_root, dataset, 'Tp'),
            os.path.join(gt_root, dataset, 'Gt'), expected_counts[dataset])
    with open(record_txt, 'x') as f:
        f.write(f'checkpoint:{checkpoints_path}\n')
        f.write('F1: pixel F1 per image, arithmetic mean; original minmax postprocessing, threshold >0.5.\n')
        f.write('AUC: continuous minmax score maps quantized to uint8; auxiliary metric.\n')
        f.write('Local reconstructed splits; NC16 includes two tests sharing a source with training.\n')

    f1_in = []
    

    for i in range(len(test_datasets)):
        dataset = test_datasets[i]
        recent_image_root = os.path.join(image_root, dataset, 'Tp/')
        recent_gt_root = os.path.join(gt_root, dataset, 'Gt/')

        test_loader = loaders[dataset]
    
        recent_save_path = os.path.join(save_path, dataset)
        os.makedirs(recent_save_path, exist_ok=False)
        probability_path = str(run_path / 'scores' / dataset)
        model.ex_attn.memory_bank = None
        get_masks(test_loader, model=model, save_path=recent_save_path, device=device,
                  datesetname=dataset, probability_path=probability_path)
        if len(os.listdir(recent_save_path)) != expected_counts[dataset]:
            raise RuntimeError(f'{dataset}: prediction count differs from input count')

        aucs, f1s, ious = evaluate(recent_save_path, recent_gt_root, dataset, record_txt,
                                  epoch, model_name=model_name, path_probability=probability_path)

        f1_in.append(f1s)

    with open(record_txt, "a") as f:
        summary = (f'f1_in:{np.mean(f1_in):.4f} datasets:{",".join(test_datasets)} '
                   '(local reconstructed split)')
        print(summary)
        f.write(summary + '\n')
