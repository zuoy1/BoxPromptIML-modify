import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from boxpromptiml import BoxPromptIML
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import random
from segment_anything import SamPredictor, sam_model_registry
import cv2
from functools import partial


def metric(premask, groundtruth):
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)

    return f1

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def weighted_cross_entropy_loss(pred, mask, gamma0=0.5, gamma1=2.5):
    bce_loss = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    weights = gamma0 * (1 - mask) + gamma1 * mask
    weighted_bce = (weights * bce_loss)

    return weighted_bce.mean()


def mask_to_boxes(mask, min_area=20):
    """Return the box in [x1,y1,x2,y2] format"""
    img_height, img_width = mask.shape[:2]
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    areas = []
    # Define floating ratio probability distribution
    float_ratios =  [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    probabilities = [0.05, 0.05, 0.07, 0.08, 0.40, 0.10, 0.08, 0.07, 0.05, 0.05]
    

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        # Centre point coordinates
        cx, cy = x + w // 2, y + h // 2  
        ratio = random.choices(float_ratios, weights=probabilities, k=1)[0]
        
        new_w = int(w * (1 + ratio))
        new_h = int(h * (1 + ratio))

        x1 = max(cx - new_w // 2, 0)
        y1 = max(cy - new_h // 2, 0)
        x2 = min(cx + new_w // 2, img_width)
        y2 = min(cy + new_h // 2, img_height)

        boxes.append([x1, y1, x2, y2])
        areas.append(area)

    if boxes:
        boxes_with_areas = list(zip(boxes, areas))
        # Sort by area
        boxes_with_areas.sort(key=lambda x: x[1], reverse=True) 
        sorted_boxes = [box for box, _ in boxes_with_areas]
        return np.array(sorted_boxes)
    else:
        print(f'no area is large than {min_area}!!!')
        return np.zeros((0, 4))
    


def train(train_loader, model, teacher_model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    loss0_all = 0
    epoch_step = 0
    predictor_origin = SamPredictor(teacher_model)
    try:
        for i, (images, images_np, gts, gts_np, images_name) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images_np = images_np.numpy()
            gts_np = gts_np.numpy()
            images = Variable(images).cuda()
      
            outputs = model(images)

            teacher_outputs = []
            with torch.no_grad():
                # SAM输出
                for j in range(images_np.shape[0]):
                    image_np = images_np[j]
                    predictor_origin.set_image(image_np)
                    combined_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
                    boxes = mask_to_boxes(gts_np[j])
                    if len(boxes) > 0:
                        index = 0
                        for box in boxes:
                            masks, _, _ = predictor_origin.predict(
                                point_coords=None,
                                box=box,
                                multimask_output=False,
                                return_logits=False
                            )
                            combined_mask = np.maximum(combined_mask, masks)
                            index += 1
                            if index >= 3:
                                break
                        mask_pred = torch.from_numpy(combined_mask).to(images.device)
                    else:
                        print(f'gts_np[i].shape={gts_np[j].shape}')
                        mask_pred = np.zeros(gts_np[j].shape, dtype=np.uint8)
                        mask_pred = torch.from_numpy(mask_pred).to(images.device)
                    teacher_outputs.append(mask_pred)

            

            teacher_masks = torch.stack(teacher_outputs, dim=0)
            teacher_masks = teacher_masks.float()
            
            loss = structure_loss(outputs, teacher_masks)
                
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.item()))
              
                with open(opt.save_log, 'a') as f:
                    f.write('{} [Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss:{:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.item()))
                    f.write('\n')
               
        loss_all /= epoch_step
        loss0_all /= epoch_step
        with open(opt.save_log, 'a') as f:
            f.write('{} [Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}, Loss0_AVG: {:.4f}'.format(datetime.now(), epoch, opt.epoch, loss_all, loss0_all))
            f.write('\n')
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch >= 1:
            torch.save(model.state_dict(), save_path + 'checkpoint-{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'checkpoint-{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=21, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=224, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=10, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='1', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='/home/xidongdong/Dataset/mai_Train/',
                        help='the training rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./checkpoints/',
                        help='the path to save model and log')

    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    num_gpus = len(opt.gpu_id.split(','))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    model = BoxPromptIML()

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Tp/',
                              gt_root=opt.train_root + 'Gt/',
                              edge_root=opt.train_root + 'Edge/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=8)
    
    total_step = len(train_loader)

    opt.save_log = os.path.join(save_path, f'{datetime.now()}_log.log')
    with open(opt.save_log, 'w') as f:
        f.write('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {} \n'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))
        f.write('\n')

    step = 0
    writer_path = os.path.join(save_path, f'{datetime.now()}')
    writer = SummaryWriter(writer_path)
    best_mae = 0
    best_epoch = 0

    sam_model = sam_model_registry['vit_h'](checkpoint='./segment-anything/sam_vit_h_4b8939.pth')
    sam_model = sam_model.to(device)
    sam_model.eval()


    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, sam_model, optimizer, epoch, save_path, writer)
