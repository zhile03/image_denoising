import os
import yaml
import argparse
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from model import DnCNN
from dataset import DenoisingDataset
from torch.utils.data import DataLoader

# Apply He initialization
def weights_init(m):
  if isinstance(m, nn.Conv2d):
    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


def cal_psnr(x, y):
    '''
    Parameters-
    x, y are two tensors has the same shape (1, C, H, W)
    Returns-
    score : PSNR.
    '''
    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    score = - 10 * torch.log10(mse)
    return score


def adjust_learning_rate(optimizer, epoch, total_epochs, initial_lr, final_lr):
    lr = initial_lr * (final_lr / initial_lr) ** (epoch / (total_epochs - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_image(img, path):
    img = img.detach().cpu().numpy()
    img = np.clip(img*255, 0, 255)
    img = img.astype(np.uint8)
    cv2.imwrite(path, img)

def train(model, dataloader, criteria, device, optimizer, cur_epoch, total_epochs, initial_lr, final_lr,
          start_epoch, save_dir):
    lr_epoch = adjust_learning_rate(optimizer, cur_epoch, total_epochs, initial_lr, final_lr)
    loss_epoch = 0.
    images_dir = os.path.join(save_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    for batch_idx, (noisy_patch, clean_patch, _) in enumerate(dataloader, start=0):
        optimizer.zero_grad()
        noisy_patch, clean_patch = noisy_patch.to(device), clean_patch.to(device)
        pred = model(noisy_patch) # predict residual
        loss = criteria(pred, noisy_patch - clean_patch) # MSE on residual
        # Backpropagation
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        # print(loss.item())

        if batch_idx  == 0:
            denoised = noisy_patch - pred
            save_image(denoised[0].squeeze(0).detach(), os.path.join(images_dir, f'{cur_epoch+1}_denoised_img.png'))
            save_image(noisy_patch[0].squeeze(0).detach(), os.path.join(images_dir, f'{cur_epoch+1}_noisy_img.png'))
            save_image(pred[0].squeeze(0).detach(), os.path.join(images_dir, f'{cur_epoch+1}_pred_img.png'))
            save_image(clean_patch[0].squeeze(0).detach(), os.path.join(images_dir, f'{cur_epoch+1}_clear_img.png'))
            
            print(f"Epoch {cur_epoch+1} | Batch {batch_idx}/{len(dataloader)}  | Loss: {loss.item():.6f}")

    loss_epoch /= len(dataloader)

    return loss_epoch, lr_epoch


def test(model, dataloader, criteria, device, save_dir, epoch):
    loss_epoch = 0.
    psnr_epoch = 0.
    pred_list = []
    gt_list = []
    name_list = []
    valid_folder = os.path.join(save_dir, 'valid_res')
    os.makedirs(valid_folder, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (noisy_patch, clean_patch, imgname) in enumerate(dataloader):
            noisy_patch, clean_patch = noisy_patch.to(device), clean_patch.to(device)
            pred = model(noisy_patch)
            denoised = noisy_patch - pred
            loss = criteria(denoised, clean_patch) # loss on denoised vs clean
            loss_epoch += loss.item()
            psnr_epoch += cal_psnr(denoised, clean_patch).item()
            
            # save the images
            save_image(denoised[0].squeeze(0).detach(), os.path.join(valid_folder, f'{epoch+1}_{imgname[0]}_denoised_img.png'))
            save_image(noisy_patch[0].squeeze(0).detach(), os.path.join(valid_folder, f'{epoch+1}_{imgname[0]}_noisy_img.png'))
            save_image(pred[0].squeeze(0).detach(), os.path.join(valid_folder, f'{epoch+1}_{imgname[0]}_pred_img.png'))
            save_image(clean_patch[0].squeeze(0).detach(), os.path.join(valid_folder, f'{epoch+1}_{imgname[0]}_clear_img.png'))
            
            pred_list.append(pred)
            gt_list.append(clean_patch)
            name_list += imgname

    loss_epoch /= len(dataloader)
    psnr_epoch /= len(dataloader)
    return loss_epoch, psnr_epoch, pred_list, gt_list, name_list


def save_checkpoints(epoch, save_dir, model, optimizer):
    checkpoints_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_{epoch+1}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def plot_metrics(train_losses, valid_losses, psnrs, learning_rate, save_dir): 
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(10, 8))

    # plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # plot psnr
    plt.subplot(2, 2, 2)
    plt.plot(epochs, psnrs, label='PSNR', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR over Epochs')
    plt.legend()
    
    # plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(epochs, learning_rates, label='Learning Rate', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'results.png'), dpi=200)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='exp/dncnn_s/first_try', help='')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training')
    parser.add_argument('--num-workers', type=int, default=8, help='the number of dataloader workers')
    parser.add_argument('--patch-size', type=int, default=40, help='')
    parser.add_argument('--noise-mean', type=float, default=0, help='')
    parser.add_argument('--noise-std', type=float, default=25, help='')
    parser.add_argument('--num-patches', type=int, default=512, help='')
    parser.add_argument('--trainset_path', type=str, default='./BSDS500-master/train', help='')
    parser.add_argument('--testset_path', type=str, default='./BSD68+Set12', help='')
    parser.add_argument('--total-epochs', type=int, default=50, help='')
    parser.add_argument('--initial-lr', type=float, default=0.1, help='')
    parser.add_argument('--final-lr', type=float, default=1e-04, help='')
    parser.add_argument('--resume', action='store_true', help='resume training from the latest checkpoint')
    parser.add_argument('--resume_weight', type=str, default='', help='path to resume weight file if needed')
    opt = parser.parse_args()

    # device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    device = torch.device(device)

    # create folders to save results
    if os.path.exists(opt.save_dir):
        print(f"Warning: {opt.save_dir} exists, please delete it manually if it is useless.")
    os.makedirs(opt.save_dir, exist_ok=True)

    # save hyp-parameter
    with open(os.path.join(opt.save_dir, 'hyp.yaml'), 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # create model
    model = DnCNN(channels=1, num_layers=17)
    model.apply(weights_init)
    model.to(device)

    # loss
    criteria = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.initial_lr, momentum=0.9, weight_decay=0.0001)

    # check for the latest checkpoint
    if opt.resume and os.path.isfile(opt.resume_weight):
        checkpoint = torch.load(opt.resume_weight, map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded: {start_epoch}")
    else:
        start_epoch = 0
        print("No checkpoint found, starting training from scratch.")


    # dataloader
    train_dataset = DenoisingDataset(image_dir=opt.trainset_path, phase='train', patch_size=opt.patch_size,
                                     mean=opt.noise_mean, sigma=opt.noise_std, num_patches=opt.num_patches)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    test_dataset = DenoisingDataset(image_dir=opt.testset_path, phase='test', patch_size=opt.patch_size,
                                    mean=opt.noise_mean, sigma=opt.noise_std, num_patches=opt.num_patches)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # list to store metrics
    train_losses = []
    valid_losses = []
    psnrs = []
    learning_rates = []
    
    results_file = os.path.join(opt.save_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write('Epoch | LR | Training Loss | Validation Loss | PSNR | Time\n')

    for idx in range(start_epoch, opt.total_epochs):
        t0 = time.time()
        train_loss_epoch, lr_epoch = train(model=model, dataloader=train_dataloader, criteria=criteria, device=device,
                                           optimizer=optimizer, cur_epoch=idx, total_epochs=opt.total_epochs,
                                           initial_lr=opt.initial_lr, final_lr=opt.final_lr, start_epoch=start_epoch,
                                           save_dir=opt.save_dir)
        
        t1 = time.time()
        valid_loss_epoch, psnr_epoch, pred_list, gt_list, img_names = test(model=model, dataloader=test_dataloader, 
                                                                           criteria=criteria, device=device, save_dir=opt.save_dir,
                                                                           epoch=idx)
        t2 = time.time()

        print("=" * 90)
        print(
            f"Epoch: {idx+1}/{opt.total_epochs} | LR: {lr_epoch:.5f} | Training Loss: {train_loss_epoch:.5f} | "
            f"Validation Loss: {valid_loss_epoch:.5f} | PSNR: {psnr_epoch:.3f} dB | Time: {t2 - t0:.1f} seconds")
        print("=" * 90)
        
        save_checkpoints(idx, opt.save_dir, model, optimizer)

        # store metrics
        train_losses.append(train_loss_epoch)
        valid_losses.append(valid_loss_epoch)
        psnrs.append(psnr_epoch)
        learning_rates.append(lr_epoch)
        
        with open(results_file, 'a') as f:
            f.write(f"{idx+1} | {lr_epoch:.5f} | {train_loss_epoch:.5f} | {valid_loss_epoch:.5f} | {psnr_epoch:.3f} | {t2-t0:.1f}\n")

    # plot metrics after training
    plot_metrics(train_losses, valid_losses, psnrs, learning_rates, opt.save_dir) 
    print("Training finished.")