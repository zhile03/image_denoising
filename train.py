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


def train(model, dataloader, criteria, device, optimizer, cur_epoch, total_epochs, initial_lr, final_lr,
          start_epoch, start_batch):
    loss_epoch = 0.
    for batch_idx, (noisy_patch, clean_patch, _) in enumerate(dataloader, start=0):
        if cur_epoch == start_epoch and batch_idx < start_batch:
            continue
        optimizer.zero_grad()
        noisy_patch, clean_patch = noisy_patch.to(device), clean_patch.to(device)
        pred = model(noisy_patch)
        loss = criteria(pred, clean_patch)
        # Backpropagation
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        # print(loss.item())

        if batch_idx+1 % 400 == 0 and batch_idx > 0:
            save_checkpoints(cur_epoch, batch_idx, opt.checkpoints_dir, model, optimizer)
            with  torch.no_grad():
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(noisy_patch[0].cpu().squeeze(0).numpy(), cmap='gray')
                axes[0].set_title("Noisy Image")
                axes[0].axis("off")
                axes[1].imshow(clean_patch[0].cpu().squeeze(0).numpy(), cmap='gray')
                axes[1].set_title("Clean Image")
                axes[1].axis("off")
                axes[2].imshow(pred[0].cpu().squeeze(0).numpy(), cmap='gray')
                axes[2].set_title("Predicted Noise")
                axes[2].axis("off")
                plt.tight_layout()
                plt.show()
            print(f"Epoch {cur_epoch+1} | Batch {batch_idx}/{len(dataloader)}  | Loss: {loss.item():.6f}")

    loss_epoch /= len(dataloader)
    lr_epoch = adjust_learning_rate(optimizer, cur_epoch, total_epochs, initial_lr, final_lr)
    return loss_epoch, lr_epoch


def test(model, dataloader, criteria, device):
    loss_epoch = 0.
    psnr_epoch = 0.
    pred_list = []
    gt_list = []
    name_list = []
    with torch.no_grad():
        for noisy_patch, clean_patch, imgname in dataloader:
            noisy_patch, clean_patch = noisy_patch.to(device), clean_patch.to(device)
            pred = model(noisy_patch)
            loss = criteria(pred, clean_patch)
            loss_epoch += loss.item()
            psnr_epoch += cal_psnr(pred, clean_patch).item()
            pred_list.append(pred)
            gt_list.append(clean_patch)
            name_list += imgname

    loss_epoch /= len(dataloader)
    psnr_epoch /= len(dataloader)
    return loss_epoch, psnr_epoch, pred_list, gt_list, name_list


def save_checkpoints(epoch, batch_idx, checkpoints_dir, model, optimizer):
    checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_{epoch+1}_batch_{batch_idx}.pth')
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def plot_metrics(train_losses, valid_losses, psnrs, learning_rates):
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
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='exp/first-try', help='')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training')
    parser.add_argument('--num-workers', type=int, default=8, help='the number of dataloader workers')
    parser.add_argument('--patch-size', type=int, default=40, help='')
    parser.add_argument('--noise-mean', type=float, default=0, help='')
    parser.add_argument('--noise-std', type=float, default=25, help='')
    parser.add_argument('--num-patches', type=int, default=512, help='')
    parser.add_argument('--trainset_path', type=str, default='./BSDS500-master/BSDS500/data/images/train', help='')
    parser.add_argument('--testset_path', type=str, default='./BSD68+Set12', help='')
    parser.add_argument('--total-epochs', type=int, default=50, help='')
    parser.add_argument('--initial-lr', type=float, default=1e-04, help='')
    parser.add_argument('--final-lr', type=float, default=1e-04, help='')
    parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints', help='check for the latest checkpoint')

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
    os.makedirs(opt.checkpoints_dir, exist_ok=True)

    # save hyp-parameter
    with open(os.path.join(opt.save_dir, 'hyp.yaml'), 'w') as f:
        yaml.dump(opt, f, sort_keys=False)

    # folder to save the predicted noise in the validation
    valid_folder = os.path.join(opt.save_dir, 'valid_res')
    os.makedirs(valid_folder, exist_ok=True)

    # create model
    model = DnCNN(channels=1, num_layers=17)
    model.apply(weights_init)
    model.to(device)

    # loss
    criteria = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.initial_lr, momentum=0.9, weight_decay=0.0001)

    # check for the latest checkpoint
    checkpoint_files = glob.glob(os.path.join(opt.checkpoints_dir, '*.pth'))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        checkpoint = torch.load(latest_checkpoint)
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch_idx']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded: {latest_checkpoint}")
    else:
        start_epoch, start_batch = 0, 0
        print("No checkpoint found, starting training from scratch.")

    # dataloader
    train_dataset = DenoisingDataset(image_dir=opt.trainset_path, phase='train', patch_size=opt.patch_size,
                                     mean=opt.noise_mean, sigma=opt.noise_std, num_patches=opt.num_patches)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    test_dataset = DenoisingDataset(image_dir=opt.testset_path, phase='test', patch_size=opt.patch_size,
                                    mean=opt.noise_mean, sigma=opt.noise_std, num_patches=opt.num_patches)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    # list to store metrics
    train_losses = []
    valid_losses = []
    psnrs = []
    learning_rates = []

    for idx in range(0, opt.total_epochs):
        t0 = time.time()
        train_loss_epoch, lr_epoch = train(model=model, dataloader=train_dataloader, criteria=criteria, device=device,
                                           optimizer=optimizer, cur_epoch=idx, total_epochs=opt.total_epochs,
                                           initial_lr=opt.initial_lr, final_lr=opt.final_lr, start_epoch=start_epoch,
                                           start_batch=start_batch)
        t1 = time.time()
        valid_loss_epoch, psnr_epoch, pred_HR, gt_list, img_names = test(
            model=model, dataloader=test_dataloader, criteria=criteria, device=device)
        t2 = time.time()

        print("=" * 90)
        print(
            f"Epoch: {idx+1}/{opt.total_epochs} | LR: {lr_epoch:.5f} | Training Loss: {train_loss_epoch:.5f} | "
            f"Validation Loss: {valid_loss_epoch:.5f} | PSNR: {psnr_epoch:.3f} dB | Time: {t2 - t0:.1f} seconds")
        print("=" * 90)

        save_checkpoints(idx, len(train_dataloader), opt.checkpoints_dir, model, optimizer)

        # store metrics
        train_losses.append(train_loss_epoch)
        valid_losses.append(valid_loss_epoch)
        psnrs.append(psnr_epoch)
        learning_rates.append(lr_epoch)

    # plot metrics after training
    plot_metrics(train_losses, valid_losses, psnrs, learning_rates)

