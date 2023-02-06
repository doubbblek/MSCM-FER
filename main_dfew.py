import os
import time
import shutil
import random
import numpy as np 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import MultiStepLR
from ops.dfew_dataset import get_dfew_dataset, get_train_val_dfew
from ops.afew_dataset  import get_afew_dataset
from ops.mscm import resnet18, ResNet18_Weights

from utils import AverageMeter, accuracy
from configs import get_cfg_defaults
from tensorboardX import SummaryWriter


# ========================== Configs ==========================

cfg = get_cfg_defaults()
cfg.freeze()


# ========================== Train ==========================

def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda()
        input = input.cuda()

        # compute output
        output = model(input) # 
        loss = criterion(output, target)
        loss = loss 

        # compute gradient and do SGD step
        loss.backward()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        if cfg.CLIP_GRAD is not None:
            total_norm = clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD)
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.PRINT_FREQ == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.6f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


# ========================== Test ==========================

def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            # compute output
            output = model(input) #  
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 3))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.PRINT_FREQ == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return top1.avg


# ========================== Tools ==========================

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (cfg.ROOT_MODEL, cfg.STORE_NAME)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [cfg.ROOT_LOG, cfg.ROOT_MODEL,
                    os.path.join(cfg.ROOT_LOG, cfg.STORE_NAME),
                    os.path.join(cfg.ROOT_MODEL, cfg.STORE_NAME)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


# ========================== Main ==========================

best_prec1 = 0

def main():
    global cfg, best_prec1
    print(cfg)
    check_rootfolders()
    if cfg.SEED is not None:
        print('Using seed')
        random_seed(cfg.SEED)

    # define model
    # assert cfg.ARCH in ['x3d', 'tsm_resnet18']
    model = resnet18(weights=ResNet18_Weights, n_frames=cfg.NUM_FRAMES)
        
    model = torch.nn.DataParallel(model, device_ids=cfg.GPU_IDS).cuda()

    # model.load_state_dict(torch.load('checkpoint/dfew_sada_sigmoid_set_5/ckpt.best.pth.tar', map_location='cuda')['state_dict'])
    # print('loaded')

    optimizer = torch.optim.SGD(model.parameters(),
                                cfg.LR,
                                momentum=cfg.MOMENTUM,
                                weight_decay=cfg.WEIGHT_DECAY)
    
    cudnn.benchmark = True

    # Data loading code
    # train_dataset, test_dataset = get_dfew_dataset(cfg.NUM_FRAMES, cfg.INPUT_SIZE, cfg.SCALE_SIZE, set='set_1')
    train_dataset, test_dataset = get_train_val_dfew(cfg.NUM_FRAMES, cfg.INPUT_SIZE, cfg.SCALE_SIZE)
    # train_dataset, test_dataset = get_afew_dataset(cfg.NUM_FRAMES, cfg.INPUT_SIZE, cfg.SCALE_SIZE)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
        drop_last=False) # bugs

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    steplr = MultiStepLR(optimizer, milestones=cfg.LR_STEP, gamma=0.1)

    log_training = open(os.path.join(cfg.ROOT_LOG, cfg.STORE_NAME, 'log.csv'), 'w')
    with open(os.path.join(cfg.ROOT_LOG, cfg.STORE_NAME, 'args.txt'), 'w') as f:
        f.write(cfg.dump())
    tf_writer = SummaryWriter(log_dir=os.path.join(cfg.ROOT_LOG, cfg.STORE_NAME))

    for epoch in range(cfg.EPOCHS):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)

        # evaluate on validation set
        if epoch >= cfg.EVAL_START:
            if (epoch + 1) % cfg.EVAL_FREQ == 0 or epoch == cfg.EPOCHS - 1:
                prec1 = validate(val_loader, model, criterion, epoch, log_training, tf_writer)

                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

                output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
                print(output_best)
                log_training.write(output_best + '\n')
                log_training.flush()

                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': cfg.ARCH,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best)
        
        steplr.step()


if __name__ == '__main__':
    main()