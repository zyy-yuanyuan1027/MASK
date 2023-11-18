import os
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits


def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        # model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)                                    # 原代码
        # embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
        model1 = vits.__dict__[args.arch2](patch_size=args.patch_size, num_classes=0)  # 此处进行了修改
        model2 = vits.__dict__[args.arch3](patch_size=args.patch_size, num_classes=0)  # 此处进行了修改
        model3 = vits.__dict__[args.arch4](patch_size=args.patch_size, num_classes=0)  # 此处进行了修改
        embed_dim = model1.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        # model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)                                 # 原代码
        # embed_dim = model.embed_dim
        model1 = torch.hub.load('facebookresearch/xcit:main', args.arch2, num_classes=0)  # 此处进行了修改
        model2 = torch.hub.load('facebookresearch/xcit:main', args.arch3, num_classes=0)  # 此处进行了修改
        model3 = torch.hub.load('facebookresearch/xcit:main', args.arch4, num_classes=0)  # 此处进行了修改
        embed_dim = model1.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        # model = torchvision_models.__dict__[args.arch]()                                                               # 原代码
        # embed_dim = model.fc.weight.shape[1]
        # model.fc = nn.Identity()
        model1 = torchvision_models.__dict__[args.arch2]()  # 此处进行了修改
        model2 = torchvision_models.__dict__[args.arch3]()  # 此处进行了修改
        model3 = torchvision_models.__dict__[args.arch4]()  # 此处进行了修改
        embed_dim = model1.fc.weight.shape[1]
        model1.fc = nn.Identity()  # 此处进行了修改
        model2.fc = nn.Identity()  # 此处进行了修改
        model3.fc = nn.Identity()  # 此处进行了修改
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    # model.cuda()                                                                                                       # 原代码
    # model.eval()
    model1.cuda()  # 此处进行了修改
    model1.eval()
    model2.cuda()  # 此处进行了修改
    model2.eval()
    model3.cuda()  # 此处进行了修改
    model3.eval()
    # load weights to evaluate
    # utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)      # 原代码
    utils.load_pretrained_weights(model1, args.pretrained_weights, args.checkpoint_key1, args.arch2,
                                  args.patch_size)  # 此处进行了修改
    utils.load_pretrained_weights(model2, args.pretrained_weights, args.checkpoint_key2, args.arch3,
                                  args.patch_size)  # 此处进行了修改
    utils.load_pretrained_weights(model3, args.pretrained_weights, args.checkpoint_key3, args.arch4,
                                  args.patch_size)  # 此处进行了修改
    print(f"Model1, Model2 and Model3 {args.arch} built.")

    # linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)                                          # 原代码
    print("embed_dim", embed_dim)
    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)  # 此处进行了修改
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    #dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)                   # 原代码
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "test"), transform=val_transform)  # 此处进行了修改
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.evaluate:
        utils.load_pretrained_linear_weights(linear_classifier, args.arch, args.patch_size)
        # test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)  # 原代码
        test_stats = validate_network(val_loader, model1, model2, model3, linear_classifier, args.n_last_blocks,  # 此处进行了修改
                                      args.avgpool_patchtokens)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        # train_stats = train(model, linear_classifier, optimizer,                                                       # 原代码
        #                    train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        train_stats = train(model1, model2, model3, linear_classifier, optimizer,  # 此处进行了修改
                            train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            # test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)       # 原代码
            test_stats = validate_network(val_loader, model1, model2, model3, linear_classifier, args.n_last_blocks,  # 此处进行了修改
                                          args.avgpool_patchtokens)
            print(
                f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
          "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


# def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool):
def train(model1, model2, model3, linear_classifier, optimizer, loader, epoch, n, avgpool):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                # intermediate_output = model.get_intermediate_layers(inp, n)                                            # 原代码
                intermediate_output1 = model1.get_intermediate_layers(inp, n)  # 此处进行了修改
                intermediate_output2 = model2.get_intermediate_layers(inp, n)  # 此处进行了修改
                intermediate_output3 = model3.get_intermediate_layers(inp, n)  # 此处进行了修改
                # output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)                                     # 原代码
                output1 = torch.cat([x[:, 0] for x in intermediate_output1], dim=-1)  # 此处进行了修改
                output2 = torch.cat([x[:, 0] for x in intermediate_output2], dim=-1)  # 此处进行了修改
                output3 = torch.cat([x[:, 0] for x in intermediate_output3], dim=-1)  # 此处进行了修改
                if avgpool:
                    # output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    # output = output.reshape(output.shape[0], -1)
                    output1 = torch.cat(
                        (output1.unsqueeze(-1), torch.mean(intermediate_output1[-1][:, 1:], dim=1).unsqueeze(-1)),
                        dim=-1)  # 此处进行了修改
                    output1 = output1.reshape(output1.shape[0], -1)
                    output2 = torch.cat(
                        (output2.unsqueeze(-1), torch.mean(intermediate_output2[-1][:, 1:], dim=1).unsqueeze(-1)),
                        dim=-1)  # 此处进行了修改
                    output2 = output2.reshape(output2.shape[0], -1)
                    output3 = torch.cat(
                        (output3.unsqueeze(-1), torch.mean(intermediate_output2[-1][:, 1:], dim=1).unsqueeze(-1)),
                        dim=-1)  # 此处进行了修改
                    output3 = output3.reshape(output3.shape[0], -1)

            else:
                # output = model(inp)                                                                                    # 原代码
                output1 = model1(inp)  # 此处进行了修改
                output2 = model2(inp)  # 此处进行了修改
                output3 = model3(inp)  # 此处进行了修改
        # output = linear_classifier(output)                                                                            # 原代码

        #print("output1的长度",output1.size())
        #print("output2的长度",len(output2))
        #print("output3的长度", output3.size())

        #output = torch.cat((output1, output2), dim=-1)
        #print("output的长度", output.size())
        #output = torch.cat((output,output3),dim=-1)
        #print("output的长度", output.size())
        # output = torch.add(output1, output2)
        # print("output的add长度",output.size())

        # output = torch.div(output, 2)
        # print("output的div长度",output.size())
        output = output1+output2
        output = output+output3

        output = linear_classifier(output)
        #print("output的线性长度",output.size())

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
# def validate_network(val_loader, model, linear_classifier, n, avgpool):
def validate_network(val_loader, model1, model2, model3, linear_classifier, n, avgpool):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                # intermediate_output = model.get_intermediate_layers(inp, n)                                            # 原代码
                intermediate_output1 = model1.get_intermediate_layers(inp, n)  # 此处进行了修改
                intermediate_output2 = model2.get_intermediate_layers(inp, n)  # 此处进行了修改
                intermediate_output3 = model3.get_intermediate_layers(inp, n)  # 此处进行了修改
                # output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)                                     # 原代码
                output1 = torch.cat([x[:, 0] for x in intermediate_output1], dim=-1)  # 此处进行了修改
                output2 = torch.cat([x[:, 0] for x in intermediate_output2], dim=-1)  # 此处进行了修改
                output3 = torch.cat([x[:, 0] for x in intermediate_output3], dim=-1)  # 此处进行了修改
                if avgpool:
                    # output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)    # 原代码
                    # output = output.reshape(output.shape[0], -1)
                    output1 = torch.cat(
                        (output1.unsqueeze(-1), torch.mean(intermediate_output1[-1][:, 1:], dim=1).unsqueeze(-1)),
                        dim=-1)  # 此处进行了修改
                    output1 = output1.reshape(output1.shape[0], -1)
                    output2 = torch.cat(  # 此处进行了修改
                        (output2.unsqueeze(-1), torch.mean(intermediate_output2[-1][:, 1:], dim=1).unsqueeze(-1)),
                        dim=-1)
                    output2 = output2.reshape(output2.shape[0], -1)
                    output3 = torch.cat(  # 此处进行了修改
                        (output3.unsqueeze(-1), torch.mean(intermediate_output2[-1][:, 1:], dim=1).unsqueeze(-1)),
                        dim=-1)
                    output3 = output3.reshape(output3.shape[0], -1)
            else:
                # output = model(inp)                                                                                    # 原代码
                output1 = model1(inp)  # 此处进行了修改
                output2 = model2(inp)  # 此处进行了修改
                output3 = model3(inp)  # 此处进行了修改

        # print("output1的长度",output1.size())
        # print("output2的长度",len(output2))
        # output = linear_classifier(output)                                                                             # 原代码
        #output = torch.cat((output1, output2), dim=-1)
        #output = torch.cat((output,output3),dim=-1)

        output = output1+output2
        output = output+output3
        # output = torch.add(output1, output2)
        # print("output的add长度",output.size())

        # output = torch.div(output, 2)
        # print("output的div长度",output.size())
        output = linear_classifier(output)
        # print("output的线性长度",output.size())

        loss = nn.CrossEntropyLoss()(output, target)

        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if linear_classifier.module.num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if linear_classifier.module.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        #self.linear = nn.Linear(int(dim * 3), num_labels)   # 此处进行了修改
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    # parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')                                                           # 原代码
    parser.add_argument('--arch', default='vit_tiny', type=str, help='Architecture')  # 此处进行了修改
    parser.add_argument('--arch2', default='vit_tiny2', type=str, help='Architecture')  # 此处进行了修改
    parser.add_argument('--arch3', default='vit_tiny3', type=str, help='Architecture')  # 此处进行了修改
    parser.add_argument('--arch4', default='vit_tiny4', type=str, help='Architecture')  # 此处进行了修改
    
    
    
    
    
    
    
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='./train-T3-S0.15T0.10T0.15/checkpoint.pth', type=str,
                        help="Path to pretrained weights to evaluate.")  # 原代码
    # parser.add_argument('--pretrained_weights', default='./main_dino/checkpoint.pth', type=str, help="Path to pretrained weights to evaluate.")  # 此处进行了添加
    # parser.add_argument("--checkpoint_key", default="student", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--checkpoint_key1", default="teacher1", type=str,  # 此处进行了修改
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--checkpoint_key2", default="teacher2", type=str,  # 此处进行了修改
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--checkpoint_key3", default="teacher3", type=str,  # 此处进行了修改
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')  # 原来为128
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)                                                      # 原代码
    parser.add_argument('--data_path', default='./cifar-10/', type=str)  # 此处进行了添加
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")

    # parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')                          # 原代码
    parser.add_argument('--output_dir', default="./linear-T3-S0.15T0.10T0.15", help='Path to save logs and checkpoints')  # 此处进行了添加
    parser.add_argument('--num_labels', default=10, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    args = parser.parse_args()
    eval_linear(args)