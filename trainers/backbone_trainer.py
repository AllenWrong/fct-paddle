
from tqdm import tqdm
import paddle
from utils.logging_utils import AverageMeter
from utils.eval_utils import accuracy


class BackboneTrainer:

    def train(self, train_loader, model, criterion, optimizer):
        losses = AverageMeter("Loss", ":.3f")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")

        model.train()

        # for i, (images, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        for i, (images, target) in enumerate(train_loader):
            output, _ = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            bs = images.shape[1]
            losses.update(loss.item(), bs)
            top1.update(acc1.item(), bs)
            top5.update(acc5.item(), bs)

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

        return top1.avg, top5.avg, losses.avg

    def valid(self, val_loader, model, criterion):
        losses = AverageMeter("Loss", ":.3f")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")

        model.eval()

        with paddle.no_grad():
            # for i, (images, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            for i, (images, target) in enumerate(val_loader):
                output, _ = model(images)

                loss = criterion(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                bs = images.shape[1]
                losses.update(loss.item(), bs)
                top1.update(acc1.item(), bs)
                top5.update(acc5.item(), bs)

        return top1.avg, top5.avg, losses.avg