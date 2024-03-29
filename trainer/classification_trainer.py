# trainer
import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter

def train(
        model, config, train_dataloader, test_dataloader, 
        checkpoint_prefix, output_dir='/tmp/model', check_interval=50, **kwargs
    ):
    kwargs = kwargs['kwargs']
    criteria = config.criteria()
    optimizer = config.optimizer(model.parameters(), lr=config.lr)
    if 'weight_decay' in config.__dict__:
        optimizer.weight_decay = config.weight_decay
    start = time.time()
    counts = 0
    writer = SummaryWriter(filename_suffix=str(config))
    # writer.add_graph(model)
    base_epoch = 0

    if 'existing_checkpoint_filepath' in kwargs:
        print('Load checkpoint from %s' % kwargs['existing_checkpoint_filepath'])
        checkpoint = torch.load(kwargs['existing_checkpoint_filepath'])
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        base_epoch = checkpoint['epoch'] 

    torch.backends.cudnn.benchmark = True
    for epoch in range(config.epochs):
        for i, (word_ids, labels, x_lens, y_lens) in enumerate(train_dataloader):
            counts += labels.shape[0]
            optimizer.zero_grad()
            output = model(word_ids.to(config.device), x_lens)
            loss = criteria(output, labels.to(config.device))
            loss.backward()
            optimizer.step()
        if epoch % check_interval == 0 or epoch == config.epochs - 1:
            checkpoint_filename = checkpoint_prefix + '_' + str(epoch + base_epoch) + '.ckpt'
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            checkpoint_filepath = os.path.join(output_dir, checkpoint_filename)
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + base_epoch,
            }, checkpoint_filepath)
            print('%d seconds(epoch: %d/%d), saved checkpoint file: %s' % (time.time() - start, epoch, config.epochs, checkpoint_filepath))
            print('%d seconds](epoch: %d/%d)[%d samples] loss: %.3f.' % (time.time() - start, epoch, config.epochs, counts, loss.mean().item()))
            # eval on test dataset
            model.eval()
            with torch.no_grad():
                acc_eval_loss = 0.0
                batches = 0
                correct = 0
                total_samples = 0
                for j, (eval_word_ids, eval_labels, eval_x_lens, eval_y_lens) in enumerate(test_dataloader):
                    eval_output = model(eval_word_ids.to(config.device), eval_x_lens)
                    eval_loss = criteria(eval_output, eval_labels.to(config.device))
                    acc_eval_loss += eval_loss.item()
                    batches += 1
                    correct += torch.sum(torch.argmax(eval_output.cpu(), dim=1) == eval_labels)
                    total_samples += eval_word_ids.shape[0]
                accuracy = 100.0 * correct / total_samples
                print('eval loss: %.3f, accuracy: %.3f%% [%d/%d]' % (acc_eval_loss / batches, accuracy, correct, total_samples))
                writer.add_scalar('Loss/eval', acc_eval_loss / batches, epoch)
                writer.add_scalar('Eval accuracy', accuracy, epoch)
            model.train()
            writer.add_scalar('Loss/train', loss.mean().item(), epoch)