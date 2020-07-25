# trainer
import time
import torch
from torch.utils.tensorboard import SummaryWriter

def train(model, config, train_dataloader, test_dataloader, check_interval=5000):
    criteria = config.criteria()
    optimizer = config.optimizer(model.parameters(), config.lr)
    start = time.time()
    counts = 0
    writer = SummaryWriter(filename_suffix=str(config))
    # writer.add_graph(model)
    torch.backends.cudnn.benchmark = True
    for epoch in range(config.epochs):
        for i, (words, word_ids, labels) in enumerate(train_dataloader):
            counts += labels.shape[0]
            optimizer.zero_grad()
            output = model(word_ids.to(config.device))
            loss = criteria(output, labels.to(config.device))
            loss.backward()
            optimizer.step()
            if ((epoch + 1) * i) % check_interval == 0:
                print("[%d seconds](epoch: %d/%d)[%d samples] loss: %.3f." % (time.time() - start, epoch + 1, config.epochs, counts, loss.mean().item()))
                # eval on test dataset
                model.eval()
                with torch.no_grad():
                    acc_eval_loss = 0.0
                    batches = 0
                    correct = 0
                    total_samples = 0
                    for j, (words, eval_word_ids, eval_labels) in enumerate(test_dataloader):
                        eval_output = model(eval_word_ids.to(config.device))
                        eval_loss = criteria(eval_output, eval_labels.to(config.device))
                        acc_eval_loss += eval_loss.item()
                        batches += 1
                        correct += torch.sum(torch.argmax(eval_output.cpu(), dim=1) == eval_labels)
                        total_samples += eval_word_ids.shape[0]
                    accuracy = 100.0 * correct / total_samples
                    print('eval loss: %.3f, accuracy: %.3f%% [%d/%d]' % (acc_eval_loss / batches, accuracy, correct, total_samples))
                    writer.add_scalar('Loss/eval', acc_eval_loss / batches, epoch + 1)
                    writer.add_scalar('Eval accuracy', accuracy, epoch + 1)
                model.train()
                writer.add_scalar('Loss/train', loss.mean().item(), epoch + 1)