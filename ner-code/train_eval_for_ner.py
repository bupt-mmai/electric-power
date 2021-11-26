# coding: UTF-8
import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
import time
from tensorboardX import SummaryWriter
from loss import FocalLoss

from pytorch_transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def model_train(config, model, train_iter, dev_iter):
    start_time = time.time()

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0
         },
    ]
    t_total = len(train_iter) * config.num_train_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=t_total * config.warmup_proportion, num_training_steps=t_total
    )
    criterion = FocalLoss(gamma =2, alpha = 1) #调整gamma=0,1,2,3
    label_map = {i: label for i, label in enumerate(config.class_list)}

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Train Num examples = %d", config.train_num_examples)
    logger.info("  Dev Num examples = %d", config.dev_num_examples)
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Instantaneous batch size GPU/CPU = %d", config.batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Train device:%s, id:%d", config.device, config.device_id)

    global_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    for epoch in range(config.num_train_epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_train_epochs))
        # scheduler.step() # 学习率衰减
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_iter):
            global_batch += 1
            model.train()

            input_ids = torch.tensor(input_ids).type(torch.LongTensor).to(config.device)
            attention_mask = torch.tensor(attention_mask).type(torch.LongTensor).to(config.device)
            token_type_ids = torch.tensor(token_type_ids).type(torch.LongTensor).to(config.device)
            labels = torch.tensor(labels).type(torch.LongTensor).to(config.device)

            outputs = model(input_ids, attention_mask, token_type_ids)

            active_loss = attention_mask.view(-1) == 1
            active_logits = outputs.view(-1, config.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]

            model.zero_grad()
            #loss = F.cross_entropy(active_logits, active_labels)
            loss = criterion(active_logits, active_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule

            active_labels = active_labels.data.cpu()
            predic = torch.max(active_logits.data, 1)[1].cpu()

            labels_all = np.append(labels_all, active_labels)
            predict_all = np.append(predict_all, predic)

            if global_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果

                true_label = [label_map[key] for key in labels_all]
                predict_label = [label_map[key] for key in predict_all]

                train_acc = metrics.accuracy_score(labels_all, predict_all)
                train_precision = precision_score(true_label, predict_label)
                train_recall = recall_score(true_label, predict_label)
                train_f1 = f1_score(true_label, predict_label)
                predict_all = np.array([], dtype=int)
                labels_all = np.array([], dtype=int)

                acc, precision, recall, f1, dev_loss = model_evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    improve = '*'
                    last_improve = global_batch
                else:
                    improve = ''
                time_dif = time.time() - start_time
                msg = '{0:>6}, Train Loss: {1:>.4f}, train_acc: {2:>.2%}, precision: {3:>.2%}, recall: {4:>.2%}, f1: {5:>.2%}' \
                      ' Val Loss: {6:>5.6f}, acc: {7:>.2%}, precision: {8:>.2%}, recall: {9:>.2%}, f1: {10:>.2%}, ' \
                      ' Time: {11} - {12}'
                logger.info(msg.format(global_batch, loss.item(), train_acc, train_precision, train_recall, train_f1,
                                       dev_loss, acc, precision, recall, f1, time_dif, improve))

            if global_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def model_evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    label_map = {i: label for i, label in enumerate(config.class_list)}
    criterion = FocalLoss(gamma =2, alpha = 1)
    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(data_iter):

            input_ids = torch.tensor(input_ids).type(torch.LongTensor).to(config.device)
            attention_mask = torch.tensor(attention_mask).type(torch.LongTensor).to(config.device)
            token_type_ids = torch.tensor(token_type_ids).type(torch.LongTensor).to(config.device)
            labels = torch.tensor(labels).type(torch.LongTensor).to(config.device)

            outputs = model(input_ids, attention_mask, token_type_ids)

            active_loss = attention_mask.view(-1) == 1
            active_logits = outputs.view(-1, config.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]

            #loss = F.cross_entropy(active_logits, active_labels)
            loss = criterion(active_logits, active_labels)
            loss_total += loss
            active_labels = active_labels.data.cpu().numpy()
            predic = torch.max(active_logits.data, 1)[1].cpu().numpy()

            labels_all = np.append(labels_all, active_labels)
            predict_all = np.append(predict_all, predic)

    true_label = [label_map[key] for key in labels_all]
    predict_label = [label_map[key] for key in predict_all]

    acc = metrics.accuracy_score(labels_all, predict_all)
    precision = precision_score(true_label, predict_label)
    recall = recall_score(true_label, predict_label)
    f1 = f1_score(true_label, predict_label)
    if test:
        report = classification_report(true_label, predict_label, digits=4)
        confusion = metrics.confusion_matrix(true_label, predict_label)
        return acc, precision, recall, f1, loss_total / len(data_iter), report, confusion
    return acc, precision, recall, f1, loss_total / len(data_iter)


def model_test(config, model, test_iter):
    # test!
    logger.info("***** Running testing *****")
    logger.info("  Test Num examples = %d", config.test_num_examples)
    start_time = time.time()
    acc, precision, recall, f1, test_loss, test_report, test_confusion = model_evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.4f},  Test acc: {1:>.2%}, precision: {2:>.2%} recall: {3:>.2%}, f1: {4:>.2%}'
    logger.info(msg.format(test_loss, acc, precision, recall, f1))
    logger.info("Precision, Recall and F1-Score...")
    logger.info(test_report)
    logger.info("Confusion Matrix...")
    logger.info(test_confusion)
    time_dif = time.time() - start_time
    logger.info("Time usage:%.6fs", time_dif)


def model_save(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = os.path.join(path, 'bert_ner.pkl')
    torch.save(model.state_dict(), file_name)
    logger.info("model saved.")


def model_laod(model, path, device='cpu', device_id=0):
    file_name = os.path.join(path, 'bert_ner.pkl')
    model.load_state_dict(torch.load(file_name,
                                     map_location=device if device == 'cpu' else "{}:{}".format(device, device_id)))
