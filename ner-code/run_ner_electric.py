import logging
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from processors.GermEval2014Processor import GermEval2014Processor
from utils import convert_examples_to_features_for_ner, BuildDataSet
from pytorch_transformers import BertTokenizer
from models.bert import BertLSMTMClassification
from train_eval_for_ner import model_train, model_test, model_save
import time


class Electirc_NER_Config:

    def __init__(self):
        absdir = os.path.dirname(os.path.abspath(__file__))
        _pretrain_path = '/bert_pretrain_models/bert-base-chinese'
        _config_file = 'config.json'
        _model_file = 'pytorch_model.bin'
        _tokenizer_file = 'vocab.txt'
        _data_path = '/data/Electirc_Ner'

        self.task = 'Electirc_Ner'
        self.config_file = os.path.join(absdir + _pretrain_path, _config_file)
        self.model_name_or_path = os.path.join(absdir + _pretrain_path, _model_file)
        self.tokenizer_file = os.path.join(absdir + _pretrain_path, _tokenizer_file)
        self.data_dir = absdir + _data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')              # 设备
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.device_id = 3
        self.do_lower_case = True
        self.label_on_test_set = True
        self.requires_grad = True
        self.class_list = []
        self.num_labels = 28
        self.train_num_examples = 0
        self.dev_num_examples = 0
        self.test_num_examples = 0
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        self.require_improvement = 5000                                                         # 若超过1000batch效果还没提升，则提前结束训练
        self.num_train_epochs = 30                                                              # epoch数
        self.batch_size = 32                                                                    # mini-batch大小
        self.pad_size = 128                                                                     # 每句话处理成的长度
        self.learning_rate = 2e-5                                                               # 学习率
        self.weight_decay = 0.01                                                                # 权重衰减因子
        self.warmup_proportion = 0.1                                                            # Proportion of training to perform linear learning rate warmup for.
        # logging
        self.is_logging2file = True
        self.logging_dir = absdir + '/logging' + '/' + self.task
        # save
        self.save_path = absdir + '/model_saved' + '/' + self.task


def Electirc_ner_task(config):

    if config.device.type == 'cuda':
        torch.cuda.set_device(config.device_id)

    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_file,
                                              do_lower_case=config.do_lower_case)
    processor = GermEval2014Processor(config.label_on_test_set)
    config.class_list = processor.get_labels(config.data_dir)
    config.num_labels = len(config.class_list)

    train_examples = processor.get_train_examples(config.data_dir)
    config.train_num_examples = len(train_examples)

    dev_examples = processor.get_dev_examples(config.data_dir)
    config.dev_num_examples = len(dev_examples)

    test_examples = processor.get_test_examples(config.data_dir)
    config.test_num_examples = len(test_examples)

    train_features = convert_examples_to_features_for_ner(
        examples=train_examples,
        tokenizer=tokenizer,
        label_list=config.class_list,
        max_seq_length=config.pad_size,
        pad_token_label_id=config.pad_token_label_id
    )
    dev_features = convert_examples_to_features_for_ner(
        examples=dev_examples,
        tokenizer=tokenizer,
        label_list=config.class_list,
        max_seq_length=config.pad_size,
        pad_token_label_id=config.pad_token_label_id
    )
    test_features = convert_examples_to_features_for_ner(
         examples=test_examples,
         tokenizer=tokenizer,
         label_list=config.class_list,
         max_seq_length=config.pad_size,
         pad_token_label_id=config.pad_token_label_id
    )

    train_dataset = BuildDataSet(train_features)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_dataset = BuildDataSet(dev_features)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataset = BuildDataSet(test_features)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    bert_model = BertLSMTMClassification(config).to(config.device)
    model_train(config, bert_model, train_loader, dev_loader)
    model_test(config, bert_model, test_loader)
    model_save(bert_model, config.save_path)


if __name__ == '__main__':

    config = Electirc_NER_Config()
    logging_filename = None
    if config.is_logging2file is True:
        file = time.strftime('%Y-%m-%d_%H:%M:%S') + '.log'
        logging_filename = os.path.join(config.logging_dir, file)
        if not os.path.exists(config.logging_dir):
            os.makedirs(config.logging_dir)

    logging.basicConfig(filename=logging_filename, format='%(levelname)s: %(message)s', level=logging.INFO)

    Electirc_ner_task(config)


