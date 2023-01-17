import os
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
# from qa_utils import read_qa_examples, convert_examples_to_features
from preprocess.cmrc2018_preprocess import json2features
import json
from config import Config

class CMRCDataLoader:
    def __init__(self, config, tokenizer, mode="train"):
        self.tokenizer = tokenizer
        self.do_lower_case = config.do_lower_case
        self.max_seq_len = config.max_seq_len

        if mode == "train":
            self.train_batch_size = config.train_batch_size
            self.dev_batch_size = config.dev_batch_size
            self.test_batch_size = config.test_batch_size
            self.train_num_epochs = config.num_train_epochs
        if mode == "dev":
            self.test_batch_size = config.test_batch_size


        # self.data_cache = config.data_cache
        self.num_train_epochs = config.num_train_epochs
        self.num_train_instances = 0
        self.num_dev_instances = 0
        self.num_test_instances = 0

        self.data_dir={
            'train':config.train_dir,
            'dev':config.dev_dir,
            'test':config.test_dir
        }
        self.data_file={
            'train':config.train_file,
            'dev':config.dev_file,
            'test':config.test_file
        }

    def convert_examples_to_features(self, data_sign="train"):
        print("loading {} data ... ...".format(data_sign))

        # if data_sign == "train":
        #     examples = read_qa_examples(self.data_dir, 'cmrc2018_train_squad.json', self.max_seq_len)
        #     self.num_train_instances = len(examples)
        # elif data_sign == "dev":
        #     examples = read_qa_examples(self.data_dir, 'cmrc2018_dev_squad_clean.json', self.max_seq_len)
        #     self.num_dev_instances = len(examples)
        # elif data_sign == "test":
        #     examples = read_qa_examples(self.data_dir, 'cmrc2018_trial_squad.json', self.max_seq_len)
        #     self.num_test_instances = len(examples)
        # else:
        #     raise ValueError("please notice that the data_sign can only be train/dev/test !!")

        features_file_name = data_sign+"_features_"+str(self.max_seq_len)+'.json'
        examples_file_name = features_file_name.replace('_features_', '_examples_')

        data_dir = self.data_dir[data_sign]
        data_file = os.path.join(data_dir,self.data_file[data_sign])

        features_path = os.path.join(data_dir,features_file_name)
        examples_path = os.path.join(data_dir,examples_file_name)

        # convert json file to feature
        if not os.path.exists(examples_path):
            print('converting json file to features...')
            json2features(data_file, [examples_path,features_path],
                          self.tokenizer, is_training=True,
                          max_seq_length=self.max_seq_len)
        else:
            print('features files already exits.')
        """
        假如是train_dir=train_features_512.json，train_file="cmrc2018_train.json"
        经过上面的操作以后，对应目录中会出现
        train_features_512.json
        和 
        train_examples_512.json
        两个文件
        """
        with open(examples_path, 'r', encoding="utf-8") as f:
            examples = json.load(f)

        with open(features_path, 'r', encoding="utf-8") as f:
            features = json.load(f)

        if data_sign == "train":
            self.num_train_instances = len(examples)
            self.num_train_features = len(features)

        elif data_sign == "dev":
            self.num_dev_instances = len(examples)
        elif data_sign == "test":
            self.num_test_instances = len(examples)
        else:
            raise ValueError("please notice that the data_sign can only be train/dev/test !!")


        return examples,features

    def get_dataloader(self, data_sign="train",):

        _,features = self.convert_examples_to_features(data_sign=data_sign)

        print(f"{len(features)} {data_sign} data loaded")

        input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        input_mask = torch.tensor([f['input_mask'] for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f['segment_ids'] for f in features], dtype=torch.long)

        if data_sign == "train":
            start_pos = torch.tensor([f['start_position'] for f in features], dtype=torch.long)
            end_pos = torch.tensor([f['end_position'] for f in features], dtype=torch.long)
            dataset = TensorDataset(input_ids, input_mask, segment_ids, start_pos, end_pos)
            # datasampler = RandomSampler(dataset)  # RandomSampler(dataset) 可能需要采样再训练
            dataloader = DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=2)
        elif data_sign == "dev":
            example_index = torch.arange(input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(input_ids, input_mask, segment_ids,example_index)
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.dev_batch_size)
        elif data_sign == "test":
            example_index = torch.arange(input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(input_ids, input_mask, segment_ids, example_index)
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size)
        return dataloader

    def get_num_train_epochs(self):
        return int((self.num_train_instances / self.train_batch_size) * self.num_train_epochs)

    def get_steps_per_epoch(self):
        return self.num_train_features // self.train_batch_size


if __name__ == '__main__':
    # test my dataloader
    from config import Config
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('./model_hub/voidful-albert-chinese-tiny/')
    myconfig = Config()
    crmcDataLoader = CMRCDataLoader(myconfig, tokenizer, mode="train")
    train_loader = crmcDataLoader.get_dataloader(data_sign="train")
    for i,batch in enumerate(train_loader):
        print(batch[0].shape)
        break
    dev_loader = crmcDataLoader.get_dataloader(data_sign="dev")
    for i, batch in enumerate(dev_loader):
        print(batch[0].shape)
        break
    test_loader = crmcDataLoader.get_dataloader(data_sign="test")
    for i, batch in enumerate(test_loader):
        print(batch[0].shape)
        break

