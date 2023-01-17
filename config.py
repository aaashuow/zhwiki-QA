class Config:
    def __init__(self):
        #  mode selection
        self.do_train = True
        self.do_test=False
        self.do_predict=False
        #
        self.vocab_size=21128
        self.seed = [123]

        # 数据路径参数
        self.train_dir = './CMRC2018'
        self.dev_dir = './CMRC2018'
        self.test_dir= './CMRC2018'
        self.train_file = 'cmrc2018_train.json'
        self.dev_file = 'cmrc2018_dev.json'
        self.test_file = 'cmrc2018_test.json'

        # parameters for train
        self.output_dir = 'model_output'
        self.saved_model = "macbert-base.bin"
        self.do_lower_case = True
        self.max_seq_len = 512
        self.train_batch_size = 32
        self.dev_batch_size = 32
        self.test_batch_size = 32
        self.num_train_epochs = 10
        self.lr = 3e-5              # 学习率
        self.schedule = 'warmup_linear'
        self.warmup_rate=0.05
        self.clip_norm=1.0
        self.weight_decay_rate=0.01
        self.float16=False

        self.prediction_dir = 'predition'
        self.bert_model = 'hfl/chinese-macbert-base'
        self.device = None
        self.learning_rate = 3e-5
        self.clip_grad = 1
        self.checkpoint = 50
        self.export_model = True
        self.hidden_size = 312
        self.weight_start = 1
        self.weight_end = 1
        self.warmup_steps_ratio = 0.1
        # optimizer: BERTAdam()
        self.dropout_prob = 0.1
        self.use_ori_albert = True

        # param for dev/predict/eval
        self.eval_epochs = 0.5

        # param for logging
        self.log_file = './log/log.txt'

        # other bert pretrained model
        # self.bert_model = './model_hub/voidful-albert-chinese-tiny/'

