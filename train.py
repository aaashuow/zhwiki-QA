
from config import Config
from preprocess.cmrc2018_output import write_predictions
from preprocess.cmrc2018_preprocess import json2features
import argparse
import collections
import json
import os
import random

import numpy as np
import torch
# from google_albert_pytorch_modeling import AlbertConfig, AlbertForMRC
from preprocess.cmrc2018_evaluate import get_eval
# from pytorch_modeling import BertConfig, BertForQuestionAnswering, ALBertConfig, ALBertForQA  # pytorch_model 是？
from transformers import AlbertForQuestionAnswering, AutoConfig, BertTokenizer,AutoModelForQuestionAnswering,AutoTokenizer

from tools.pytorch_optimization import get_optimization, warmup_linear  # 优化器和预热器
from tools import official_tokenization as tokenization, utils  # 需要tokenization 的导入

from tools.pytorch_optimization import get_optimization, warmup_linear  # 优化器

from tqdm import tqdm
from data_loader import CMRCDataLoader
import pandas as pd

def load_data(config):
    """
    we need a tokenizer and some dataloaders
    """

    tokenizer = AutoTokenizer.from_pretrained(config.bert_model,return_dict=False) # return_dict controls

    assert config.vocab_size==len(tokenizer.vocab)

    # 最好是重新写一个dataloader
    dataset_loaders = CMRCDataLoader(config, tokenizer, mode="train", )
    train_dataloader = dataset_loaders.get_dataloader(data_sign="train")
    dev_dataloader = dataset_loaders.get_dataloader(data_sign="dev")
    test_dataloader = dataset_loaders.get_dataloader(data_sign="test")

    return dataset_loaders,train_dataloader, dev_dataloader, test_dataloader

def load_model(config,total_steps):
    # 加载模型
    # if config.use_ori_albert:
    #     model = AutoModelForQuestionAnswering.from_pretrained(config.bert_model)
    # else:
        # model = BertQA(config)

    model = AutoModelForQuestionAnswering.from_pretrained(config.bert_model)

    import tools.utils as utils
    utils.torch_show_all_params(model)
    # utils.torch_init_model(model, config.bert_model)

    model.to(config.device)

    param_optimizer = list(model.named_parameters())

    # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
    #     {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=10e-8)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps_ratio * t_total,
    #                                             num_training_steps=t_total)

    optimizer = get_optimization(model=model,
                                 float16=config.float16,
                                 learning_rate=config.lr,
                                 total_steps=total_steps,
                                 schedule=config.schedule,
                                 warmup_rate=config.warmup_rate,
                                 max_grad_norm=config.clip_norm,
                                 weight_decay_rate=config.weight_decay_rate)

    return model, optimizer


def train(config):
    # prepare the GPUs
    args.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {}".format(device, n_gpu))

    # load data
    print('loading data...')
    dataset_loaders,train_dataloader, \
    dev_dataloader, test_dataloader, \
     = load_data(config)

    total_steps = dataset_loaders.get_num_train_epochs()
    steps_per_epoch = dataset_loaders.get_steps_per_epoch()
    eval_steps = int(steps_per_epoch*config.eval_epochs)
    dev_examples,dev_features = dataset_loaders.convert_examples_to_features(data_sign="dev")

    F1s = []
    EMs = []

    best_f1_em = 0

    for seed_ in args.seed:

        with open(config.log_file, 'a') as aw:
            aw.write('===================================' +
                     'SEED:' + str(seed_)
                     + '===================================' + '\n')
        print('SEED:', seed_)

        random.seed(seed_)
        np.random.seed(seed_)
        torch.manual_seed(seed_)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed_)
        print('loading model and optimizer ...')
        model,optimizer = load_model(config,total_steps)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        print('-------------- Training ----------------')
        global_steps = 1
        best_f1, best_em = 0, 0

        model.train()

        for i in range(int(int(config.num_train_epochs))):
            print('Strating epoch %d'%(i+1))
            total_loss = 0
            iteration = 1

            with tqdm(total=steps_per_epoch,desc='Epoch %d'%(i+1)) as pbar:

                for step,batch in enumerate(train_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids,input_mask,segment_ids,start_positions,end_positions = batch
                    # 进行训练
                    output = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                   start_positions=start_positions, end_positions=end_positions)
                    # 求损失函数
                    loss = output.loss

                    if n_gpu > 1:
                        loss = loss.mean()
                    total_loss += loss.item()

                    pbar.set_postfix({'train_loss': '{0:1.5f}'.format(total_loss / (iteration + 1e-5))})
                    pbar.update(1)

                    loss.backward() #
                    model.zero_grad() #
                    optimizer.step() #

                    iteration+=1
                    global_steps+=1


                    if global_steps % eval_steps == 0:
                        print("【train】 epoch:{}/{} step:{}/{} loss:{}".format(
                            i + 1, config.num_train_epochs, global_steps, iteration, loss.item()
                        ))
                        # evaluate 这个部分要再修改一下
                        best_f1, best_em, _ = evaluate(model, args, dev_examples, dev_features, dev_dataloader,
                                                                            device, global_steps, best_f1, best_em, best_f1_em)

                        print("【dev】 f1:{} EM:{}".format(best_f1,best_em))
                        F1s.append(best_f1) # 记录每个step的F1和EM值
                        EMs.append(best_em)


            # release the memory
            del model
            del optimizer
            torch.cuda.empty_cache()

        score = pd.DataFrame([F1s,EMs],columns=['F1','EM'])
        score.to_csv('score.csv',index=None)

        print('Mean F1:', np.mean(F1s), 'Mean EM:', np.mean(EMs))
        print('Best F1:', np.max(F1s), 'Best EM:', np.max(EMs))
        with open(args.log_file, 'a') as aw:
            aw.write('Mean(Best) F1:{}({})\n'.format(np.mean(F1s), np.max(F1s)))
            aw.write('Mean(Best) EM:{}({})\n'.format(np.mean(EMs), np.max(EMs)))
def myevaluate(model, args, eval_examples, eval_features, eval_dataloader,device, global_steps, best_f1, best_em, best_f1_em):
    print("***** Eval *****")

    eval_loss = 0
    eval_steps = 0

    for input_ids, input_mask, segment_ids, start_pos, end_pos in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_pos = start_pos.to(device)
        end_pos = end_pos.to(device)

        with torch.no_grad():
            output = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
            start_positions=start_pos, end_positions=end_pos)
            tmp_eval_loss = output.loss
            start_logits = output.start_logits
            end_logits = output.end_logits
            start_logits = torch.argmax(start_logits, 1)
            end_logits = torch.argmax(end_logits, 1)

        start_pos = start_pos.to("cpu").numpy().tolist()
        end_pos = end_pos.to("cpu").numpy().tolist()

        start_label = start_logits.detach().cpu().numpy()
        end_label = end_logits.detach().cpu().numpy()
        input_mask = input_mask.to("cpu").detach().numpy().tolist()

        eval_loss += tmp_eval_loss.mean().item()
        # mask_lst += input_mask
        eval_steps += 1

        # start_pred_lst += start_label
        # end_pred_lst += end_label
        #
        # start_gold_lst += start_pos
        # end_gold_lst += end_pos
        from decode_utils import decode_qa, calculate_metric, get_p_r_f  # utils文件 -

        pred_res = decode_qa(start_label, end_label)
        true_res = decode_qa(start_pos, end_pos)
        tp, fp, fn = calculate_metric(true_res, pred_res)
        tp_all += tp
        fp_all += fp
        fn_all += fn

    eval_precision, eval_recall, eval_f1 = get_p_r_f(tp_all, fp_all, fn_all)
    average_loss = round(eval_loss / eval_steps, 4)
    eval_f1 = round(eval_f1, 4)
    eval_precision = round(eval_precision, 4)
    eval_recall = round(eval_recall, 4)

    return average_loss, eval_precision, eval_recall, eval_f1




def evaluate(model, args, eval_examples, eval_features, eval_dataloader,device, global_steps, best_f1, best_em, best_f1_em):
    print("***** Eval *****")
    """
    模型评估函数

    """
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])

    output_prediction_file = os.path.join(args.checkpoint_dir,
                                          "predictions_steps" + str(global_steps) + ".json")

    output_nbest_file = output_prediction_file.replace('predictions', 'nbest')


    model.eval()
    all_results = []
    print("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_pos,end_pos = example_indices

        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask) # 函数重载
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, start_pos, end_pos)

        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature['unique_id'])
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    write_predictions(eval_examples, eval_features, all_results,
                      n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                      do_lower_case=True, output_prediction_file=output_prediction_file,
                      output_nbest_file=output_nbest_file)

    tmp_result = get_eval(args.dev_file, output_prediction_file)
    tmp_result['STEP'] = global_steps
    with open(args.log_file, 'a') as aw:
        aw.write(json.dumps(tmp_result) + '\n')
    print(tmp_result)

    if float(tmp_result['F1']) > best_f1:
        best_f1 = float(tmp_result['F1'])

    if float(tmp_result['EM']) > best_em:
        best_em = float(tmp_result['EM'])

    if float(tmp_result['F1']) + float(tmp_result['EM']) > best_f1_em:
        best_f1_em = float(tmp_result['F1']) + float(tmp_result['EM'])

        modle_to_save = model.module if hasattr(model, 'module') else model
        model_name = "checkpoint_f1" + str(best_f1) + "_em" + str(best_em) + ".bin"
        output_model_dir = os.path.join(args.output_dir, model_name)
        torch.save(modle_to_save.state_dict(), output_model_dir)

        # utils.torch_save_model(model, args.checkpoint_dir,
        #                        {'f1': float(tmp_result['F1']), 'em': float(tmp_result['EM'])}, max_save_num=1)

    model.train()

    return best_f1, best_em, best_f1_em

# dev / test
# predict


if __name__=='__main__':
    args = Config()

    # select mode:
    do_train = args.do_train
    do_test=args.do_test
    do_predict = args.do_predict

    if do_train:
        train(args)



