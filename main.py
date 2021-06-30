import os
import sys
import pickle
import random
import copy
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm, trange
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

from utils import Config, Logger, make_log_dir
from modeling import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification_SPV,
    AutoModelForSequenceClassification_MIP,
    AutoModelForSequenceClassification_SPV_MIP,
)
from run_classifier_dataset_utils import processors, output_modes, compute_metrics
from data_loader import load_train_data, load_train_data_kf, load_test_data

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
ARGS_NAME = "training_args.bin"


def main():
    # read configs
    config = Config(main_conf_path="./")

    # apply system arguments if exist
    argv = sys.argv[1:]
    if len(argv) > 0:
        cmd_arg = OrderedDict()
        argvs = " ".join(sys.argv[1:]).split(" ")
        for i in range(0, len(argvs), 2):
            arg_name, arg_value = argvs[i], argvs[i + 1]
            arg_name = arg_name.strip("-")
            cmd_arg[arg_name] = arg_value
        config.update_params(cmd_arg)

    args = config
    print(args.__dict__)

    # logger
    if "saves" in args.bert_model:
        log_dir = args.bert_model
        logger = Logger(log_dir)
        config = Config(main_conf_path=log_dir)
        old_args = copy.deepcopy(args)
        args.__dict__.update(config.__dict__)

        args.bert_model = old_args.bert_model
        args.do_train = old_args.do_train
        args.data_dir = old_args.data_dir
        args.task_name = old_args.task_name

        # apply system arguments if exist
        argv = sys.argv[1:]
        if len(argv) > 0:
            cmd_arg = OrderedDict()
            argvs = " ".join(sys.argv[1:]).split(" ")
            for i in range(0, len(argvs), 2):
                arg_name, arg_value = argvs[i], argvs[i + 1]
                arg_name = arg_name.strip("-")
                cmd_arg[arg_name] = arg_value
            config.update_params(cmd_arg)
    else:
        if not os.path.exists("saves"):
            os.mkdir("saves")
        log_dir = make_log_dir(os.path.join("saves", args.bert_model))
        logger = Logger(log_dir)
        config.save(log_dir)
    args.log_dir = log_dir

    # set CUDA devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logger.info("device: {} n_gpu: {}".format(device, args.n_gpu))

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # get dataset and processor
    task_name = args.task_name.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    args.num_labels = len(label_list)

    # build tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = load_pretrained_model(args)

    ########### Training ###########
    # VUA-18 / VUA-20
    if args.do_train and args.task_name == "vua":
        train_dataloader = load_train_data(
            args, logger, processor, task_name, label_list, tokenizer, output_mode
        )
        model, best_result = run_train(
            args,
            logger,
            model,
            train_dataloader,
            processor,
            task_name,
            label_list,
            tokenizer,
            output_mode,
        )

    # TroFi / MOH-X (K-fold)
    elif args.do_train and args.task_name == "trofi":
        k_result = []
        for k in tqdm(range(args.kfold), desc="K-fold"):
            model = load_pretrained_model(args)
            train_dataloader = load_train_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            model, best_result = run_train(
                args,
                logger,
                model,
                train_dataloader,
                processor,
                task_name,
                label_list,
                tokenizer,
                output_mode,
                k,
            )
            k_result.append(best_result)

        # Calculate average result
        avg_result = copy.deepcopy(k_result[0])
        for result in k_result[1:]:
            for k, v in result.items():
                avg_result[k] += v
        for k, v in avg_result.items():
            avg_result[k] /= len(k_result)

        logger.info(f"-----Averge Result-----")
        for key in sorted(avg_result.keys()):
            logger.info(f"  {key} = {str(avg_result[key])}")

    # Load trained model
    if "saves" in args.bert_model:
        model = load_trained_model(args, model, tokenizer)

    ########### Inference ###########
    # VUA-18 / VUA-20
    if (args.do_eval or args.do_test) and task_name == "vua":
        # if test data is genre or POS tag data
        if ("genre" in args.data_dir) or ("pos" in args.data_dir):
            if "genre" in args.data_dir:
                targets = ["acad", "conv", "fict", "news"]
            elif "pos" in args.data_dir:
                targets = ["adj", "adv", "noun", "verb"]
            orig_data_dir = args.data_dir
            for idx, target in tqdm(enumerate(targets)):
                logger.info(f"====================== Evaluating {target} =====================")
                args.data_dir = os.path.join(orig_data_dir, target)
                all_guids, eval_dataloader = load_test_data(
                    args, logger, processor, task_name, label_list, tokenizer, output_mode
                )
                run_eval(args, logger, model, eval_dataloader, all_guids, task_name)
        else:
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode
            )
            run_eval(args, logger, model, eval_dataloader, all_guids, task_name)

    # TroFi / MOH-X (K-fold)
    elif (args.do_eval or args.do_test) and args.task_name == "trofi":
        logger.info(f"***** Evaluating with {args.data_dir}")
        k_result = []
        for k in tqdm(range(10), desc="K-fold"):
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            result = run_eval(args, logger, model, eval_dataloader, all_guids, task_name)
            k_result.append(result)

        # Calculate average result
        avg_result = copy.deepcopy(k_result[0])
        for result in k_result[1:]:
            for k, v in result.items():
                avg_result[k] += v
        for k, v in avg_result.items():
            avg_result[k] /= len(k_result)

        logger.info(f"-----Averge Result-----")
        for key in sorted(avg_result.keys()):
            logger.info(f"  {key} = {str(avg_result[key])}")
    logger.info(f"Saved to {logger.log_dir}")


def run_train(
    args,
    logger,
    model,
    train_dataloader,
    processor,
    task_name,
    label_list,
    tokenizer,
    output_mode,
    k=None,
):
    tr_loss = 0
    num_train_optimization_steps = len(train_dataloader) * args.num_train_epoch

    # Prepare optimizer, scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.lr_schedule != False or args.lr_schedule.lower() != "none":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_epoch * len(train_dataloader)),
            num_training_steps=num_train_optimization_steps,
        )

    logger.info("***** Running training *****")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Num steps = { num_train_optimization_steps}")

    # Run training
    model.train()
    max_val_f1 = -1
    max_result = {}
    for epoch in trange(int(args.num_train_epoch), desc="Epoch"):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # move batch data to gpu
            batch = tuple(t.to(args.device) for t in batch)

            if args.model_type in ["MELBERT_MIP", "MELBERT"]:
                (
                    input_ids,
                    input_mask,
                    segment_ids,
                    label_ids,
                    input_ids_2,
                    input_mask_2,
                    segment_ids_2,
                ) = batch
            else:
                input_ids, input_mask, segment_ids, label_ids = batch

            # compute loss values
            if args.model_type in ["BERT_SEQ", "BERT_BASE", "MELBERT_SPV"]:
                logits = model(
                    input_ids,
                    target_mask=(segment_ids == 1),
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
            elif args.model_type in ["MELBERT_MIP", "MELBERT"]:
                logits = model(
                    input_ids,
                    input_ids_2,
                    target_mask=(segment_ids == 1),
                    target_mask_2=segment_ids_2,
                    attention_mask_2=input_mask_2,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))

            # average loss if on multi-gpu.
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if args.lr_schedule != False or args.lr_schedule.lower() != "none":
                scheduler.step()

            optimizer.zero_grad()

            tr_loss += loss.item()

        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[epoch {epoch+1}] ,lr: {cur_lr} ,tr_loss: {tr_loss}")

        # evaluate
        if args.do_eval:
            all_guids, eval_dataloader = load_test_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            result = run_eval(args, logger, model, eval_dataloader, all_guids, task_name)

            # update
            if result["f1"] > max_val_f1:
                max_val_f1 = result["f1"]
                max_result = result
                if args.task_name == "trofi":
                    save_model(args, model, tokenizer)
            if args.task_name == "vua":
                save_model(args, model, tokenizer)

    logger.info(f"-----Best Result-----")
    for key in sorted(max_result.keys()):
        logger.info(f"  {key} = {str(max_result[key])}")

    return model, max_result


def run_eval(args, logger, model, eval_dataloader, all_guids, task_name, return_preds=False):
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    pred_guids = []
    out_label_ids = None

    for eval_batch in tqdm(eval_dataloader, desc="Evaluating"):
        eval_batch = tuple(t.to(args.device) for t in eval_batch)

        if args.model_type in ["MELBERT_MIP", "MELBERT"]:
            (
                input_ids,
                input_mask,
                segment_ids,
                label_ids,
                idx,
                input_ids_2,
                input_mask_2,
                segment_ids_2,
            ) = eval_batch
        else:
            input_ids, input_mask, segment_ids, label_ids, idx = eval_batch

        with torch.no_grad():
            # compute loss values
            if args.model_type in ["BERT_BASE", "BERT_SEQ", "MELBERT_SPV"]:
                logits = model(
                    input_ids,
                    target_mask=(segment_ids == 1),
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                    pred_guids.append([all_guids[i] for i in idx])
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                    pred_guids[0].extend([all_guids[i] for i in idx])
                    out_label_ids = np.append(
                        out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                    )

            elif args.model_type in ["MELBERT_MIP", "MELBERT"]:
                logits = model(
                    input_ids,
                    input_ids_2,
                    target_mask=(segment_ids == 1),
                    target_mask_2=segment_ids_2,
                    attention_mask_2=input_mask_2,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                    pred_guids.append([all_guids[i] for i in idx])
                    out_label_ids = label_ids.detach().cpu().numpy()
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                    pred_guids[0].extend([all_guids[i] for i in idx])
                    out_label_ids = np.append(
                        out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                    )

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)
    print(preds, out_label_ids)

    # compute metrics
    result = compute_metrics(preds, out_label_ids)

    for key in sorted(result.keys()):
        logger.info(f"  {key} = {str(result[key])}")

    if return_preds:
        return preds
    return result


def load_pretrained_model(args):
    # Pretrained Model
    bert = AutoModel.from_pretrained(args.bert_model)
    config = bert.config
    config.type_vocab_size = 4
    if "albert" in args.bert_model:
        bert.embeddings.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_size
        )
    else:
        bert.embeddings.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
    bert._init_weights(bert.embeddings.token_type_embeddings)

    # Additional Layers
    if args.model_type in ["BERT_BASE"]:
        model = AutoModelForSequenceClassification(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "BERT_SEQ":
        model = AutoModelForTokenClassification(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT_SPV":
        model = AutoModelForSequenceClassification_SPV(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT_MIP":
        model = AutoModelForSequenceClassification_MIP(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT":
        model = AutoModelForSequenceClassification_SPV_MIP(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )

    model.to(args.device)
    if args.n_gpu > 1 and not args.no_cuda:
        model = torch.nn.DataParallel(model)
    return model


def save_model(args, model, tokenizer):
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.log_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.log_dir)

    # Good practice: save your training arguments together with the trained model
    output_args_file = os.path.join(args.log_dir, ARGS_NAME)
    torch.save(args, output_args_file)


def load_trained_model(args, model, tokenizer):
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)

    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(output_model_file))
    else:
        model.load_state_dict(torch.load(output_model_file))

    return model


if __name__ == "__main__":
    main()
