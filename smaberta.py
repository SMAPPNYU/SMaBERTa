#!/usr/bin/env python
# coding: utf-8
# Full credit to simpletransformers v0.6

from __future__ import absolute_import, division, print_function

import os
import random
import json

import numpy as np
from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer, 
                                  PreTrainedTokenizer,
                                  PreTrainedModel,
                                  AutoModelWithLMHead)

from transformers import AdamW, get_linear_schedule_with_warmup

from simpletransformers.classification.classification_utils import (convert_examples_to_features, InputExample)
from typing import Dict, List, Tuple
import math
from tensorboardX import SummaryWriter
from tqdm import trange, tqdm
from multiprocessing import cpu_count

import logging
logger = logging.getLogger(__name__)

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args, mlm_probability=0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


class TransformerModel:
    def __init__(self, model_type, model_name, finetune=False, num_labels=2, args=None, use_cuda=True, location=""):
        """
        Initializes a Transformer model.
        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            num_labels (optional): The number of labels or classes in the dataset.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
        """

        MODEL_CLASSES = {
                    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
                    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
                    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
                    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
                    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
                }

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if location=="":
            self.tokenizer = tokenizer_class.from_pretrained(model_name)
            if finetune:
                self.model=AutoModelWithLMHead.from_pretrained(model_name)
            else:
                self.model = model_class.from_pretrained(model_name, num_labels=num_labels)            
        else:
            self.tokenizer = tokenizer_class.from_pretrained(location)
            if finetune:
                self.model=AutoModelWithLMHead.from_pretrained(location)
            else:    
                self.model = model_class.from_pretrained(location, num_labels=num_labels)
       
        if use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        
        self.results = {}

        self.args = {
            'output_dir': 'outputs/',
            'cache_dir': 'cache_dir',
            'fp16': True,
            'fp16_opt_level': 'O1',
            'max_seq_length': 128,
            'train_batch_size': 25,
            'finetune_batch_size': 4,
            'gradient_accumulation_steps': 1,
            'eval_batch_size': 50,
            'finetune_eval_batch_size': 4,
            'num_train_epochs': 1,
            'num_finetune_epochs': 1,
            'weight_decay': 0,
            'learning_rate': 4e-5,
            'finetune_learning_rate': 5e-5,
            'adam_epsilon': 1e-8,
            'warmup_ratio': 0.06,
            'warmup_steps': 0,
            'max_grad_norm': 1.0,
            'mlm': True,
            'logging_steps': 50,
            'finetune_logging_steps': 100,
            'save_steps': 2000,
            'finetune_save_steps': 500,
            'overwrite_output_dir': False,
            'reprocess_input_data': False,
            'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
            'device': self.device,
            'model_name_or_path': False,
        }

        if args:
            self.args.update(args)

        if use_cuda:
            self.args['n_gpu'] : torch.cuda.device_count()

        self.args['model_name'] = model_name
        self.args['model_type'] = model_type

    def train_model(self, train_df, output_dir=None, show_running_loss=False, args=None):
        """
        Trains the model using 'train_df'
        Args:
            train_df: Pandas Dataframe (no header) of two columns, first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
        Returns:
            None
        """
        if args:
            self.args.update(args)

        if not output_dir:
            output_dir = self.args['output_dir']

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args['overwrite_output_dir']:
            raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(output_dir))
        
        self.model.to(self.device)

        train_examples = [InputExample(i, text, None, label) for i, (text, label) in enumerate(zip(train_df.iloc[:, 0], train_df.iloc[:, 1]))]

        train_dataset = self.load_and_cache_examples(train_examples)
        global_step, tr_loss = self.train(train_dataset, output_dir, show_running_loss=show_running_loss)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))

        print(f'Training of {self.args["model_type"]} model complete. Saved to {output_dir}.')

    def eval_model(self, eval_df, output_dir=None, verbose=False, **kwargs):
        """
        Evaluates the model on eval_df. Saves results to output_dir.
        Args:
            eval_df: Pandas Dataframe (no header) of two columns, first column containing the text, and the second column containing the label. The model will be evaluated on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            model_outputs: List of model outputs for each row in eval_df
            wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model
        """

        if not output_dir:
            output_dir = self.args['output_dir']

        self.model.to(self.device)

        result, model_outputs, wrong_preds = self.evaluate(eval_df, output_dir, **kwargs)
        self.results.update(result)

        if not verbose:
            print(self.results)

        return result, model_outputs, wrong_preds

    def evaluate(self, eval_df, output_dir, prefix="", **kwargs):
        """
        Evaluates the model on eval_df.
        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        args = self.args
        eval_output_dir = output_dir

        results = {}

        eval_examples = [InputExample(i, text, None, label) for i, (text, label) in enumerate(zip(eval_df.iloc[:, 0], eval_df.iloc[:, 1]))]
        eval_dataset = self.load_and_cache_examples(eval_examples, evaluate=True)
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        #for batch in tqdm(eval_dataloader):
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args['model_type'] != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args['model_type'] in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds
        preds = np.argmax(preds, axis=1)
        result, wrong = self.compute_metrics(preds, out_label_ids, eval_examples, **kwargs)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

        return results, model_outputs, wrong

              
    def load_and_cache_examples(self, examples, evaluate=False, no_cache=False):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.
        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        process_count = self.args['process_count']

        tokenizer = self.tokenizer
        output_mode = 'classification'
        args=self.args

        if not os.path.isdir(self.args['cache_dir']):
            os.mkdir(self.args['cache_dir'])

        mode = 'dev' if evaluate else 'train'
        cached_features_file = os.path.join(args['cache_dir'], f"cached_{mode}_{args['model_type']}_{args['max_seq_length']}_binary")

        if os.path.exists(cached_features_file) and not args['reprocess_input_data'] and not no_cache:
            features = torch.load(cached_features_file)

        else:
            features = convert_examples_to_features(examples, args['max_seq_length'], tokenizer, output_mode,
                                                    # xlnet has a cls token at the end
                                                    cls_token_at_end=bool(args['model_type'] in ['xlnet']),
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if args['model_type'] in ['xlnet'] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                    sep_token_extra=bool(args['model_type'] in ['roberta']),
                                                    # pad on the left for xlnet
                                                    pad_on_left=bool(args['model_type'] in ['xlnet']),
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0,
                                                    process_count=process_count, silent=True)

            if not no_cache:
                torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset


    def train(self, train_dataset, output_dir, show_running_loss=True):
        """
        Trains the model on train_dataset.
        Utility function to be used by the train_model() method. Not intended to be used directly.
        """
        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        args = self.args
        tb_writer = SummaryWriter()
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])
        
        t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        warmup_steps = math.ceil(t_total * args['warmup_ratio'])
        args['warmup_steps'] = warmup_steps if args['warmup_steps'] == 0 else args['warmup_steps']

        optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=t_total)

        if args['fp16']:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = range(int(args['num_train_epochs']))#, desc="Epoch")
        ctr = 0
        for _ in train_iterator:
            print("Starting Epoch: ", ctr)
            ctr+=1
            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(train_dataloader):#, desc="Current iteration"):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'labels':         batch[3]}
                # XLM, DistilBERT and RoBERTa don't use segment_ids
                if args['model_type'] != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args['model_type'] in ['bert', 'xlnet'] else None  
                outputs = model(**inputs)
                # model outputs are always tuple in pytorch-transformers (see doc)
                loss = outputs[0]
                if show_running_loss:
                    print("\rRunning loss: %f" % loss, end='')

                if args['gradient_accumulation_steps'] > 1:
                    loss = loss / args['gradient_accumulation_steps']

                if args['fp16']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])

                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

                tr_loss += loss.item()
                if (step + 1) % args['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                        # Log metrics
                        # Only evaluate when single GPU otherwise metrics may not average well
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args['logging_steps'], global_step)
                        logging_loss = tr_loss

                    if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(
                            output_dir, 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        model_to_save.save_pretrained(output_dir)
        return global_step, tr_loss / global_step

                                            
    def compute_metrics(self, preds, labels, eval_examples, **kwargs):
        """
        Computes the evaluation metrics for the model predictions.
        Args:
            preds: Model predictions
            labels: Ground truth labels
            eval_examples: List of examples on which evaluation was performed
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            wrong: List of InputExample objects corresponding to each incorrect prediction by the model
        """
        assert len(preds) == len(labels)

        mcc = matthews_corrcoef(labels, preds)

        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(labels, preds)

        mismatched = labels != preds
        wrong = [i for (i, v) in zip(eval_examples, mismatched) if v]
        
        if self.model.num_labels == 2:
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            return {**{
                "mcc": mcc,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn
            }, **extra_metrics}, wrong
        
        else:
            return {**{"mcc": mcc}, **extra_metrics}, wrong

    def predict(self, to_predict):
        """
        Performs predictions on a list of text.
        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.
        Returns:
            preds: A python list of the predictions (0 or 1) for each text.
            model_outputs: A python list of the raw model outputs for each text.
        """

        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        args = self.args

        self.model.to(self.device)

        eval_examples = [InputExample(i, text, None, 0) for i, text in enumerate(to_predict)]

        eval_dataset = self.load_and_cache_examples(eval_examples, evaluate=True, no_cache=True)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        #for batch in tqdm(eval_dataloader):
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        # XLM don't use segment_ids
                        'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,
                        'labels':         batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds
        preds = np.argmax(preds, axis=1)

        return preds, model_outputs
        
    def finetune(self, train_file_path, eval_file_path):
        model = self.model
        tokenizer = self.tokenizer
        args = self.args
        print(args)
        #print("Starting model finetuning")
        train_dataset = LineByLineTextDataset(tokenizer, file_path=train_file_path)
        """ Train the model """
        tb_writer = SummaryWriter()

        def collate(examples: List[torch.Tensor]):
            if tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args["finetune_batch_size"], collate_fn=collate
        )

        t_total = len(train_dataloader) // args["num_finetune_epochs"]

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args["weight_decay"],
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args["finetune_learning_rate"], eps=args["adam_epsilon"])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            args["model_name_or_path"]
            and os.path.isfile(os.path.join(args["model_name_or_path"], "optimizer.pt"))
            and os.path.isfile(os.path.join(args["model_name_or_path"], "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args["model_name_or_path"], "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args["model_name_or_path"], "scheduler.pt")))

        if args["fp16"]:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        #if args["n_gpu"] > 1:
        #    model = torch.nn.DataParallel(model)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args["num_finetune_epochs"])
        logger.info("  Instantaneous batch size per GPU = %d", args["finetune_batch_size"])
        logger.info("  Gradient Accumulation steps = %d", 1)
        logger.info("  Total optimization steps = %d", t_total)
        #print("Beginning")
        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if args["model_name_or_path"] and os.path.exists(args["model_name_or_path"]):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args["model_name_or_path"].split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader))
                steps_trained_in_current_epoch = global_step % (len(train_dataloader))

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("  Starting fine-tuning.")

        tr_loss, logging_loss = 0.0, 0.0

        model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
        model_to_resize.resize_token_embeddings(len(tokenizer))
        model.to(self.device)
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(args["num_train_epochs"]), desc="Epoch", disable=False)
        #set_seed(args)  # Added here for reproducibility
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs, labels = mask_tokens(batch, tokenizer, args) if args["mlm"] else (batch, batch)
                inputs = inputs.to(args["device"])
                labels = labels.to(args["device"])
                model.train()
                outputs = model(inputs, masked_lm_labels=labels) if args["mlm"] else model(inputs, labels=labels)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                #if args["n_gpu"] > 1:
                #    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if args["fp16"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()

                if args["fp16"]:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args["max_grad_norm"])
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % args["finetune_logging_steps"] == 0:
                    # Log metrics
                    results = self.lm_evaluate(eval_file_path)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args["finetune_logging_steps"], global_step)
                    logging_loss = tr_loss

                if global_step % args["finetune_save_steps"] == 0 or global_step==t_total-1:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args["output_dir"], "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    #_rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

                #if args.max_steps > 0 and global_step > args.max_steps:
                #    epoch_iterator.close()
                #    break
            #if args.max_steps > 0 and global_step > args.max_steps:
            #    train_iterator.close()
            #    break

        #if args.local_rank in [-1, 0]:
        tb_writer.close()

        return global_step, tr_loss / global_step

    def lm_evaluate(self, eval_file_path, prefix="") -> Dict:
        model = self.model
        tokenizer = self.tokenizer
        args = self.args
        #print(args)
        #print("Starting evaluation")
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_output_dir = args["output_dir"]

        eval_dataset = LineByLineTextDataset(tokenizer, file_path=eval_file_path)

        #if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

        def collate(examples: List[torch.Tensor]):
            if tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args["finetune_eval_batch_size"], collate_fn=collate
        )

        # multi-gpu evaluate
        #if args.n_gpu > 1:
        #    model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args["finetune_eval_batch_size"])
        eval_loss = 0.0
        nb_eval_steps = 0
        model.to(self.device)
        model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs, labels = mask_tokens(batch, tokenizer, args) if args["mlm"] else (batch, batch)
            inputs = inputs.to(args["device"])
            labels = labels.to(args["device"])

            with torch.no_grad():
                outputs = model(inputs, masked_lm_labels=labels) if args["mlm"] else model(inputs, labels=labels)
                lm_loss = outputs[0]
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        result = {"perplexity": perplexity}
        print("Evaluation perplexity: ", result)
        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        return result

