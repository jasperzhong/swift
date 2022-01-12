import logging
import sys
import subprocess
import torch
import torch.distributed.fault_tolerance
import os
import json
from tokenization import get_tokenizer
import schedule
from Squad import read_squad_examples, convert_examples_to_features, RawResult, get_answers
from schedule import (is_pipeline_first_stage, is_pipeline_last_stage,
                      recv_forward, send_forward)
from torch.utils.data import (DataLoader, RandomSampler, 
                              SequentialSampler,TensorDataset)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_val_dataloader():
    tokenizer = get_tokenizer()
    args = schedule._GLOBAL_ARGS
    eval_examples = read_squad_examples(
            input_file=args.predict_file, is_training=False, version_2_with_negative=False)
    eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.test_batch_size, drop_last=True)

    return eval_dataloader, eval_features, eval_examples

def fault_tolerance_val(config, model, eval_loader, loss_func):
    args = schedule._GLOBAL_ARGS
    eval_dataloader, eval_features, eval_examples = create_val_dataloader()
    if os.path.exists("./eval_dataloader.pt"):
        eval_dataloader = torch.load("./eval_dataloader.pt")
    else:
        torch.save(eval_dataloader, "eval_dataloader.pt")
    if os.path.exists("./eval_features.pt"):
        eval_features = torch.load("./eval_features.pt")
    else:
        torch.save(eval_features, "eval_features.pt")
    if os.path.exists("./eval_examples.pt"):
        eval_features = torch.load("./eval_examples.pt")
    else:
        torch.save(eval_examples, "eval_examples.pt")

    model.eval()
    all_results = []
    test_iters = len(eval_dataloader)
    data_iter = iter(eval_dataloader)
    print("test iters:{}".format(test_iters))
    logger.info("***** Running Validation *****")
    all_results = []
    for _ in range(test_iters):
        with torch.no_grad():
            if is_pipeline_last_stage():
                results = forward(config, data_iter, model, eval_features)
                all_results.extend(results)
            else:
                forward(config, data_iter, model, eval_features)
    
    if is_pipeline_last_stage():
        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")

        answers, nbest_answers = get_answers(eval_examples, eval_features, all_results, args)
        with open(output_prediction_file, "w") as f:
            f.write(json.dumps(answers, indent=4) + "\n")
        with open(output_nbest_file, "w") as f:
            f.write(json.dumps(nbest_answers, indent=4) + "\n")

        eval_out = subprocess.check_output([sys.executable, args.eval_script,
                                                    args.predict_file, args.output_dir + "/predictions.json"])
        scores = str(eval_out).strip()
        exact_match = float(scores.split(":")[1].split(",")[0])
        f1 = float(scores.split(":")[2].split("}")[0])
        print("exact_match: {} F1: {}".format(exact_match, f1))

def forward(config, data_iter, model, eval_features):
    shape = (schedule._GLOBAL_ARGS.test_batch_size, *model.input_shape[1:])
    input_tensor = recv_forward(shape)
    output_tensor = forward_step(data_iter, model, input_tensor, eval_features)
    send_forward(output_tensor)
    return output_tensor
  
def forward_step(data_iter, model, input_tensor, eval_features):
    # all need to get the data
    data = next(data_iter)
    batch = [t.cuda() for t in data]
    input_ids, input_mask, segment_ids, example_indices = batch

    if is_pipeline_first_stage():
        assert input_tensor is None
        output_tensor = model(input_ids, segment_ids, input_mask)
    elif is_pipeline_last_stage():
        assert input_tensor is not None
        all_results = []
        batch_start_logits, batch_end_logits = model(input_tensor, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))
        
        return all_results
    
    else:
        assert input_tensor is not None
        output_tensor = model(input_tensor, segment_ids, input_mask)
        
    return output_tensor