import argparse
import os
import json
import time

from text2sql import _test
from utils.spider_metric.optimized_evaluator import OptimizedEvaluateTool

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for selecting the best ckpt (optimized version).")
    
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'input batch size.')
    parser.add_argument('--device', type = str, default = "2",
                        help = 'the id of used GPU device.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    parser.add_argument('--save_path', type = str, default = "./models/text2sql",
                        help = 'save path of fine-tuned text2sql models.')
    parser.add_argument('--eval_results_path', type = str, default = "./eval_results/text2sql",
                        help = 'the evaluation results of fine-tuned text2sql models.')
    parser.add_argument('--mode', type = str, default = "eval",
                        help='eval.')
    parser.add_argument('--dev_filepath', type = str, default = "./data/preprocessed_data/resdsql_test_natsql.json",
                        help = 'file path of test2sql dev set (should match target_type: resdsql_test_natsql.json for natsql, resdsql_test.json for sql).')
    parser.add_argument('--original_dev_filepath', type = str, default = "./data/spider/dev.json",
                        help = 'file path of the original dev set (for registing evaluator).')
    parser.add_argument('--db_path', type = str, default = "./database",
                        help = 'file path of database.')
    parser.add_argument('--tables_for_natsql', type = str, default = "./data/preprocessed_data/test_tables_for_natsql.json",
                        help = 'file path of tables_for_natsql.json (should use test_tables_for_natsql.json for dev/test evaluation).')
    parser.add_argument('--num_beams', type = int, default = 8,
                        help = 'beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type = int, default = 8,
                        help = 'the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    parser.add_argument("--target_type", type = str, default = "sql",
                help = "sql or natsql.")
    parser.add_argument("--output", type = str, default = "predicted_sql.txt")
    
    # Optimization arguments
    parser.add_argument('--num_workers', type = int, default = 4,
                        help = 'number of parallel workers for evaluation (default: 4). Set to 1 to disable parallel processing.')
    parser.add_argument('--use_parallel', action = 'store_true', default = True,
                        help = 'use parallel evaluation (default: True). Use --no-use_parallel to disable.')
    parser.add_argument('--no-use_parallel', dest='use_parallel', action='store_false',
                        help = 'disable parallel evaluation')
    
    opt = parser.parse_args()
    return opt


def _test_optimized(opt):
    """Optimized version of _test that uses OptimizedEvaluateTool"""
    import torch
    from transformers import T5TokenizerFast, T5ForConditionalGeneration, MT5ForConditionalGeneration, AddedToken
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from text2sql import set_seed
    from utils.text2sql_decoding_utils import decode_sqls, decode_natsqls
    from utils.load_dataset import Text2SQLDataset
    
    set_seed(opt.seed)
    print(opt)
    print(f"Using optimized evaluator with {opt.num_workers} workers (parallel: {opt.use_parallel})")
    print(f"Target type: {opt.target_type}")
    print(f"Dev filepath: {opt.dev_filepath}")
    print(f"Database path: {opt.db_path}")

    start_time = time.time()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    
    # Verify dev_filepath exists
    if not os.path.exists(opt.dev_filepath):
        raise FileNotFoundError(
            f"Dev file not found at {opt.dev_filepath}. "
            f"Please ensure the file exists or specify the correct path with --dev_filepath"
        )
    
    # Initialize table_dict for NatSQL mode (similar to API server fix)
    table_dict = None
    if opt.target_type == "natsql":
        if not os.path.exists(opt.tables_for_natsql):
            raise FileNotFoundError(
                f"NatSQL tables file not found at {opt.tables_for_natsql}. "
                f"Please ensure the file exists or specify the correct path with --tables_for_natsql"
            )
        tables = json.load(open(opt.tables_for_natsql, 'r', encoding='utf-8'))
        table_dict = dict()
        for t in tables:
            table_dict[t["db_id"]] = t
        print(f"Loaded {len(table_dict)} databases for NatSQL evaluation")

    # initialize tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(
        opt.save_path,
        add_prefix_space = True
    )
    
    if isinstance(tokenizer, T5TokenizerFast):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    
    dev_dataset = Text2SQLDataset(
        dir_ = opt.dev_filepath,
        mode = opt.mode
    )

    dev_dataloder = DataLoader(
        dev_dataset, 
        batch_size = opt.batch_size, 
        shuffle = False,
        collate_fn = lambda x: x,
        drop_last = False
    )

    model_class = MT5ForConditionalGeneration if "mt5" in opt.save_path else T5ForConditionalGeneration

    # initialize model
    model = model_class.from_pretrained(opt.save_path)
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    predict_sqls = []
    for batch in tqdm(dev_dataloder, desc="Generating SQL"):
        batch_inputs = [data[0] for data in batch]
        batch_db_ids = [data[1] for data in batch]
        batch_tc_original = [data[2] for data in batch]

        tokenized_inputs = tokenizer(
            batch_inputs, 
            return_tensors="pt",
            padding = "max_length",
            max_length = 512,
            truncation = True
        )
        
        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]
        if torch.cuda.is_available():
            encoder_input_ids = encoder_input_ids.cuda()
            encoder_input_attention_mask = encoder_input_attention_mask.cuda()

        with torch.no_grad():
            model_outputs = model.generate(
                input_ids = encoder_input_ids,
                attention_mask = encoder_input_attention_mask,
                max_length = 256,
                decoder_start_token_id = model.config.decoder_start_token_id,
                num_beams = opt.num_beams,
                num_return_sequences = opt.num_return_sequences
            )

            model_outputs = model_outputs.view(len(batch_inputs), opt.num_return_sequences, model_outputs.shape[1])
            if opt.target_type == "sql":
                predict_sqls += decode_sqls(
                    opt.db_path, 
                    model_outputs, 
                    batch_db_ids, 
                    batch_inputs, 
                    tokenizer, 
                    batch_tc_original
                )
            elif opt.target_type == "natsql":
                if table_dict is None or len(table_dict) == 0:
                    raise ValueError(
                        f"table_dict is not initialized for NatSQL mode. "
                        f"Please ensure --tables_for_natsql points to a valid file."
                    )
                predict_sqls += decode_natsqls(
                    opt.db_path, 
                    model_outputs, 
                    batch_db_ids, 
                    batch_inputs, 
                    tokenizer, 
                    batch_tc_original, 
                    table_dict
                )
            else:
                raise ValueError(f"Invalid target_type: {opt.target_type}. Must be 'sql' or 'natsql'")
    
    new_dir = "/".join(opt.output.split("/")[:-1]).strip()
    if new_dir != "":
        os.makedirs(new_dir, exist_ok = True)
    
    # save results
    with open(opt.output, "w", encoding = 'utf-8') as f:
        for pred in predict_sqls:
            f.write(pred + "\n")
    
    end_time = time.time()
    print("Text-to-SQL inference spends {:.2f}s.".format(end_time-start_time))
    
    if opt.mode == "eval":
        # Use optimized evaluator
        print(f"\nStarting evaluation with {opt.num_workers} workers...")
        evaluator = OptimizedEvaluateTool(
            num_workers=opt.num_workers,
            use_parallel=opt.use_parallel
        )
        evaluator.register_golds(opt.original_dev_filepath, opt.db_path)
        
        eval_start = time.time()
        spider_metric_result = evaluator.evaluate(predict_sqls)
        eval_end = time.time()
        
        print('exact_match score: {}'.format(spider_metric_result["exact_match"]))
        print('exec score: {}'.format(spider_metric_result["exec"]))
        print(f'Evaluation time: {eval_end - eval_start:.2f}s (optimized)')
    
        return spider_metric_result["exact_match"], spider_metric_result["exec"]
    
    return None, None

    
if __name__ == "__main__":
    opt = parse_option()
    
    ckpt_names = os.listdir(opt.save_path)
    ckpt_names = sorted(ckpt_names, key = lambda x:eval(x.split("-")[1]))
    
    print("ckpt_names:", ckpt_names)

    save_path = opt.save_path
    os.makedirs(opt.eval_results_path, exist_ok = True)

    eval_results = []
    for ckpt_name in ckpt_names:
        print("\n" + "="*80)
        print("Start evaluating ckpt: {}".format(ckpt_name))
        print("="*80)
        
        opt.save_path = save_path + "/{}".format(ckpt_name)
        em, exec = _test_optimized(opt)
        
        eval_result = dict()
        eval_result["ckpt"] = opt.save_path
        eval_result["EM"] = em
        eval_result["EXEC"] = exec

        with open(opt.eval_results_path+"/{}.txt".format(ckpt_name), "w") as f:
            f.write(json.dumps(eval_result, indent = 2, ensure_ascii = False))
        
        eval_results.append(eval_result)
    
    print("\n" + "="*80)
    print("Evaluation Summary")
    print("="*80)
    for eval_result in eval_results:
        print("ckpt name:", eval_result["ckpt"])
        print("EM:", eval_result["EM"])
        print("EXEC:", eval_result["EXEC"])
        print("-----------")

    em_list = [er["EM"] for er in eval_results]
    exec_list = [er["EXEC"] for er in eval_results]
    em_and_exec_list = [em + exec for em, exec in zip(em_list, exec_list)]

    # find best EM ckpt
    best_em, exec_in_best_em = 0.00, 0.00
    best_em_idx = 0

    # find best EXEC ckpt
    best_exec, em_in_best_exec = 0.00, 0.00
    best_exec_idx = 0

    # find best EM + EXEC ckpt
    best_em_plus_exec = 0.00
    best_em_plus_exec_idx = 0

    for idx, (em, exec) in enumerate(zip(em_list, exec_list)):
        if em > best_em or (em == best_em and exec > exec_in_best_em):
            best_em = em
            exec_in_best_em = exec
            best_em_idx = idx
        
        if exec > best_exec or (exec == best_exec and em > em_in_best_exec):
            best_exec = exec
            em_in_best_exec = em
            best_exec_idx = idx
        
        if em+exec > best_em_plus_exec:
            best_em_plus_exec = em+exec
            best_em_plus_exec_idx = idx
    
    print("\n" + "="*80)
    print("Best Checkpoints")
    print("="*80)
    print("Best EM ckpt:", eval_results[best_em_idx])
    print("Best EXEC ckpt:", eval_results[best_exec_idx])
    print("Best EM+EXEC ckpt:", eval_results[best_em_plus_exec_idx])
