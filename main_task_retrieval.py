from __future__ import absolute_import, division, unicode_literals, print_function

import os
import numpy as np
import random
import time
import argparse
import gc
import torch
import torch.distributed as dist

from metrics import compute_metrics_rect
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam

from util import get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT
from shutil import copyfile

from dataloaders.dataloader_cova_retrieval import get_cova_gallery_loader

torch.distributed.init_process_group(backend="nccl")
global logger


def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--debug_mode", action='store_true')

    parser.add_argument('--train_csv', type=str, default='data/.train.csv')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle')
    parser.add_argument('--audio_path', type=str, default='data/videos/audios_all')

    parser.add_argument('--num_thread_reader', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_size_val', type=int, default=16)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--n_display', type=int, default=100)
    parser.add_argument('--video_dim', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_words', type=int, default=32)
    parser.add_argument('--max_frames', type=int, default=12)
    parser.add_argument('--eval_max_frames', type=int, default=12)
    parser.add_argument('--feature_framerate', type=int, default=1)
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--hard_negative_rate', type=float, default=0.5)
    parser.add_argument('--negative_weighting', type=int, default=1)
    parser.add_argument('--n_pair', type=int, default=1)

    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--cross_model", default="cross-base", type=str)
    parser.add_argument("--init_model", default=None, type=str)
    parser.add_argument("--resume_model", default=None, type=str)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--n_gpu', type=int, default=1)

    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')

    parser.add_argument("--task_type", default="retrieval", type=str)
    parser.add_argument("--datatype", default="cova", type=str)

    parser.add_argument("--world_size", default=0, type=int)
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", default=0, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument('--coef_lr', type=float, default=1.)
    parser.add_argument('--use_mil', action='store_true')
    parser.add_argument('--sampled_use_mil', action='store_true')

    parser.add_argument('--text_num_hidden_layers', type=int, default=12)
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12)
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4)
    parser.add_argument('--audio_query_layers', type=int, default=4)

    parser.add_argument('--loose_type', action='store_true')
    parser.add_argument('--expand_msrvtt_sentences', action='store_true')

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2])

    parser.add_argument('--freeze_layer_num', type=int, default=12)
    parser.add_argument('--slice_framepos', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"])
    parser.add_argument('--sim_header', type=str, default="seqTransf",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"])
    parser.add_argument("--temperature", default=32, type=float)

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str)

    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--margin_BD', type=float, default=0.1)

    parser.add_argument("--train_json", type=str, default=None)
    parser.add_argument("--test_json",  type=str, default=None)
    parser.add_argument("--gallery_json", type=str, default=None)

    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps: {}".format(args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)
    return args


def set_seed_logger(args):
    global logger
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, "log.txt"))
    return args


def init_device(args, local_rank):
    global logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
    n_gpu = torch.cuda.device_count()
    args.n_gpu = n_gpu

    if n_gpu > 0:
        if args.batch_size % n_gpu != 0 or args.batch_size_val % n_gpu != 0:
            raise ValueError("Invalid batch_size/batch_size_val and n_gpu: {}%{} and {}%{}".format(
                args.batch_size, n_gpu, args.batch_size_val, n_gpu))
    return device, n_gpu


def init_model(args, device, n_gpu, local_rank):
    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    if hasattr(model, 'clip') and hasattr(model.clip, 'logit_scale'):
        model.clip.logit_scale.data = torch.log(torch.tensor(1.0 / 0.07))

    model.to(device)
    return model


def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):
    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_param_tp], 'weight_decay': weight_decay, 'lr': args.lr},
        {'params': [p for n, p in no_decay_param_tp], 'weight_decay': 0.0, 'lr': args.lr}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
    )
    return optimizer, scheduler, model


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    total_loss = 0
    nan_count = 0
    valid_steps = 0

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch[:9]) + tuple(batch[9:])

        (q_input_ids, q_input_mask, q_segment_ids,
         q_video, q_video_mask, q_audio,
         t_video, t_video_mask, t_audio, *_) = batch

        if (torch.isnan(q_video).any() or torch.isnan(t_video).any() or
            torch.isnan(q_audio).any() or torch.isnan(t_audio).any()):
            nan_count += 1
            continue

        try:
            loss = model(q_input_ids, q_segment_ids, q_input_mask,
                         q_video, q_video_mask, q_audio,
                         t_video, t_video_mask, t_audio)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                optimizer.zero_grad()
                continue

            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            total_loss += float(loss)
            valid_steps += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scheduler is not None:
                    scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

                if hasattr(model, 'module') and hasattr(model.module, 'clip'):
                    torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
                elif hasattr(model, 'clip'):
                    torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

                global_step += 1
                if global_step % log_step == 0 and local_rank == 0:
                    avg_loss = total_loss / valid_steps if valid_steps > 0 else 0.0
                    logger.info("Epoch: %d/%s, Step: %d/%d, Loss: %.4f, NaN_count: %d",
                                epoch + 1, args.epochs, step + 1, len(train_dataloader),
                                avg_loss, nan_count)

        except Exception as e:
            logger.error(f"Step {step}: Error during forward/backward: {e}")
            optimizer.zero_grad()
            continue

    avg_loss = total_loss / valid_steps if valid_steps > 0 else 0.0
    logger.info(f"Epoch {epoch+1} finished. Average loss: {avg_loss:.4f}, NaN batches: {nan_count}/{len(train_dataloader)}")
    return avg_loss, global_step


@torch.no_grad()
def eval_epoch(args, model, test_dataloader, device, n_gpu, gallery_dataloader=None):
    if hasattr(model, 'module'):
        model = model.module
    model = model.to(device)
    model.eval()

    Q_list, Ttest_list = [], []
    valid_batches = 0

    for bid, batch in enumerate(test_dataloader):
        tensors = tuple(t.to(device) for t in batch[:9])
        (q_input_ids, q_input_mask, q_segment_ids,
         q_video, q_video_mask, q_audio,
         t_video, t_video_mask, t_audio) = tensors

        if (torch.isnan(q_video).any() or torch.isnan(t_video).any() or
            torch.isnan(q_audio).any() or torch.isnan(t_audio).any()):
            continue

        try:
            q_global = model.encode_query(
                q_input_ids, q_segment_ids, q_input_mask,
                q_video, q_video_mask, q_audio
            )
            t_global = model.encode_target(
                t_video, t_video_mask, t_audio
            )
            if torch.isnan(q_global).any() or torch.isnan(t_global).any():
                continue

            Q_list.append(q_global.detach().cpu())
            Ttest_list.append(t_global.detach().cpu())
            valid_batches += 1
        except Exception as e:
            logger.error(f"Eval batch {bid}: Error during encoding: {e}")
            continue

        if args.local_rank == 0:
            print("{}/{}\r".format(bid + 1, len(test_dataloader)), end="")

    logger.info(f"\nValid test batches: {valid_batches}/{len(test_dataloader)}")
    if valid_batches == 0:
        logger.error("No valid batches in evaluation!")
        return 0.0

    Q = torch.cat(Q_list, dim=0)
    T_test = torch.cat(Ttest_list, dim=0)

    T_all = T_test
    if gallery_dataloader is not None:
        Tg_list = []
        for gbid, gbatch in enumerate(gallery_dataloader):
            gt_video = gbatch[0].to(device)
            gt_video_mask = gbatch[1].to(device)
            gt_audio = gbatch[2].to(device)
            try:
                tg = model.encode_target(gt_video, gt_video_mask, gt_audio)
                if torch.isnan(tg).any() or torch.isinf(tg).any():
                    continue
                Tg_list.append(tg.detach().cpu())
            except Exception as e:
                logger.error(f"[GALLERY] b{gbid} encode error: {e}")
                continue

        if len(Tg_list) > 0:
            T_gallery = torch.cat(Tg_list, dim=0)
            T_all = torch.cat([T_test, T_gallery], dim=0)

    try:
        logit_scale = model.clip.logit_scale.exp().detach().cpu().item()
        logit_scale = max(0.01, min(100.0, logit_scale))
    except Exception:
        logit_scale = 1.0

    sim_matrix = (Q @ T_all.t()) * logit_scale
    sim_np = sim_matrix.detach().cpu().numpy()
    Nq = Q.size(0)
    Nt = T_test.size(0)
    metric_rows = min(Nq, Nt)
    sim_eval = sim_np[:metric_rows, :]
    gt_cols = np.arange(metric_rows, dtype=np.int64)

    tv_metrics = compute_metrics_rect(sim_eval, gt_cols=gt_cols, ks=(1, 5, 10))
    logger.info(">>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".format(
        tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']
    ))

    return tv_metrics['R1']


def save_model(epoch, args, model, optimizer, tr_loss, global_step=0):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(args.output_dir, f"pytorch_model.bin.{epoch}")
    torch.save(model_to_save.state_dict(), output_model_file)

    optimizer_file = os.path.join(args.output_dir, f"optimizer.bin.{epoch}")
    torch.save({'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(), 'loss': tr_loss}, optimizer_file)
    return output_model_file


def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    if args.do_train:
        copyfile('./main_task_retrieval.py', os.path.join(args.output_dir, 'main_task_retrieval.py'))
        copyfile('./modules/modeling.py', os.path.join(args.output_dir, 'modeling.py'))
        copyfile('./dataloaders/data_dataloaders.py', os.path.join(args.output_dir, 'data_dataloaders.py'))
        copyfile('./dataloaders/dataloader_cova_retrieval.py', os.path.join(args.output_dir, 'dataloader_cova.py'))
        copyfile('./run.sh', os.path.join(args.output_dir, 'run.sh'))

    tokenizer = ClipTokenizer()
    model = init_model(args, device, n_gpu, args.local_rank)

    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0 or name.find("audio_projection") == 0 \
                    or name.find("audio.v.norm.") == 0:
                continue

            if name.find("audio.v.blocks.") == 0:
                layer_num = int(name.split(".blocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue

            if args.linear_patch == "3d" and name.find("conv2."):
                continue

            param.requires_grad_(False)

    for name, param in model.named_parameters():
        if any(keyword in name for keyword in [
            "audio.v.head",
            "disent_gate_simple",
            "transformer_Fusion",
            "weight_predictor",
            "query_tokens",
            "frame_position_embeddings",
            "query_position_embeddings",
            "audio_projection",
            "transformer_Five",
            "five_gate", "five_mix_gate",
            "five_sa_extra", "transformer_FiveSA",
            "five_pair_gate", "five_pairattn_extra", "transformer_FivePairAttn"
        ]):
            param.requires_grad_(True)

    test_dataloader, test_length  = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)
    if args.local_rank == 0:
        logger.info(f"Test loaded: {test_length}")

    gallery_dataloader = None
    if getattr(args, "gallery_json", None):
        try:
            gallery_dataloader, gallery_length = get_cova_gallery_loader(args, tokenizer)
            if args.local_rank == 0:
                logger.info(f"[GALLERY] loaded: {gallery_length} videos")
        except Exception as e:
            logger.error(f"Failed to build gallery loader: {e}")
            gallery_dataloader = None

    try:
        if args.do_train:
            train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
            if args.local_rank == 0:
                logger.info(f"Train loaded: {train_length}")
            num_train_optimization_steps = (
                int(len(train_dataloader) + args.gradient_accumulation_steps - 1) /
                args.gradient_accumulation_steps
            ) * args.epochs

            optimizer, scheduler, model = prep_optimizer(
                args, model, num_train_optimization_steps, device, n_gpu, args.local_rank
            )

            best_score = 0.00001
            best_output_model_file = "None"

            global_step = 0
            for epoch in range(args.epochs):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                tr_loss, global_step = train_epoch(
                    epoch, args, model, train_dataloader, device, n_gpu,
                    optimizer, scheduler, global_step, local_rank=args.local_rank
                )

                if args.local_rank == 0:
                    output_model_file = save_model(epoch, args, model, optimizer, tr_loss)
                    R1 = eval_epoch(args, model, test_dataloader, device, n_gpu, gallery_dataloader=gallery_dataloader)

                    if best_score <= R1:
                        best_score = R1
                        best_output_model_file = output_model_file

                    logger.info(f"Best model: {best_output_model_file}, R@1: {best_score:.4f}")

        elif args.do_eval:
            if args.local_rank == 0:
                eval_epoch(args, model, test_dataloader, device, n_gpu, gallery_dataloader=gallery_dataloader)
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
