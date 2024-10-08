import torch

from os import path as osp
from tqdm import tqdm

from basicsr.dataloaders import CPUPrefetcher, CUDAPrefetcher
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.models.metrics import calculate_metric


def dist_validation(model, opt, dataloader, current_iter, tb_logger, save_img):
    if opt['rank'] == 0:
        nondist_validation(model, opt, dataloader, current_iter, tb_logger, save_img)


def nondist_validation(model, opt, dataloader, current_iter, tb_logger, save_img):
    # dataloader prefetcher
    if torch.cuda.is_available():
        prefetcher = CUDAPrefetcher(dataloader)
    else:
        prefetcher = CPUPrefetcher(dataloader)

    dataset_name = opt['val'].get('name', '')
    with_metrics = opt['val'].get('metrics') is not None
    use_pbar = opt['val'].get('pbar', False)
    metric_results = {metric: 0 for metric in opt['val']['metrics'].keys()}
    pbar = tqdm(total=len(dataloader), unit='image')

    if with_metrics:
        # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        best_metric_results = dict()
        record = dict()
        for metric, content in opt['val']['metrics'].items():
            better = content.get('better', 'higher')
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[metric] = dict(better=better, val=init_val, iter=-1)
        best_metric_results[dataset_name] = record

    # zero metric_results
    if with_metrics:
        metric_results = {metric: 0 for metric in metric_results}

    metric_data = dict()
    prefetcher.reset()
    val_data = prefetcher.next()
    while val_data is not None:
        img_name = osp.splitext(osp.basename(val_data['name'][0]))[0]
        model.feed_data(val_data)
        model.test()
        val_data = prefetcher.next()

        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        metric_data['img'] = sr_img
        if 'gt' in visuals:
            gt_img = tensor2img([visuals['gt']])
            metric_data['img2'] = gt_img
            del model.gt

        # tentative for out of GPU memory
        del model.lq
        del model.output
        torch.cuda.empty_cache()

        if save_img:
            if opt['is_train']:
                save_img_path = osp.join(opt['model']['path']['visualization'], img_name, f'{img_name}_{"%08d"% current_iter}.png')
            else:
                if opt['val']['suffix']:
                    save_img_path = osp.join(opt['model']['path']['visualization'], dataset_name, f'{img_name}_{opt["val"]["suffix"]}.png')
                else:
                    save_img_path = osp.join(opt['model']['path']['visualization'], dataset_name, f'{img_name}_{opt["name"]}.png')

            if current_iter == 0 and 'gt' in visuals:
                imwrite(gt_img, save_img_path)
            else:
                imwrite(sr_img, save_img_path)

        if with_metrics:
            # calculate metrics
            for name, opt_ in opt['val']['metrics'].items():
                metric_results[name] += calculate_metric(metric_data, opt_)
        if use_pbar:
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

    if use_pbar:
        pbar.close()

    if with_metrics:
        for metric in metric_results.keys():
            metric_results[metric] /= (len(dataloader) + 1)
            # update the best metric result
            if best_metric_results[dataset_name][metric]['better'] == 'higher':
                if metric_results[metric] >= best_metric_results[dataset_name][metric]['val']:
                    best_metric_results[dataset_name][metric]['val'] = metric_results[metric]
                    best_metric_results[dataset_name][metric]['iter'] = current_iter
            else:
                if metric_results[metric] <= best_metric_results[dataset_name][metric]['val']:
                    best_metric_results[dataset_name][metric]['val'] = metric_results[metric]
                    best_metric_results[dataset_name][metric]['iter'] = current_iter

        log_str = f'Validation {dataset_name}\n'
        for metric, value in metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            log_str += (f'\tBest: {best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                        
                        f'{best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)


def validation(model, opt, dataloader, current_iter, tb_logger, save_img=False):
    """Validation function.

    Args:
        opt: complete opt
        model: model object
        dataloader (torch.utils.data.DataLoader): Validation dataloader.
        current_iter (int): Current iteration.
        tb_logger (tensorboard logger): Tensorboard logger.
        save_img (bool): Whether to save images. Default: False.
    """
    if opt['dist']:
        dist_validation(model, opt, dataloader, current_iter, tb_logger, save_img)
    else:
        nondist_validation(model, opt, dataloader, current_iter, tb_logger, save_img)
