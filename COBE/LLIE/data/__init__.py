"""create dataset and dataloader"""
import logging
import paddle
import paddle.io

def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = paddle.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return paddle.io.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=num_workers, sampler=sampler, drop_last=True,
                                    use_shared_memory=False)
    else:
        return paddle.io.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                    use_shared_memory=False)


def create_dataset(dataset_opt):

    from COBE.LLIE.data.LL_dataset import ll_dataset as D
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
