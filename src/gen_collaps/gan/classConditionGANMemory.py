# [norm]_Flower_AdaFM_bias_classCondition_FinalVersion
import argparse
import glob
import os, sys
from os import path
import time
import copy
from pathlib import Path
from torchvision import transforms
import torch
from torch import nn
import numpy as np
import random
from diffusers.utils import make_image_grid
from torchvision.io import read_image
from tqdm.auto import trange, tqdm
from datasets import load_dataset
from torchvision.utils import save_image


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(999)

# from torchsummary import summary
import shutil
import scipy.io as sio
from gan_training import utils
from gan_training.utils_model_load import *
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    load_config,
    build_models,
    build_optimizers,
    build_lr_scheduler,
)

''' ===================--- Set the traning mode ---==========================
DATA: going to train
DATA_FIX: used as a fixed pre-trained model
============================================================================='''
seed_torch(999)
DATA_FIX = 'CELEBA'
Num_epoch = 50_000 // 448
# select the name of the task from ['fish', 'bird', 'snake', 'dog', 'butterfly', 'insect']

NNN = 7200
image_path = './data/102flowers/'
main_path = '.'
out_path = '/scratch/shared/beegfs/dzverev/gen_collaps/gan'

config_path = main_path + '/configs/' + 'classcondition' + '_celeba.yaml'
config = load_config(config_path, 'configs/default.yaml')
config['data']['train_dir'] = image_path
config['training']['out_dir'] = out_path

if not os.path.isdir(config['training']['out_dir']):
    os.makedirs(config['training']['out_dir'])

config['synth_dataset_num_images'] = 8_000
config['synth_dataset_batch_size'] = 128


def train(train_dataset, step_id, nlabels=102):
    config['training']['out_dir'] = out_path + f'/step_{step_id}'
    if not os.path.isdir(config['training']['out_dir']):
        os.makedirs(config['training']['out_dir'])

    if 1:
        # Short hands
        batch_size = config['training']['batch_size']
        d_steps = config['training']['d_steps']
        restart_every = config['training']['restart_every']
        inception_every = config['training']['inception_every']
        save_every = config['training']['save_every']
        backup_every = config['training']['backup_every']
        sample_nlabels = config['training']['sample_nlabels']
        dim_z = config['z_dist']['dim']

        out_dir = config['training']['out_dir']
        checkpoint_dir = path.join(out_dir, 'chkpts')

        # Create missing directories
        if not path.exists(out_dir):
            os.makedirs(out_dir)
        if not path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')

        # Logger
        checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=config['training']['nworkers'],
            shuffle=True,
            pin_memory=True,
            sampler=None,
            drop_last=True,
        )
        # test_dataset, _ = get_dataset(
        #     name=config['data']['type'],
        #     data_dir=config['data']['test_dir'],
        #     size=config['data']['img_size'],
        #     lsun_categories=config['data']['lsun_categories_train']
        # )
        # test_loader = torch.utils.data.DataLoader(
        #     test_dataset,
        #     batch_size=batch_size,
        #     num_workers=config['training']['nworkers'],
        #     shuffle=True, pin_memory=True, sampler=None, drop_last=True
        # )

        # Number of labels
        # print('nlabels=======================', nlabels)
        nlabels = min(nlabels, config['data']['nlabels'])
        sample_nlabels = min(nlabels, sample_nlabels)

        # Create models
        ''' --------- Choose the fixed layer ---------------'''
        generator, discriminator = build_models(config)

        generator = load_model_norm(generator)
        discriminator = load_model_norm(discriminator, is_G=False)

        for name, param in generator.named_parameters():
            if name.find('AdaFM_') >= 0:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, param in discriminator.named_parameters():
            if name.find('AdaFM_') >= 0:
                param.requires_grad = True
            elif name.find('fc') >= 0:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # # # =========================================================================================
        # # # --------------- test, generate samples for each class with label as input ---------------
        # # # =========================================================================================
        # model_file = 'F:/RUN_CODE_OUT/OWM/imagenet_'+DATA+'_AdaFM_bias_[norm]/G_8_D_2/' + 'models/'
        # dict_G = torch.load(model_file + DATA + '_%08d_Pre_generator' % 59999)
        # generator = model_equal_all(generator, dict_G)
        # # # =========================================================================================
        # # # =========================================================================================

        # Put models on gpu if needed
        generator, discriminator = generator.to(device), discriminator.to(device)
        g_optimizer, d_optimizer = build_optimizers(generator, discriminator, config)

        # summary(generator, input_size=[(256,), (1,)])
        # summary(discriminator, input_size=[(3, 128, 128), (1,)])

        # Register modules to checkpoint
        checkpoint_io.register_modules(
            generator=generator,
            discriminator=discriminator,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
        )

        # Logger
        logger = Logger(
            log_dir=path.join(out_dir, 'logs'),
            img_dir=path.join(out_dir, 'imgs'),
            monitoring=config['training']['monitoring'],
            monitoring_dir=path.join(out_dir, 'monitoring'),
        )

        # Distributions
        ydist = get_ydist(nlabels, device=device)
        zdist = get_zdist(
            config['z_dist']['type'], config['z_dist']['dim'], device=device
        )

        # Save for tests
        ntest = 20
        x_real, ytest = utils.get_nsamples(train_loader, ntest)
        ytest.clamp_(None, nlabels - 1)
        ytest = ytest.to(device)
        ztest = zdist.sample((ntest,)).to(device)
        utils.save_images(x_real, path.join(out_dir, 'real.png'))

        # Test generator
        if config['training']['take_model_average']:
            generator_test = copy.deepcopy(generator)
            checkpoint_io.register_modules(generator_test=generator_test)
        else:
            generator_test = generator

        # Evaluator
        evaluator = Evaluator(
            generator_test, zdist, ydist, batch_size=batch_size, device=device
        )
        # x_real, _ = utils.get_nsamples(train_loader, NNN)
        # evaluator = Evaluator(generator_test, zdist, ydist,
        #                       batch_size=batch_size, device=device,
        #                       fid_real_samples=x_real, inception_nsamples=NNN, fid_sample_size=NNN)

        # Train
        it = -1
        epoch_idx = -1
        # Reinitialize model average if needed
        if (
            config['training']['take_model_average']
            and config['training']['model_average_reinit']
        ):
            update_average(generator_test, generator, 0.0)
        # Learning rate anneling
        g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
        d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)

        # Trainer
        trainer = Trainer(
            generator,
            discriminator,
            g_optimizer,
            d_optimizer,
            gan_type=config['training']['gan_type'],
            reg_type=config['training']['reg_type'],
            reg_param=config['training']['reg_param'],
        )

    # Training loop
    print('Start training...')
    save_dir = config['training']['out_dir'] + '/models/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    get_parameter_number(generator)
    get_parameter_number(discriminator)

    inception_mean_all = []
    inception_std_all = []
    fid_all = []

    tstart = time.time()

    for epoch_idx in trange(Num_epoch, desc="Epoch: "):

        print('Start epoch %d...' % epoch_idx)

        for batch_data in tqdm(train_loader, leave=False, desc="Batch: "):
            it += 1

            x_real = batch_data['image']
            y = batch_data['label']

            d_lr = d_optimizer.param_groups[0]['lr']
            g_lr = g_optimizer.param_groups[0]['lr']

            x_real, y = x_real.to(device), y.to(device)
            y.clamp_(None, nlabels - 1)

            # Generators updates
            z = zdist.sample((batch_size,))
            gloss, x_fake, _ = trainer.generator_trainstep(y, z)

            if config['training']['take_model_average']:
                update_average(
                    generator_test,
                    generator,
                    beta=config['training']['model_average_beta'],
                )

            # Discriminator updates
            dloss, reg = trainer.discriminator_trainstep(x_real, y, x_fake)

            # step
            d_scheduler.step()
            g_scheduler.step()

            with torch.no_grad():

                # (i) Sample if necessary
                if (it % config['training']['sample_every']) == 0:
                    d_fix, d_update = (
                        discriminator.conv_img.weight[1, 1, 1, 1],
                        discriminator.fc.weight[0, 1],
                    )
                    g_fix, g_update = generator.conv_img.weight[1, 1, 1, 1], 0.0

                    print(
                        '[epoch %0d, it %4d] g_loss = %.4f, d_loss = %.4f, reg=%.4f, time=%.2f'
                        % (epoch_idx, it, gloss, dloss, reg, time.time() - tstart)
                    )
                    tstart = time.time()
                    # print('Creating samples...')
                    x, _ = evaluator.create_samples(ztest, ytest)
                    logger.add_imgs(x, 'all', it, nrow=2)

                # # (ii) Compute inception if necessary
                # if inception_every > 0 and ((it + 2) % inception_every) == 0:
                #     inception_mean, inception_std, fid = evaluator.compute_inception_score()
                #     inception_mean_all.append(inception_mean)
                #     inception_std_all.append(inception_std)
                #     fid_all.append(fid)
                #     print('test it %d: IS: mean %.2f, std %.2f, FID: mean %.2f, time: %2f' % (
                #         it, inception_mean, inception_std, fid, time.time() - tstart))
                #
                #     FID = np.stack(fid_all)
                #     Inception_mean = np.stack(inception_mean_all)
                #     Inception_std = np.stack(inception_std_all)
                #     sio.savemat(out_path + DATA + 'base_FID_IS.mat', {'FID': FID,
                #                                            'Inception_mean': Inception_mean,
                #                                            'Inception_std': Inception_std})

                # (iii) Backup if necessary
                if ((it + 1) % backup_every) == 0:
                    print('Saving backup...')
                    TrainModeSave = str(step_id) + '_%08d_' % it
                    generator_test_part = save_adafm_only(generator_test)
                    torch.save(
                        generator_test_part, save_dir + TrainModeSave + 'Pre_generator'
                    )
                # if it + 1 == 60000:
                #     TrainModeSave = DATA + '_%08d_' % it
                #     discriminator_part = save_adafm_only(discriminator, is_G=False)
                #     torch.save(discriminator_part, save_dir + TrainModeSave + 'Pre_discriminator')

    sample_synthetic_dataset(config, device, evaluator, logger)


def load_synthetic_dataset(config):
    synthetic_images = []
    synthetic_labels = []

    synthetic_dataset_path = (
        Path(config['training']['out_dir']) / 'synth_dataset' / 'all'
    )

    # read dataset to memory
    sample_images = sorted(glob.glob(f"{synthetic_dataset_path}/*.png"))
    sample_labels = sorted(glob.glob(f"{synthetic_dataset_path}/*.pt"))

    for batched_images, batched_labels in zip(sample_images, sample_labels):
        batched_images = read_image(batched_images)

        synthetic_images.extend(
            [
                batched_images[:, :, i * 64 : (i + 1) * 64]
                for i in range(batched_images.shape[-1] // 64)
            ]
        )
        synthetic_labels.extend(torch.load(batched_labels).tolist())

    # transform dataset
    preprocess = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
        ]
    )

    dataset = [
        {'image': preprocess(image).float(), 'label': torch.tensor(label)}
        for image, label in zip(synthetic_images, synthetic_labels)
    ]

    return dataset


@torch.no_grad()
def sample_synthetic_dataset(config, device, evaluator, logger):
    synthetic_dataset_path = Path(config['training']['out_dir']) / 'synth_dataset'
    synthetic_dataset_path.mkdir(exist_ok=True, parents=True)

    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)

    for i in range(
        config['synth_dataset_num_images'] // config['synth_dataset_batch_size']
    ):
        ztest = zdist.sample((config['synth_dataset_batch_size'],)).to(device)

        x, y = evaluator.create_samples(ztest)
        logger.img_dir = str(synthetic_dataset_path)
        logger.add_imgs(x, 'all', 100_000 + i, nrow=config['synth_dataset_batch_size'])
        torch.save(y.cpu(), f'{str(synthetic_dataset_path)}/all/{100_000 + i}.pt')


def transform(examples):
    preprocess = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {'image': images, 'label': examples["label"]}


initial_dataset = load_dataset("nelorth/oxford-flowers", split="train")
initial_dataset.set_transform(transform)

STEP_ID = 0
train(initial_dataset, step_id=STEP_ID)

for _ in range(12):
    STEP_ID += 1

    synth_dataset = load_synthetic_dataset(config)
    train(synth_dataset, step_id=STEP_ID)
