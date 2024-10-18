# Run this file with argument "--num_clients 1"

import argparse
import collections
import copy
import os
from functools import partial

import numpy as np
import torch
from torchvision.utils import save_image

from utils import (
    epoch, get_daparam, get_dataset, get_eval_pool, get_loops,
    get_network, get_time, match_loss, ParamDiffAug
)
from client import Client
from attack import perform_attack
from utils import *
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from utils_attack import *


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument(
        '--dataset', type=str, default='CIFAR10', help='dataset'
    )
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument(
        '--ipc', type=int, default=10, help='image(s) per class'
    )
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode')
    parser.add_argument(
        '--num_exp', type=int, default=5, help='the number of experiments'
    )
    parser.add_argument(
        '--num_eval', type=int, default=5,
        help='the number of evaluating randomly initialized models'
    )
    parser.add_argument(
        '--epoch_eval_train', type=int, default=300,
        help='epochs to train a model with synthetic data'
    )
    parser.add_argument(
        '--Iteration', type=int, default=1000, help='training iterations'
    )
    parser.add_argument(
        '--lr_img', type=float, default=0.1,
        help='learning rate for updating synthetic images'
    )
    parser.add_argument(
        '--lr_net', type=float, default=0.01,
        help='learning rate for updating network parameters'
    )
    parser.add_argument(
        '--batch_real', type=int, default=64, help='batch size for real data'
    )
    parser.add_argument(
        '--batch_train', type=int, default=64,
        help='batch size for training networks'
    )
    parser.add_argument(
        '--init', type=str, default='noise',
        help='initialize synthetic images from noise or real data'
    )
    parser.add_argument(
        '--dsa_strategy', type=str, default='None',
        help='differentiable Siamese augmentation strategy'
    )
    parser.add_argument(
        '--data_path', type=str, default='data', help='dataset path'
    )
    parser.add_argument(
        '--save_path', type=str, default='result', help='path to save results'
    )
    parser.add_argument(
        '--dis_metric', type=str, default='ours', help='distance metric'
    )

    parser.add_argument(
        '--eval_frequency', type=int, default=500,
        help='evaluation frequency during synthetic images training'
    )
    parser.add_argument(
        '--device', type=str,
        default=('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='hardware device to use'
    )
    parser.add_argument(
        '--num_clients', type=int, default=3,
        help='number of clients for federated learning'
    )
    parser.add_argument(
        '--net_act', type=str, default='relu',
        help='activation function inside networks'
    )
    parser.add_argument(
        '--net_norm', type=str, default='instancenorm',
        help='normalization inside networks'
    )
    parser.add_argument(
        '--net_pooling', type=str, default='avgpooling',
        help='pooling inside networks'
    )
    parser.add_argument(
        '--model_sharing', default=False,
        action=argparse.BooleanOptionalAction, help=(
            'initialize the attacker\'s model with the weights of the '
            'real client\'s model'
        )
    )
    parser.add_argument(
        '--delta', type=float, default=None,
        help='delta hyperparameter for gradient clipping'
    )
    parser.add_argument(
        '--lambda_', type=float, default=None, help=(
            'lambda hyperparameter for laplacian noise added to clipped '
            'gradients'
        )
    )
    parser.add_argument(
        '--batch_real_shrinkage', default=True,
        action=argparse.BooleanOptionalAction, help=(
            'allow shrinkage of args.batch_real if clients do not have enough '
            'images to use its initial value'
        )
    )
    parser.add_argument(
        '--soft_labeling', default=False,
        action=argparse.BooleanOptionalAction,
        help='apply soft labeling during dataset distillation'
    )
    parser.add_argument(
        '--num_attack_iterations', type=int, default=None, help=(
            'number of attack iterations, or None for dataset distillation '
            'only'
        )
    )

    # Arguments for the DOORPING attack
    parser.add_argument('--doorping', action='store_true', help='Enable DOORPING backdoor attack')
    parser.add_argument('--portion', type=float, default=0.4, help='Portion of data to be backdoored')
    parser.add_argument('--backdoor_size', type=int, default=2, help='Size of the backdoor trigger')
    parser.add_argument('--trigger_label', type=int, default=0, help='Label to assign to backdoored images')
    parser.add_argument('--ori', type=float, default=1.0, help='Portion of the dataset to be used for training')
    parser.add_argument('--layer', type=int, default=-1, help='Layer to use for the doorping attack')
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--trigger_update_strategy', type=str, choices=['individual', 'average'], default='individual', help='Strategy for updating the trigger: "individual" updates the trigger using each client’s model, "average" uses the averaged model of all attacking clients.')
    parser.add_argument('--gradient_leakage', default=False) 
    
    return parser.parse_args()

def print_arguments(args):
    arg_descriptions = {
        'Iteration': 'The total number of training iterations',
        'batch_real': 'Batch size for processing real data',
        'batch_real_shrinkage': (
            'Allow shrinkage of args.batch_real if clients do not have enough '
            'images to use its initial value'
        ),
        'batch_train': 'Batch size for training models',
        'data_path': 'Path to the dataset',
        'dataset': 'The name of the dataset (e.g., CIFAR10)',
        'delta': 'Delta hyperparameter for gradient clipping',
        'device': 'The hardware device to use (\'cpu\', \'cuda:0\', ...)',
        'dis_metric': 'The distance metric used for evaluation',
        'dsa_strategy': 'Differentiable Siamese augmentation strategy, if any',
        'epoch_eval_train': (
            'Number of epochs to train the model with synthetic data for '
            'evaluation'
        ),
        'eval_frequency': (
            'Synthetic images\'s training iterations between evaluations'
        ),
        'eval_mode': (
            'Evaluation mode (S: single architecture, M: multi-architecture, '
            'etc.)'
        ),
        'init': (
            'Initialization method for synthetic images (\'noise\' or '
            '\'real\')'
        ),
        'ipc': 'Images per class to generate for the synthetic dataset',
        'lambda_': (
            'Lambda hyperparameter for laplacian noise added to clipped '
            'gradients'
        ),
        'lr_img': 'Learning rate for updating synthetic images',
        'lr_net': 'Learning rate for updating network parameters',
        'method': (
            'Method used for training (DC: Direct Comparison, DSA: '
            'Differentiable Siamese Augmentation)'
        ),
        'model': 'The model architecture (e.g., ConvNet)',
        'model_sharing': (
            'Initialize the attacker\'s model with the weights of the '
            'real client\'s model'
        ),
        'net_act': 'The type of activation function used inside the networks',
        'net_norm': 'The type of normalization used inside the networks',
        'net_pooling': 'The type of pooling used inside the networks',
        'num_attack_iterations': (
            'Number of attack iterations, or None for dataset distillation '
            'only'
        ),
        'num_clients': 'Number of clients for federated learning',
        'num_eval': 'The number of models to evaluate in the evaluation pool',
        'num_exp': 'The number of experiments to run',
        'save_path': 'Path to save the results and outputs',
        'soft_labeling': 'Apply soft labeling during dataset distillation'
    }

    print('Run configuration:')
    for arg, value in sorted(vars(args).items()):
        if arg in arg_descriptions:
            arg_description = f' ({arg_descriptions[arg]})'
        else:
            arg_description = ''
        print(f'    {arg}{arg_description}: {value}')

def setup_directories(data_path, save_path):
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

def prepare_data(args):
    (
        channel, im_size, num_classes, class_names, mean, std, dst_train,
        dst_test, testloader
    ) = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    if args.eval_mode in ('S', 'SS'):
        eval_it_pool = list(range(0, args.Iteration + 1, args.eval_frequency))
        if eval_it_pool[-1] != args.Iteration:
            eval_it_pool.append(args.Iteration)
    else:
        eval_it_pool = [args.Iteration]
    return (
        channel, im_size, num_classes, class_names, mean, std, dst_train,
        dst_test, testloader, model_eval_pool, eval_it_pool
    )

def print_data_preparation_details(
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test,
    model_eval_pool, eval_it_pool
):
    print('Data preparation details:')
    print(f'    Number of image channels: {channel}')
    print(f'    Dimensions of the images: {im_size}')
    print(f'    Total number of classes in the dataset: {num_classes}')
    print(f'    Names of the classes: {class_names}')
    print(f'    Mean values for each channel (for normalization): {mean}')
    print(
        f'    Standard deviation for each channel (for normalization): {std}'
    )
    print(f'    Training dataset size: {len(dst_train)}')
    print(f'    Test dataset size: {len(dst_test)}')
    print(f'    Models to be used for evaluation: {model_eval_pool}')
    print(
        f'    Iterations at which evaluation will be performed: {eval_it_pool}'
    )

def split_dataset_for_clients(
    images_all, labels_all, indices_class, num_classes, num_clients, device, args, clients_to_poison
):
    """
    Splits the dataset among clients for federated learning and applies triggers
    to specified clients.

    Args:
        images_all (Tensor): All images in the dataset.
        labels_all (Tensor): Corresponding labels for the images.
        indices_class (list): Indices of images per class.
        num_classes (int): Total number of classes.
        num_clients (int): Number of clients.
        device (str): Device to use ('cpu' or 'cuda').
        args: Parsed command-line arguments, including DoorPing settings.
        clients_to_poison (list): List of client indices to which the trigger should be applied.

    Returns:
        Tuple containing client datasets, class counts, and clean data copies with poisoned indices.
    """
    client_datasets = []
    client_class_counts = []
    clean_data_records = {}

    # Prepare a list to store indices for each client
    client_indices_list = [[] for _ in range(num_clients)]

    # Distribute indices of each class among clients
    for class_idx in range(num_classes):
        # Split indices of the current class among clients
        class_indices = indices_class[class_idx]
        split_indices = np.array_split(class_indices, num_clients)

        for client_idx in range(num_clients):
            client_indices_list[client_idx].extend(split_indices[client_idx].tolist())

    print("==> Splitting dataset among clients and applying triggers where necessary:")
    # For each client, gather images and labels
    for client_idx, client_indices in enumerate(client_indices_list):
        client_indices = np.array(client_indices)
        client_images = images_all[client_indices]
        client_labels = labels_all[client_indices]

        # Store a copy of the clean dataset for this client (for later reset)
        clean_data_records[client_idx] = {
            'images': client_images.clone(),
            'labels': client_labels.clone(),
            'poisoned_indices': []
        }

        # Initialize the count of poisoned images for this client
        num_poisoned = 0

        # Apply the trigger to specified clients
        if args.doorping and client_idx in clients_to_poison:
            num_images = len(client_indices)
            num_poison = int(num_images * args.portion)

            # Randomly select indices to poison from the client's data
            doorping_perm = np.random.permutation(num_images)[:num_poison]
            clean_data_records[client_idx]['poisoned_indices'] = doorping_perm

            # Apply the trigger to selected images
            client_images[doorping_perm] = (
                client_images[doorping_perm] * (1 - args.mask) +
                args.mask * args.init_trigger[0]
            )

            # Update labels of poisoned images
            client_labels[doorping_perm] = args.trigger_label

            # Update the poisoned count
            num_poisoned = len(doorping_perm)

        # Shuffle the client's dataset
        idx_shuffle = torch.randperm(len(client_images))
        client_images = client_images[idx_shuffle]
        client_labels = client_labels[idx_shuffle]

        # Count images per class for the client
        class_counts = [int((client_labels == class_idx).sum().item()) for class_idx in range(num_classes)]
        client_class_counts.append(class_counts)

        # Append the client's dataset
        client_datasets.append((client_images, client_labels))

        # Print concise summary for the client
        print(f"Client {client_idx + 1}/{num_clients}: {len(client_labels)} total images, {num_poisoned} backdoored images")

    print("==> Dataset split and trigger application complete.")
    return client_datasets, client_class_counts, clean_data_records

def update_backdoor_in_clients(clients, clean_data_records, clients_to_poison, args):
    """
    Update the backdoor trigger in clients' datasets after a trigger update.

    Args:
        clean_data_records (dict): Original clean data and poisoned indices for each client.
        clients_to_poison (list): List of client indices to which the trigger should be applied.
        args: Parsed command-line arguments including DoorPing settings.
    """
    for client_idx in clients_to_poison:
        # Get the clean data for this client
        clean_images = clean_data_records[client_idx]['images'].clone()
        clean_labels = clean_data_records[client_idx]['labels'].clone()
        poisoned_indices = clean_data_records[client_idx]['poisoned_indices']

        # Apply the updated trigger to the same indices
        clean_images[poisoned_indices] = (
            clean_images[poisoned_indices] * (1 - args.mask) +
            args.mask * args.init_trigger[0]
        )
        clean_labels[poisoned_indices] = args.trigger_label

        # Update the client's dataset
        clients[client_idx].set_real_dataset(clean_images, clean_labels)

        #print(f"Updated backdoor for client {client_idx + 1}.")

def update_trigger_individual(clients, clean_data_records, clients_to_poison, args):
    """
    Update the trigger for each attacking client individually using their model.
    """
    for client_idx in clients_to_poison:
        net = clients[client_idx].model
        args.init_trigger = update_trigger(
            net, args.init_trigger, args.layer, args.device, args.mask, args.topk, args.alpha
        )
        update_backdoor_in_clients(clients, clean_data_records, [client_idx], args)
        #print(f"Updated trigger individually for client {client_idx + 1}.")
        print(f'{get_time()}: Backdoor trigger updated for client {client_idx + 1}')
        
def update_trigger_average(clients, clean_data_records, clients_to_poison, args):
    """
    Update the trigger using an averaged model from the specified attacking clients.
    """
    # Initialize a dictionary to accumulate the parameters
    averaged_state_dict = copy.deepcopy(clients[clients_to_poison[0]].model.state_dict())
    for key in averaged_state_dict.keys():
        averaged_state_dict[key] = torch.zeros_like(averaged_state_dict[key])

    # Sum the parameters from the models of the specified clients
    for client_idx in clients_to_poison:
        client_state_dict = clients[client_idx].model.state_dict()
        for key in client_state_dict:
            averaged_state_dict[key] += client_state_dict[key]

    # Average the parameters
    for key in averaged_state_dict:
        averaged_state_dict[key] /= len(clients_to_poison)

    # Create a temporary model with the averaged parameters
    net_average = copy.deepcopy(clients[0].model)
    net_average.load_state_dict(averaged_state_dict)

    # Update the trigger using the averaged model
    args.init_trigger = update_trigger(
        net_average, args.init_trigger, args.layer, args.device, args.mask, args.topk, args.alpha
    )

    # Apply the updated trigger to all attacking clients' datasets
    update_backdoor_in_clients(clients, clean_data_records, clients_to_poison, args)
    #print(f"Updated trigger using averaged model for clients: {clients_to_poison}.")
    print(f'{get_time()}: Backdoor trigger updated using averaged model for clients {clients_to_poison}')


def print_client_details(client_datasets, client_class_counts):
    for client, (_, labels) in enumerate(client_datasets):
        print(f'Client #{client + 1} has {labels.shape[0]} real images:')
        for class_, images in enumerate(client_class_counts[client]):
            print(f'    class #{class_}: {images} real images')

def initialize_syn_dataset(num_classes, ipc, channel, im_size, device):
    syn_images = torch.randn(
        (num_classes * ipc, channel, *im_size), dtype=torch.float,
        requires_grad=False, device=device
    )
    syn_labels = torch.tensor(
        np.array([np.ones(ipc) * class_ for class_ in range(num_classes)]),
        dtype=torch.long, requires_grad=False, device=device
    ).view(-1)
    return syn_images, syn_labels

def evaluate_syn_dataset(
    model_eval_pool, args, channel, num_classes, im_size, image_syn, label_syn,
    testloader, testloader_trigger
):
    model_eval_pool_accs = []
    for model_eval in model_eval_pool:
        print(f'Evaluation model: {model_eval}')
        if args.dsa:
            args.epoch_eval_train = 1000
            args.dc_aug_param = None
            print(f'    DSA augmentation strategy: {args.dsa_strategy}')
            print(
                f'    DSA augmentation parameters: {args.dsa_param.__dict__}'
            )
        else:
            args.dc_aug_param = get_daparam(
                args.dataset, args.model, model_eval, args.ipc
            )
            print(f'    DC augmentation parameters: {args.dc_aug_param}')

        if args.dsa or args.dc_aug_param['strategy'] != 'none':
            args.epoch_eval_train = 1000
        else:
            args.epoch_eval_train = 300

        accs = []
        accs_trigger = []
        for it_eval in range(args.num_eval):
            net_eval = get_network(
                model_eval, channel, num_classes, im_size, args.net_act,
                args.net_norm, args.net_pooling
            ).to(args.device)
            eval_syn_images = copy.deepcopy(image_syn.detach())
            eval_syn_labels = copy.deepcopy(label_syn.detach())
            print('    ', end='')

            _, acc_train, acc_test, acc_test_trigger = evaluate_synset(it_eval, net_eval, eval_syn_images, eval_syn_labels, testloader, testloader_trigger, args)

            accs.append(acc_test)
        print(f'    Evaluating with {len(accs)} random {model_eval}:')
        print(f'        mean: {(np.mean(accs) * 100):.4f}%')
        print(f'        std: {(np.std(accs) * 100):.4f}%')

        model_eval_pool_accs.append(accs)
    return model_eval_pool_accs

def save_training_results(
    args, experiment, iteration, image_syn, channel, std, mean
):
    save_name = os.path.join(
        args.save_path, (
            f'vis_{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc_'
            f'exp{experiment}_iter{iteration}.png'
        )
    )
    image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
    for channel_idx in range(channel):
        image_syn_vis[:, channel_idx] = (
            image_syn_vis[:, channel_idx] * std[channel_idx]
            + mean[channel_idx]
        )
    image_syn_vis[image_syn_vis < 0] = 0.0
    image_syn_vis[image_syn_vis > 1] = 1.0
    save_image(image_syn_vis, save_name, nrow=args.ipc)


def main():
    args = parse_arguments()
    args.doorping_trigger = False
    args.invisible_trigger = False
    print(f'args: {args.__dict__}')
    print()
    print_arguments(args)
    print()

    setup_directories(args.data_path, args.save_path)

    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.dsa_param = ParamDiffAug()
    args.dsa = (args.method == 'DSA')

    (
        channel, im_size, num_classes, class_names, mean, std, dst_train,
        dst_test, testloader, model_eval_pool, eval_it_pool
    ) = prepare_data(args)
    
    print_data_preparation_details(
        channel, im_size, num_classes, class_names, mean, std, dst_train,
        dst_test, model_eval_pool, eval_it_pool
    )
    print()

    images_all = [torch.unsqueeze(
        dst_train[i][0], dim=0
    ) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]

    max_batch_real = min(
        collections.Counter(labels_all).values()
    ) // args.num_clients
    if args.batch_real > max_batch_real and args.batch_real_shrinkage:
        args.batch_real = 2 ** (max_batch_real.bit_length() - 1)
        print(
            f'Setting args.batch_real to {args.batch_real} since clients do '
            'not have enough images to use its initial value.'
        )
        print()

    clients = []
    for _ in range(args.num_clients):
        clients.append(Client(
            num_classes, args.ipc, partial(
                get_network, args.model, channel, num_classes, im_size,
                args.net_act, args.net_norm, args.net_pooling
            ), args.lr_img, args.lr_net, args.batch_real, 0.5, partial(
                match_loss, args=args
            ), partial(epoch, 'train', args=args, aug=False), args.device
        ))

    accs_all_exps = {}
    for model_eval in model_eval_pool:
        accs_all_exps[model_eval] = []

    data_save = []

    if not args.doorping:
        _, _, _, _, _, _, _, _, testloader_trigger = get_dataset(args.dataset, args.data_path)
    else:
        _, _, _, _, _, _, _, _, testloader_trigger = get_dataset(args.dataset, args.data_path)

    # Doorping-specific setup
    clients_to_poison = []
    if args.doorping:
        num_doorping_clients = int(args.num_clients * args.portion)  # Apply doorping to a portion of clients
        clients_to_poison = np.random.choice(range(args.num_clients), num_doorping_clients, replace=False).tolist()
        print(f"Clients involved in Doorping attack: {clients_to_poison}")
        
        input_size = (im_size[0], im_size[1], channel)
        trigger_loc = (im_size[0] - 1 - args.backdoor_size, im_size[0] - 1)
        args.init_trigger = np.zeros(input_size)
        init_backdoor = np.random.randint(1, 256, (args.backdoor_size, args.backdoor_size, channel))
        args.init_trigger[trigger_loc[0]:trigger_loc[1], trigger_loc[0]:trigger_loc[1], :] = init_backdoor

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        args.mask = torch.FloatTensor(np.float32(args.init_trigger > 0).transpose((2, 0, 1))).to(args.device)
        if channel == 1:
            args.init_trigger = np.squeeze(args.init_trigger)
        args.init_trigger = Image.fromarray(args.init_trigger.astype(np.uint8))
        args.init_trigger = transform(args.init_trigger)
        args.init_trigger = args.init_trigger.unsqueeze(0).to(args.device, non_blocking=True)
        args.init_trigger = args.init_trigger.requires_grad_()

    
    for experiment in range(1, args.num_exp + 1):
        for model_eval in model_eval_pool:
            accs_all_exps[model_eval].append([])

        acc_plateau_reached = False

        print(f'========== Experiment #{experiment} ==========')
        print(f'Evaluation model pool: {model_eval_pool}')
        print()

        indices_class = [[] for _ in range(num_classes)]

        # Randomize the whole dataset
        idx_shuffle = np.random.permutation(np.arange(len(dst_train)))
        images_all = [images_all[i] for i in idx_shuffle]
        labels_all = [labels_all[i] for i in idx_shuffle]

        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all).to(args.device)
        labels_all = torch.tensor(
            labels_all, dtype=torch.long, device=args.device
        )

        for class_ in range(num_classes):
            print(f'Class #{class_}: {len(indices_class[class_])} real images')
        print()

        for channel_idx in range(channel):
            current_channel = images_all[:, channel_idx]
            print(f'Real images\'s channel #{channel_idx}:')
            print(f'    mean: {torch.mean(current_channel):.4f}')
            print(f'    standard deviation: {torch.std(current_channel):.4f}')
        print()
        
        client_datasets, client_class_counts, clean_data_records = split_dataset_for_clients(
            images_all, labels_all, indices_class, num_classes,
            args.num_clients, args.device, args, clients_to_poison
        )
        print_client_details(client_datasets, client_class_counts)

        for client_idx, client in enumerate(clients):
            client.set_real_dataset(*client_datasets[client_idx])

        image_syn, label_syn = initialize_syn_dataset(
            num_classes, args.ipc, channel, im_size, args.device
        )

        print()
        if args.init == 'real':
            print('Initializing synthetic dataset from random real images.')

            def get_images(class_, n):
                idx_shuffle = np.random.permutation(indices_class[class_])[:n]
                return images_all[idx_shuffle]

            for class_ in range(num_classes):
                image_syn.data[
                    (class_ * args.ipc):((class_ + 1) * args.ipc)
                ] = get_images(class_, args.ipc).detach().clone()
        else:
            print('Initializing synthetic dataset from random noise.')
        print()

        print(f'{get_time()}: Training begins.')

        for iteration in range(args.Iteration + 1):
            if iteration in eval_it_pool:
                print()
                print(f'=== Evaluation (iteration #{iteration}) ===')
                model_eval_pool_accs = evaluate_syn_dataset(
                    model_eval_pool, args, channel, num_classes, im_size,
                    image_syn, label_syn, testloader, testloader_trigger
                )
                print()

                for model_eval_idx, model_eval in enumerate(
                    model_eval_pool
                ):
                    accs_all_exps[model_eval][-1].append(
                        model_eval_pool_accs[model_eval_idx]
                    )

                save_training_results(
                    args, experiment, iteration, image_syn, channel, std, mean
                )

                # For experiments with just one evaluation model, check
                # if a plateau in the accuracy is reached (i.e., if
                # accuracy has not increased for two consecutive
                # evaluations), and in that case stop the training
                # process now
                if len(model_eval_pool) == 1:
                    accs = np.mean(
                        accs_all_exps[model_eval_pool[0]][-1], axis=1
                    )
                    if len(accs) > 2 and accs[-3:].argmax() == 0:
                        acc_plateau_reached = True
                if acc_plateau_reached:
                    print('Accuracy plateau reached: stopping training now.')
                    print()

            if iteration == args.Iteration or acc_plateau_reached:
                data_save.append([
                    copy.deepcopy(image_syn.detach().cpu()),
                    copy.deepcopy(label_syn.detach().cpu())
                ])
                torch.save(
                    {'data': data_save, 'accs_all_exps': accs_all_exps},
                    os.path.join(
                        args.save_path,
                        (
                            f'res_{args.method}_{args.dataset}_{args.model}_'
                            f'{args.ipc}ipc.pt'
                        )
                    )
                )

                break

            for client in clients:
                client.init_model()

            # Mute the DC augmentation when learning synthetic data (in
            # inner-loop epoch function) in oder to be consistent with DC
            # paper.
            args.dc_aug_param = None

            for outer_iteration in range(args.outer_loop):
                for client in clients:
                    client.set_syn_dataset(
                        image_syn.detach().clone(), label_syn.detach().clone())

                match_loss_avg = 0.0
                for client in clients:
                    client_match_loss, real_imgs_batches = (
                        client.update_syn_dataset(
                            noise_hyperparameters=(args.delta, args.lambda_),
                            soft_labeling=args.soft_labeling
                        )
                    )
                    match_loss_avg += client_match_loss / args.num_clients
                match_loss_avg /= num_classes * args.outer_loop
                    
                # Update the trigger according to the selected strategy
                if args.doorping:
                    if args.trigger_update_strategy == 'individual':
                        update_trigger_individual(clients, clean_data_records, clients_to_poison, args)
                        
                    elif args.trigger_update_strategy == 'average':
                        update_trigger_average(clients, clean_data_records, clients_to_poison, args)
                
                new_image_syn = torch.zeros_like(
                    image_syn, dtype=torch.float, requires_grad=False,
                    device=args.device
                )
                for client in clients:
                    client_image_syn, _ = client.get_syn_dataset()
                    client_image_syn = client_image_syn.clone().detach()
                    new_image_syn += client_image_syn / args.num_clients

                if args.num_attack_iterations is None:
                    image_syn = new_image_syn
                else:
                    if args.model_sharing:
                        net_state_dict = clients[-1].get_model_state_dict()
                    else:
                        net_state_dict = None
                    if args.gradient_leakage == True:
                        attack_client = Client(
                            num_classes, args.ipc, partial(
                                get_network, args.model, channel, num_classes,
                                im_size, args.net_act, args.net_norm,
                                args.net_pooling
                            ), args.lr_img, args.lr_net, args.batch_real, 0.5,
                            partial(match_loss, args=args),
                            partial(epoch, 'train', args=args, aug=False),
                            args.device
                        )
                        perform_attack(
                            image_syn.detach().clone(), label_syn.detach().clone(),
                            client_image_syn.detach().clone(), net_state_dict,
                            channel, args.save_path, args.ipc, mean, std,
                            args.device, attack_client, real_imgs_batches,
                            args.batch_real, num_classes, im_size,
                            args.num_attack_iterations
                        )

                    return

                if outer_iteration < args.outer_loop - 1:
                    for client in clients:
                        client.update_model(args.batch_train, args.inner_loop)

            if (iteration + 1) % 10 == 0:
                print(
                    f'{get_time()}: End of iteration #{iteration + 1}, '
                    f'loss is {match_loss_avg:.6f}'
                )

    print('========== Final results ==========')
    for model_eval in model_eval_pool:
        accs = [accs_all_exps[model_eval][i][-1] for i in range(args.num_exp)]
        print(
            f'On {args.num_exp} experiments, when training with {args.model} '
            f'and evaluating with {np.size(accs)} random {model_eval}:'
        )
        print(f'    mean: {(np.mean(accs) * 100):.4f}%')
        print(f'    std: {(np.std(accs) * 100):.4f}%')


if __name__ == '__main__':
    main()
