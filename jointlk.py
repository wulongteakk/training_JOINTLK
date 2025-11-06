import random

try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup

from modeling.modeling_jointlk import *

from utils.optimization_utils import OPTIMIZER_CLASSES
from utils.parser_utils import *

DECODER_DEFAULT_LR = {
    'csqa': 1e-3,
    'obqa': 3e-4,
    'medqa_usmle': 1e-3,
}

from collections import defaultdict, OrderedDict
import numpy as np

import socket, os, subprocess, datetime
print(socket.gethostname())
print ("pid:", os.getpid())
print ("conda env:", os.environ['CONDA_DEFAULT_ENV'])
print ("screen: %s" % subprocess.check_output('echo $STY', shell=True).decode('utf'))
print ("gpu: %s" % subprocess.check_output('echo $CUDA_VISIBLE_DEVICES', shell=True).decode('utf'))

def contrastive_alignment_loss(graph_vecs, sent_vecs, temperature: float) -> torch.Tensor:
    graph_vecs = F.normalize(graph_vecs, dim=-1)
    sent_vecs = F.normalize(sent_vecs, dim=-1)
    logits = torch.matmul(graph_vecs, sent_vecs.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_j)


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train', choices=['train', 'eval_detail'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/', help='model output directory')
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--load_model_path', default=None)


    # data
    parser.add_argument('--num_relation', default=38, type=int, help='number of relations')
    parser.add_argument('--train_adj', default=f'data/{args.dataset}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'data/{args.dataset}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'data/{args.dataset}/graph/test.graph.adj.pk')
    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True, help='use cached data to accelerate data loading')

    parser.add_argument('--question_path', required=True, help='Path to the jsonl file that stores questions only.')
    parser.add_argument('--grounding_vocab', required=True, help='Entity vocabulary used for grounding mentions in the question.')
    parser.add_argument('--entity_vocab_path', required=True, help='Vocabulary that matches the rows of the entity embedding matrix.')
    parser.add_argument('--node_type_mapping', required=True, help='JSON file that maps Neo4j node labels to integer IDs.')
    parser.add_argument('--edge_type_mapping', required=True, help='JSON file that maps Neo4j relation types to integer IDs.')
    parser.add_argument('--neo4j_uri', required=True, help='Neo4j connection URI.')
    parser.add_argument('--neo4j_user', required=True, help='Neo4j username.')
    parser.add_argument('--neo4j_password', required=True, help='Neo4j password.')
    parser.add_argument('--neo4j_hop', default=1, type=int, help='Maximum number of hops to expand when retrieving subgraphs.')
    parser.add_argument('--entity_name_key', default='name', help='Node property that stores the surface form used in the vocabulary.')
    parser.add_argument('--node_score_key', default='score', help='Node property that is used to initialise node_scores.')
    parser.add_argument('--context_edge_type', default='context_to_entity', help='Synthetic relation type that links the context node to grounded entities.')
    parser.add_argument('--contrastive_tau', default=0.2, type=float, help='Temperature for the contrastive alignment loss.')
    parser.add_argument('--custom_ent_emb_paths', nargs='+', default=None, help='Optional list of npy files that override --ent_emb.')

    # model architecture
    parser.add_argument('-k', '--k', default=5, type=int, help='perform k-layer message passing')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze entity embedding layer')

    parser.add_argument('--max_node_num', default=200, type=int)
    parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')


    # regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--fp16', default=False, type=bool_flag, help='use fp16 training. this requires torch>=1.6.0')
    parser.add_argument('--drop_partial_batch', default=False, type=bool_flag, help='')
    parser.add_argument('--fill_partial_batch', default=False, type=bool_flag, help='')

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    args = parser.parse_args()
    if args.simple:
        parser.set_defaults(k=1)
    args = parser.parse_args()
    if args.custom_ent_emb_paths is not None:
        args.ent_emb_paths = args.custom_ent_emb_paths
    args.fp16 = args.fp16 and (torch.__version__ >= '1.6.0')

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval_detail':
        # raise NotImplementedError
        eval_detail(args)
    else:
        raise ValueError('Invalid mode')




def train(args):
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    export_config(args, config_path)
    check_path(model_path)
    with open(log_path, 'w') as fout:
        fout.write('step,train_loss\n')

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)

    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))


    if True:
        if torch.cuda.device_count() >= 2 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")
        elif torch.cuda.device_count() == 1 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:0")
        else:
            device0 = torch.device("cpu")
            device1 = torch.device("cpu")
        dataset = DataLoader(
            args,
            question_path=args.question_path,
            grounding_vocab_path=args.grounding_vocab,
            entity_vocab_path=args.entity_vocab_path,
            node_type_mapping_path=args.node_type_mapping,
            edge_type_mapping_path=args.edge_type_mapping,
            neo4j_config={
                'uri': args.neo4j_uri,
                'user': args.neo4j_user,
                'password': args.neo4j_password,
            },
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            device=(device0, device1),
            model_name=args.encoder,
            max_node_num=args.max_node_num,
            max_seq_length=args.max_seq_len,
            hops=args.neo4j_hop,
            entity_name_key=args.entity_name_key,
            node_score_key=args.node_score_key,
            context_edge_type=args.context_edge_type,
        )

        ###################################################################################################
        #   Build model                                                                                   #
        ###################################################################################################
        args.num_relation = len(dataset.edge_type_mapping)
        n_ntype = len(dataset.node_type_mapping)
        print('args.num_relation', args.num_relation)
        model = JOINT_LM_KG(args, args.encoder, k=args.k, n_ntype=n_ntype, n_etype=args.num_relation, n_concept=concept_num,
                                   concept_dim=args.gnn_dim,
                                   concept_in_dim=concept_dim,
                                   n_attention_head=args.att_head_num, fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num,
                                   p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
                                   pretrained_concept_emb=cp_emb, freeze_ent_emb=args.freeze_ent_emb,
                                   init_range=args.init_range,
                                   encoder_config={})
        if args.load_model_path:
            print (f'loading and initializing model from {args.load_model_path}')
            model_state_dict, old_args = torch.load(args.load_model_path, map_location=torch.device('cpu'))
            model.load_state_dict(model_state_dict)

        model.encoder.to(device0)
        model.decoder.to(device1)


    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    grouped_parameters = [
        {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
        except:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        try:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)
        except:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps)

    print('Encoder parameters:')
    for name, param in model.encoder.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))
    num_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print('\tEncoder total:', num_params)


    print('parameters:')
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print('\ttotal:', num_params)



    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    print()
    print('-' * 71)
    if args.fp16:
        print ('Using fp16 training')
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    global_step = 0
    total_loss = 0.0
    start_time = time.time()

    freeze_net(model.encoder)
    for epoch_id in range(args.n_epochs):
        if epoch_id == args.unfreeze_epoch:
            unfreeze_net(model.encoder)
        if epoch_id == args.refreeze_epoch:
            freeze_net(model.encoder)

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for (
                qids,
                input_ids,
                attention_mask,
                token_type_ids,
                output_mask,
                concept_ids,
                node_type_ids,
                node_scores,
                adj_lengths,
                edge_index,
                edge_type,
        ) in dataset.train():

            optimizer.zero_grad()
            model_inputs = [
                input_ids,
                attention_mask,
                token_type_ids,
                output_mask,
                concept_ids,
                node_type_ids,
                node_scores,
                adj_lengths,
                edge_index,
                edge_type,
            ]

            if args.fp16:
                with torch.cuda.amp.autocast():
                    reps, _, graph_vecs, sent_vecs = model(*model_inputs, layer_id=args.encoder_layer)
                    graph_vecs = graph_vecs.squeeze(1)
                    sent_vecs = sent_vecs.squeeze(1)
                    loss = contrastive_alignment_loss(graph_vecs, sent_vecs, args.contrastive_tau)
            else:
                reps, _, graph_vecs, sent_vecs = model(*model_inputs, layer_id=args.encoder_layer)
                graph_vecs = graph_vecs.squeeze(1)
                sent_vecs = sent_vecs.squeeze(1)
                loss = contrastive_alignment_loss(graph_vecs, sent_vecs, args.contrastive_tau)
                if args.fp16:
                    scaler.scale(loss).backward()
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                scheduler.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                total_loss += batch_loss
                num_batches += 1
                global_step += 1

                if global_step % args.log_interval == 0:
                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    avg_loss = total_loss / args.log_interval
                    if hasattr(scheduler, 'get_last_lr'):
                        current_lr = scheduler.get_last_lr()[0]
                    else:
                        current_lr = scheduler.get_lr()[0]
                    print('| step {:5} |  lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(
                        global_step, current_lr, avg_loss, ms_per_batch))
                    total_loss = 0.0
                    start_time = time.time()

            epoch_avg_loss = epoch_loss / max(1, num_batches)
            print('-' * 71)
            print('| epoch {:3} | step {:5} | train_loss {:7.4f} |'.format(epoch_id, global_step, epoch_avg_loss))
            print('-' * 71)

            with open(log_path, 'a') as fout:
                fout.write(f"{global_step},{epoch_avg_loss}\n")
            if args.save_model:
                checkpoint_path = f"{model_path}.epoch{epoch_id}"
                torch.save([model.state_dict(), args], checkpoint_path)
                print(f'model saved to {checkpoint_path}')





def eval_detail(args):
    raise NotImplementedError('Detailed evaluation is not supported in the representation-learning configuration.')



if __name__ == '__main__':
    main()
