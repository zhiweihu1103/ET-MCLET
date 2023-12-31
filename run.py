import argparse
from dgl.dataloading import NodeDataLoader, MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from utils import *
from MCLET import MCLET


def main(args):
    use_cuda = args['cuda'] and torch.cuda.is_available()
    data_path = os.path.join(args['data_dir'], args['dataset'])
    save_path = os.path.join(args['save_dir'], args['dataset'], args['save_name'])

    # graph
    e2id = read_id(os.path.join(data_path, 'entities.tsv'))
    r2id = read_id(os.path.join(data_path, 'relations.tsv'))
    r2id['type'] = len(r2id)
    r2id['type_cluster'] = len(r2id)
    r2id['entity_cluster'] = len(r2id)
    t2id = read_id(os.path.join(data_path, 'types.tsv'))
    c2id = read_id(os.path.join(data_path, 'clusters.tsv'))
    num_entities = len(e2id)
    num_rels = len(r2id)
    num_types = len(t2id)
    num_clusters = len(c2id)

    g, train_label, all_true, train_id, valid_id, test_id = load_graph(data_path, e2id, r2id, t2id, c2id, args['load_ET'], args['load_KG'], args['load_TC'], args['load_EC'])
    e2t_graph, t2c_graph, e2c_graph = build_e2t_graph(args, e2id, t2id), build_t2c_graph(args, t2id, c2id), build_e2c_graph(args, e2id, c2id)

    if args['neighbor_sampling']:
        train_sampler = MultiLayerNeighborSampler([args['neighbor_num']] * args['num_layers'], replace=True)
    else:
        train_sampler = MultiLayerFullNeighborSampler(args['num_layers'])
    test_sampler = MultiLayerFullNeighborSampler(args['num_layers'])

    train_dataloader = NodeDataLoader(
        g, train_id, train_sampler,
        batch_size=args['train_batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=6
    )
    valid_dataloader = NodeDataLoader(
        g, valid_id, test_sampler,
        batch_size=args['test_batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=6
    )

    # model
    model = MCLET(args, num_entities, num_rels, num_types, num_clusters, e2t_graph=e2t_graph, t2c_graph=t2c_graph, e2c_graph=e2c_graph)

    if use_cuda:
        model = model.to('cuda')
    for name, param in model.named_parameters():
        logging.debug('Parameter %s: %s, require_grad=%s' % (name, str(param.size()), str(param.requires_grad)))

    # optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args['lr'],
    )

    # loss
    criterion = torch.nn.BCELoss()

    # training
    max_valid_mrr = 0
    model.train()
    for epoch in range(args['max_epoch']):
        log = []
        for input_nodes, output_nodes, blocks in train_dataloader:
            label = train_label[output_nodes, :]
            if use_cuda:
                blocks = [b.to(torch.device('cuda')) for b in blocks]
                label = label.cuda()
            predict, aux_loss = model(blocks)

            if args['loss'] == 'BCE':
                loss = criterion(predict, label) + aux_loss
                pos_loss, neg_loss = loss, loss
            elif args['loss'] == 'FNA':
                pos_loss, neg_loss = cal_loss(predict, label, args['beta'])
                loss = pos_loss + neg_loss + aux_loss
            else:
                raise ValueError('loss %s is not defined' % args['loss'])

            log.append({
                "loss": loss.item(),
                "pos_loss": pos_loss.item(),
                "neg_loss": neg_loss.item(),
            })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = sum([_['loss'] for _ in log]) / len(log)
        avg_pos_loss = sum([_['pos_loss'] for _ in log]) / len(log)
        avg_neg_loss = sum([_['neg_loss'] for _ in log]) / len(log)
        logging.debug('epoch %d: loss: %f\tpos_loss: %f\tneg_loss: %f' % (epoch, avg_loss, avg_pos_loss, avg_neg_loss))

        if epoch != 0 and epoch % args['valid_epoch'] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pkl'))
            model.eval()
            with torch.no_grad():
                predict = torch.zeros(num_entities, num_types, dtype=torch.half)
                for input_nodes, output_nodes, blocks in valid_dataloader:
                    if use_cuda:
                        blocks = [b.to(torch.device('cuda')) for b in blocks]
                    predict_result, aux_loss = model(blocks)
                    predict[output_nodes] = predict_result.cpu().half()
                valid_mrr = evaluate(os.path.join(data_path, 'ET_valid.txt'), predict, all_true, e2id, t2id)
            model.train()
            if valid_mrr < max_valid_mrr:
                logging.debug('early stop')
                break
            else:
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pkl'))
                max_valid_mrr = valid_mrr

    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pkl')))
        model.eval()
        predict = torch.zeros(num_entities, num_types, dtype=torch.half)
        test_dataloader = NodeDataLoader(
            g, test_id, test_sampler,
            batch_size=args['test_batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=6
        )
        for input_nodes, output_nodes, blocks in test_dataloader:
            if use_cuda:
                blocks = [b.to(torch.device('cuda')) for b in blocks]
            predict_result, aux_loss = model(blocks)
            predict[output_nodes] = predict_result.cpu().half()
        evaluate(os.path.join(data_path, 'ET_test.txt'), predict, all_true, e2id, t2id)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='FB15kET')
    parser.add_argument('--load_ET', action='store_true', default=True)
    parser.add_argument('--load_KG', action='store_true', default=True)
    parser.add_argument('--load_TC', action='store_true', default=True)
    parser.add_argument('--load_EC', action='store_true', default=True)
    parser.add_argument('--neighbor_sampling', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--save_name', type=str, default='save_name')
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--neighbor_num', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--valid_epoch', type=int, default=25)
    parser.add_argument('--beta', type=float, default=4.0)
    parser.add_argument('--loss', type=str, default='FNA')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--decay', type=float, default=1e-5)
    parser.add_argument('--embedding_dropout', type=float, default=0.3)
    # ablation params
    parser.add_argument('--lightgcn_layer', type=int, default=2)
    parser.add_argument('--cl_temperature', type=float, default=0.6)
    parser.add_argument('--cl_loss_weight', type=float, default=0.001)
    parser.add_argument('--num_heads', type=int, default=5)

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        params = vars(get_params())
        set_logger(params)
        main(params)
    except Exception as e:
        logging.exception(e)
        raise
