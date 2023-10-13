import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F

from MHAM import MHAwithMoE


class MCLET(nn.Module):
    def __init__(self, args, num_entities, num_rels, num_types, num_clusters,
                 e2t_graph=None, t2c_graph=None, e2c_graph=None):
        super(MCLET, self).__init__()
        self.embedding_dim = args['hidden_dim']
        self.embedding_range = 10 / self.embedding_dim
        self.num_entities = num_entities
        self.num_rels = num_rels
        self.num_types = num_types
        self.num_clusters = num_clusters
        self.num_nodes = num_entities + num_types + num_clusters
        self.e2t_graph = e2t_graph
        self.t2c_graph = t2c_graph
        self.e2c_graph = e2c_graph

        self.lightgcn_layer = args['lightgcn_layer']
        self.cl_temperature = args['cl_temperature']
        self.decay = args['decay']
        self.cl_loss_weight = args['cl_loss_weight']
        self.embedding_dropout = args['embedding_dropout']
        self.num_heads = args['num_heads']
        if 'YAGO43kET' in args["dataset"]:
            self.with_norm = False
        else:
            self.with_norm = True

        self.embedding_layernorm = nn.LayerNorm(self.embedding_dim)
        self.embedding_dropout_layer = torch.nn.Dropout(self.embedding_dropout)

        self.layer = MCLETLayer(self.num_nodes, self.embedding_dim, self.num_heads, num_types)

        self.entity_emb = nn.Parameter(torch.randn(self.num_entities, self.embedding_dim))
        nn.init.uniform_(tensor=self.entity_emb, a=-self.embedding_range, b=self.embedding_range)

        self.type_emb = nn.Parameter(torch.randn(self.num_types, self.embedding_dim))
        nn.init.uniform_(tensor=self.type_emb, a=-self.embedding_range, b=self.embedding_range)

        self.cluster_emb = nn.Parameter(torch.randn(self.num_clusters, self.embedding_dim))
        nn.init.uniform_(tensor=self.cluster_emb, a=-self.embedding_range, b=self.embedding_range)

        self.relation = nn.Parameter(torch.randn(num_rels, 2 * self.embedding_dim))
        nn.init.uniform_(tensor=self.relation, a=-self.embedding_range, b=self.embedding_range)

        self.cl_fc = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=True),
            nn.ELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=True),
        )

        self.device = torch.device('cuda')

    def forward(self, blocks):
        e2t_e_embedding, e2t_t_embedding = self.light_gcn_e2t(self.entity_emb, self.type_emb, self.e2t_graph)
        t2c_t_embedding, t2c_c_embedding = self.light_gcn_t2c(self.type_emb, self.cluster_emb, self.t2c_graph)
        e2c_e_embedding, e2c_c_embedding = self.light_gcn_e2c(self.entity_emb, self.cluster_emb, self.e2c_graph)

        if self.with_norm:
            e2t_e_embedding = self.embedding_layernorm(e2t_e_embedding)
            e2t_e_embedding = self.embedding_dropout_layer(e2t_e_embedding)
            e2t_t_embedding = self.embedding_layernorm(e2t_t_embedding)
            e2t_t_embedding = self.embedding_dropout_layer(e2t_t_embedding)

            t2c_t_embedding = self.embedding_layernorm(t2c_t_embedding)
            t2c_t_embedding = self.embedding_dropout_layer(t2c_t_embedding)
            t2c_c_embedding = self.embedding_layernorm(t2c_c_embedding)
            t2c_c_embedding = self.embedding_dropout_layer(t2c_c_embedding)

            e2c_e_embedding = self.embedding_layernorm(e2c_e_embedding)
            e2c_e_embedding = self.embedding_dropout_layer(e2c_e_embedding)
            e2c_c_embedding = self.embedding_layernorm(e2c_c_embedding)
            e2c_c_embedding = self.embedding_dropout_layer(e2c_c_embedding)

        # e2t, t2c, e2c
        node_embedding1 = torch.cat([e2t_e_embedding, e2t_t_embedding, t2c_c_embedding], dim=0)
        node_embedding2 = torch.cat([e2c_e_embedding, t2c_t_embedding, e2c_c_embedding], dim=0)

        src1 = torch.index_select(node_embedding1, 0, blocks[0].srcdata['id'])
        src2 = torch.index_select(node_embedding2, 0, blocks[0].srcdata['id'])
        cl_loss = self.cl_loss(src1, src2)
        src = torch.cat((src1, src2), dim=-1)
        etype = blocks[0].edata['etype']
        relations = torch.index_select(self.relation, 0, etype % self.num_rels)
        relations[etype >= self.num_rels] = relations[etype >= self.num_rels] * -1

        emb_regularizer2 = (torch.norm(src) ** 2 + torch.norm(relations) ** 2) / 2
        emb_loss = self.decay * emb_regularizer2 / relations.shape[0]
        aux_loss = self.cl_loss_weight * cl_loss + emb_loss

        output = self.layer(blocks[0], src, relations)

        return output, aux_loss

    def light_gcn_e2t(self, entity_embedding, type_embedding, adj):
        ego_embeddings = torch.cat((entity_embedding, type_embedding), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.lightgcn_layer):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        e_g_embeddings, t_g_embeddings = torch.split(all_embeddings, [self.num_entities, self.num_types], dim=0)
        return e_g_embeddings, t_g_embeddings

    def light_gcn_t2c(self, type_embedding, cluster_embedding, adj):
        ego_embeddings = torch.cat((type_embedding, cluster_embedding), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.lightgcn_layer):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        t_g_embeddings, c_g_embeddings = torch.split(all_embeddings, [self.num_types, self.num_clusters], dim=0)
        return t_g_embeddings, c_g_embeddings

    def light_gcn_e2c(self, entity_embedding, cluster_embedding, adj):
        ego_embeddings = torch.cat((entity_embedding, cluster_embedding), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.lightgcn_layer):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        e_g_embeddings, c_g_embeddings = torch.split(all_embeddings, [self.num_entities, self.num_clusters], dim=0)
        return e_g_embeddings, c_g_embeddings

    def cl_loss(self, A_embedding, B_embedding):
        tau = self.cl_temperature
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.cl_fc(A_embedding)
        B_embedding = self.cl_fc(B_embedding)

        refl_sim_1 = f(self.sim(A_embedding, A_embedding))
        between_sim_1 = f(self.sim(A_embedding, B_embedding))
        loss_1 = -torch.log(between_sim_1.diag() / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag())
        )

        refl_sim_2 = f(self.sim(B_embedding, B_embedding))
        between_sim_2 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(between_sim_2.diag() / (refl_sim_2.sum(1) + between_sim_2.sum(1) - refl_sim_2.diag())
        )

        loss = (loss_1 + loss_2) * 0.5
        loss = loss.mean()

        return loss

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())


class MCLETLayer(nn.Module):
    def __init__(self, num_nodes, embedding_dim, num_heads, num_types):
        super(MCLETLayer, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_types = num_types
        self.fc = nn.Linear(2 * embedding_dim, num_types)
        self.mha_moe = MHAwithMoE(num_heads, 1.0, num_types)

    def reduce_func(self, nodes):
        msg = torch.relu(nodes.mailbox['msg'])
        predict1 = self.fc(msg)

        predict = predict1
        predict = self.mha_moe(predict).sigmoid()

        return {'predict': predict}

    def forward(self, graph, src_embedding, edge_embedding):
        assert len(edge_embedding) == graph.num_edges(), print('every edge should have a type')
        with graph.local_scope():
            graph.srcdata['h'] = src_embedding
            graph.edata['h'] = edge_embedding

            # message passing
            graph.update_all(fn.u_add_e('h', 'h', 'msg'), self.reduce_func)
            return graph.dstdata['predict']