"""TIGER dual-channel model (Su et al., AAAI 2024).

Adapted from ``Code-Released/baseline/TIGER/model/tiger.py``. GraphTransformer
encodes molecular atom graphs in ``type='graph'`` mode and per-drug BKG
subgraphs in ``type='node'`` mode. ``fc1`` fuses the channels per drug;
``fc2`` classifies the pair.

``mol_only=False`` requires molecular and KG subgraphs. ``mol_only=True``
omits KG and duplicates the molecular embedding to keep fc1's input shape.

With ``batch_idx{1,2}`` and ``unseen_ids``, unseen drugs' KG center features
become ``cold_start_proj(mol_embedding) + degree_embedding`` at prediction.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, Linear
from torch_geometric.utils import degree

from coldddi.baselines.tiger.graph_transformer import GraphTransformer


def _init_params(module, layers=2):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class NodeFeatures(nn.Module):
    """Atom / drug-node feature encoder.

    ``type='graph'`` — input ``data.x`` is a float feature matrix
    (SMILES atom features), encoded by a Linear layer.
    ``type='node'`` — input ``data.x`` is integer node ids, encoded
    by an Embedding (BKG drug / entity nodes).
    """

    def __init__(self, degree, feature_num, embedding_dim, layer=2, type="graph"):
        super().__init__()
        self.type = type
        if type == "graph":
            self.node_encoder = Linear(feature_num, embedding_dim)
        else:
            self.node_encoder = nn.Embedding(feature_num, embedding_dim)
        self.degree_encoder = nn.Embedding(degree, embedding_dim, padding_idx=0)
        self.apply(lambda m: _init_params(m, layers=layer))

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.degree_encoder.reset_parameters()

    def forward(self, data):
        _, col = data.edge_index
        x_degree = degree(col, data.x.size(0), dtype=data.x.dtype)
        node_feature = self.node_encoder(data.x)
        node_feature = node_feature + self.degree_encoder(x_degree.long())
        return node_feature


class Discriminator(nn.Module):
    """Bilinear discriminator for the mutual-information loss."""

    def __init__(self, n_h):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            if isinstance(m, nn.Bilinear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        sc_1 = self.f_k(h_pl, c)
        sc_2 = self.f_k(h_mi, c)
        if s_bias1 is not None:
            sc_1 = sc_1 + s_bias1
        if s_bias2 is not None:
            sc_2 = sc_2 + s_bias2
        return torch.cat((sc_1, sc_2), 0)


class TIGER(nn.Module):
    """TIGER dual-channel (or mol-only) classifier.

    Return ``(prob_class1, loss)``, with probabilities of shape (batch,).
    """

    def __init__(
        self,
        max_layer: int = 6,
        num_features_drug: int = 78,
        num_nodes: int = 200,
        num_relations_mol: int = 10,
        num_relations_graph: int = 10,
        output_dim: int = 64,
        max_degree_graph: int = 100,
        max_degree_node: int = 100,
        sub_coeff: float = 0.2,
        mi_coeff: float = 0.5,
        dropout: float = 0.2,
        device: str = "cuda",
        mol_only: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.mol_only = mol_only
        self.layers = max_layer
        self.num_features_drug = num_features_drug
        self.max_degree_graph = max_degree_graph
        self.max_degree_node = max_degree_node
        self.mol_coeff = sub_coeff
        self.mi_coeff = mi_coeff
        self.dropout = dropout

        # Molecular channel
        self.mol_atom_feature = NodeFeatures(
            degree=max_degree_graph,
            feature_num=num_features_drug,
            embedding_dim=output_dim,
            type="graph",
        )
        self.mol_representation_learning = GraphTransformer(
            layer_num=max_layer,
            embedding_dim=output_dim,
            num_heads=4,
            num_rel=num_relations_mol,
            dropout=dropout,
            type="graph",
        )

        # Optional KG channel
        if not mol_only:
            self.drug_node_feature = NodeFeatures(
                degree=max_degree_node,
                feature_num=num_nodes,
                embedding_dim=output_dim,
                type="node",
            )
            self.node_representation_learning = GraphTransformer(
                layer_num=max_layer,
                embedding_dim=output_dim,
                num_heads=4,
                num_rel=num_relations_graph,
                dropout=dropout,
                type="node",
            )
            self.cold_start_proj = nn.Linear(output_dim, output_dim)

        # Shared fusion and classification head
        self.fc1 = nn.Sequential(
            nn.Linear(output_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(output_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2),
        )
        self.disc = Discriminator(output_dim)
        self.b_xent = BCEWithLogitsLoss()

    def to(self, device):  # noqa: D401 — override torch.nn.Module.to
        self.mol_atom_feature.to(device)
        self.mol_representation_learning.to(device)
        self.fc1.to(device)
        self.fc2.to(device)
        self.disc.to(device)
        self.b_xent.to(device)
        if not self.mol_only:
            self.drug_node_feature.to(device)
            self.node_representation_learning.to(device)
            self.cold_start_proj.to(device)
        self.device = str(device)
        return self

    def reset_parameters(self):
        self.mol_atom_feature.reset_parameters()
        self.mol_representation_learning.reset_parameters()
        if not self.mol_only:
            self.drug_node_feature.reset_parameters()
            self.node_representation_learning.reset_parameters()
            self.cold_start_proj.reset_parameters()

    # Cold-start patch (KG channel only)

    def _patch_unseen_center_nodes(
        self, node_feature, subgraph, mol_embedding, batch_idx, unseen_ids,
    ):
        """Replace unseen drugs' KG center rows in place.

        Use ``cold_start_proj(mol_embedding) + z_deg`` instead of their
        ID embeddings. Skip samples without a center marker.
        """
        if unseen_ids is None or len(unseen_ids) == 0:
            return
        dev = node_feature.device
        _, col = subgraph.edge_index[0], subgraph.edge_index[1]
        x_degree = degree(col, subgraph.x.size(0), dtype=subgraph.x.dtype)
        batch_size = mol_embedding.size(0)
        for i in range(batch_size):
            if batch_idx[i].item() not in unseen_ids:
                continue
            center_mask = (subgraph.batch == i) & subgraph.id.bool()
            if not center_mask.any():
                continue
            center_global_idx = center_mask.nonzero(as_tuple=True)[0][0]
            deg_val = x_degree[center_global_idx].long().clamp(
                0, self.max_degree_node - 1,
            )
            z_deg = self.drug_node_feature.degree_encoder(
                deg_val.unsqueeze(0)
            ).squeeze(0)
            node_feature[center_global_idx] = (
                self.cold_start_proj(mol_embedding[i]) + z_deg.to(dev)
            )

    # Forward

    def forward(
        self,
        drug1_mol,
        drug2_mol,
        drug1_subgraph=None,
        drug2_subgraph=None,
        batch_idx1=None,
        batch_idx2=None,
        unseen_ids=None,
        mask_channel: str | None = None,
    ):
        # Unlike upstream's interleaved argument order, mol inputs come first
        # so model(mol1, mol2) works in mol-only mode. Pass KG inputs by keyword.
        #
        # Fusion-time masks follow upstream
        # ``exps/sec5-3/2_indicators/baseline_mask_predictors/_tiger_runner_mask.py``
        # "mol" zeros molecular outputs (KPS-mol); "kg" zeros KG outputs
        # (KPS-KG), for both drugs before fc1. None leaves both unchanged.
        # Channel masks require dual-channel mode.
        if mask_channel not in (None, "mol", "kg"):
            raise ValueError(
                f"mask_channel must be one of {{None, 'mol', 'kg'}}; "
                f"got {mask_channel!r}"
            )
        if self.mol_only and mask_channel is not None:
            raise ValueError(
                "mask_channel requires dual-channel TIGER; "
                "this model was built with mol_only=True."
            )
        # Mol channel — always
        mol1_atom_feature = self.mol_atom_feature(drug1_mol)
        mol2_atom_feature = self.mol_atom_feature(drug2_mol)
        mol1_graph_emb, mol1_atom_emb, _ = self.mol_representation_learning(
            mol1_atom_feature, drug1_mol,
        )
        mol2_graph_emb, mol2_atom_emb, _ = self.mol_representation_learning(
            mol2_atom_feature, drug2_mol,
        )

        if self.mol_only:
            # Duplicate the molecular embedding to preserve fc1's output_dim*2 input.
            drug1_emb = self.fc1(torch.cat([mol1_graph_emb, mol1_graph_emb], dim=-1))
            drug2_emb = self.fc1(torch.cat([mol2_graph_emb, mol2_graph_emb], dim=-1))
            score = self.fc2(torch.cat([drug1_emb, drug2_emb], dim=-1))
            loss_s_m = (
                self.loss_MI(self.MI(drug1_emb, mol1_atom_emb))
                + self.loss_MI(self.MI(drug2_emb, mol2_atom_emb))
            )
            log_probs = F.log_softmax(score, dim=-1)
            loss_label = F.nll_loss(log_probs, drug1_mol.y.view(-1))
            loss = loss_label + self.mol_coeff * loss_s_m
            return torch.exp(log_probs)[:, 1], loss

        # Dual-channel path — requires subgraph inputs.
        if drug1_subgraph is None or drug2_subgraph is None:
            raise ValueError(
                "Dual-channel TIGER requires drug{1,2}_subgraph; got None. "
                "Pass them in or construct the model with mol_only=True."
            )
        drug1_node_feature = self.drug_node_feature(drug1_subgraph)
        drug2_node_feature = self.drug_node_feature(drug2_subgraph)

        # Cold-start patch.
        if unseen_ids is not None and len(unseen_ids) > 0 \
                and batch_idx1 is not None and batch_idx2 is not None:
            batch_idx1 = batch_idx1.to(drug1_node_feature.device)
            batch_idx2 = batch_idx2.to(drug2_node_feature.device)
            self._patch_unseen_center_nodes(
                drug1_node_feature, drug1_subgraph, mol1_graph_emb,
                batch_idx1, unseen_ids,
            )
            self._patch_unseen_center_nodes(
                drug2_node_feature, drug2_subgraph, mol2_graph_emb,
                batch_idx2, unseen_ids,
            )

        drug1_node_emb, drug1_sub_emb, _ = self.node_representation_learning(
            drug1_node_feature, drug1_subgraph,
        )
        drug2_node_emb, drug2_sub_emb, _ = self.node_representation_learning(
            drug2_node_feature, drug2_subgraph,
        )

        # Mask both drugs before fusion, as in upstream _tiger_runner_mask.py:480-485.
        if mask_channel == "mol":
            mol1_graph_emb = torch.zeros_like(mol1_graph_emb)
            mol2_graph_emb = torch.zeros_like(mol2_graph_emb)
        elif mask_channel == "kg":
            drug1_node_emb = torch.zeros_like(drug1_node_emb)
            drug2_node_emb = torch.zeros_like(drug2_node_emb)

        drug1_emb = self.fc1(torch.cat([drug1_node_emb, mol1_graph_emb], dim=-1))
        drug2_emb = self.fc1(torch.cat([drug2_node_emb, mol2_graph_emb], dim=-1))
        score = self.fc2(torch.cat([drug1_emb, drug2_emb], dim=-1))

        loss_s_m = (
            self.loss_MI(self.MI(drug1_emb, mol1_atom_emb))
            + self.loss_MI(self.MI(drug2_emb, mol2_atom_emb))
        )
        loss_s_d = (
            self.loss_MI(self.MI(drug1_emb, drug1_sub_emb))
            + self.loss_MI(self.MI(drug2_emb, drug2_sub_emb))
        )

        log_probs = F.log_softmax(score, dim=-1)
        loss_label = F.nll_loss(log_probs, drug1_mol.y.view(-1))
        loss = loss_label + self.mol_coeff * loss_s_m + self.mi_coeff * loss_s_d
        return torch.exp(log_probs)[:, 1], loss

    # Mutual-information loss

    def MI(self, graph_embeddings, sub_embeddings):
        idx = torch.arange(graph_embeddings.shape[0] - 1, -1, -1)
        if len(idx) > 1:
            mid = len(idx) // 2
            idx[mid] = idx[mid + 1] if mid + 1 < len(idx) else idx[mid]
        shuffle_embeddings = torch.index_select(
            graph_embeddings, 0, idx.to(graph_embeddings.device),
        )
        c_0_list, c_1_list = [], []
        for c_0, c_1, sub in zip(graph_embeddings, shuffle_embeddings, sub_embeddings):
            c_0_list.append(c_0.expand_as(sub))
            c_1_list.append(c_1.expand_as(sub))
        c_0 = torch.cat(c_0_list)
        c_1 = torch.cat(c_1_list)
        sub = torch.cat(sub_embeddings)
        return self.disc(sub, c_0, c_1)

    def loss_MI(self, logits):
        num_logits = logits.shape[0] // 2
        temp = torch.rand(num_logits)
        lbl = torch.cat(
            [torch.ones_like(temp), torch.zeros_like(temp)], dim=0,
        ).float().to(logits.device)
        return self.b_xent(logits.view([1, -1]), lbl.view([1, -1]))

    # Upstream save API

    def save(self, path):
        save_path = os.path.join(path, self.__class__.__name__ + ".pt")
        torch.save(self.state_dict(), save_path)
        return save_path
