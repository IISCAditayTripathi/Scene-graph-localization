from typing import Union, Tuple, Optional, Callable
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor,Size)
# from torch_geometric.typing import PairTensor, Adj

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Linear, Sigmoid, BatchNorm1d
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_sparse import SparseTensor, set_diag
import torch.nn.functional as F
from torch_geometric.nn import MetaLayer
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch import nn

# from ..inits import zeros

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class ResGatedGraphConv(MessagePassing):
    r"""The residual gated graph convolutional operator from the
    `"Residual Gated Graph ConvNets" <https://arxiv.org/abs/1711.07553>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \eta_{i,j} \odot \mathbf{W}_2 \mathbf{x}_j

    where the gate :math:`\eta_{i,j}` is defined as

    .. math::
        \eta_{i,j} = \sigma(\mathbf{W}_3 \mathbf{x}_i + \mathbf{W}_4
        \mathbf{x}_j)

    with :math:`\sigma` denoting the sigmoid function.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        act (callable, optional): Gating function :math:`\sigma`.
            (default: :meth:`torch.nn.Sigmoid()`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        act: Optional[Callable] = Sigmoid(),
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):

        kwargs.setdefault('aggr', 'add')
        super(ResGatedGraphConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[1], out_channels)
        self.lin_query = Linear(in_channels[0], out_channels)
        self.lin_value = Linear(in_channels[0], out_channels)

        if root_weight:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=False)
        else:
            self.register_parameter('lin_skip', None)

        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        k = self.lin_key(x[1])
        q = self.lin_query(x[0])
        v = self.lin_value(x[0])

        # propagate_type: (k: Tensor, q: Tensor, v: Tensor)
        out = self.propagate(edge_index, k=k, q=q, v=v, size=None)

        if self.root_weight:
            out += self.lin_skip(x[1])

        if self.bias is not None:
            out += self.bias

        return out


    def message(self, k_i: Tensor, q_j: Tensor, v_j: Tensor) -> Tensor:
        print(self.act(k_i + q_j))
        return self.act(k_i + q_j) * v_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class GATv2Conv(MessagePassing):
    r"""The GATv2 operator from the `"How Attentive are Graph Attention Networks?"
    <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the static
    attention problem of the standard :class:`~torch_geometric.conv.GATConv`
    layer: since the linear layers in the standard GAT are applied right after
    each other, the ranking of attended nodes is unconditioned on the query
    node. In contrast, in GATv2, every node can attend to any other node.

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: int,
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True,
                 share_weights: bool = False,
                 **kwargs):
        super(GATv2Conv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights

        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att)
        zeros(self.bias)


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights: bool = None):
        # type: (Union[Tensor, PairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: PairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, x_j: Tensor, x_i: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class CGConv(MessagePassing):
    r"""The crystal graph convolutional operator from the
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)}
        \sigma \left( \mathbf{z}_{i,j} \mathbf{W}_f + \mathbf{b}_f \right)
        \odot g \left( \mathbf{z}_{i,j} \mathbf{W}_s + \mathbf{b}_s  \right)

    where :math:`\mathbf{z}_{i,j} = [ \mathbf{x}_i, \mathbf{x}_j,
    \mathbf{e}_{i,j} ]` denotes the concatenation of central node features,
    neighboring node features and edge features.
    In addition, :math:`\sigma` and :math:`g` denote the sigmoid and softplus
    functions, respectively.

    Args:
        channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        dim (int, optional): Edge feature dimensionality. (default: :obj:`0`)
        aggr (string, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        batch_norm (bool, optional): If set to :obj:`True`, will make use of
            batch normalization. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0,
                 aggr: str = 'add', batch_norm: bool = False,
                 bias: bool = True, **kwargs):
        super(CGConv, self).__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim
        self.batch_norm = batch_norm

        if isinstance(channels, int):
            channels = (channels, channels)

        self.lin_f = Linear(sum(channels) + dim, channels[1], bias=bias)
        self.lin_s = Linear(sum(channels) + dim, channels[1], bias=bias)
        self.bn = BatchNorm1d(channels[1])

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        self.bn.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.bn(out) if self.batch_norm else out
        out += x[1]
        return out


    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

    def __repr__(self):
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels,
                                       self.dim)

class EdgeModel(torch.nn.Module):
    def __init__(self,  div=1, in_dim=1024):
        super(EdgeModel, self).__init__()
        # self.edge_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
        batch_norm = torch.nn.BatchNorm1d
        dropout = torch.nn.Dropout
        # self.edge_mlp = Seq(Lin(3*1024, 1024), dropout(0.2), ReLU(), Lin(1024, 1024))
        # self.edge_mlp = Seq(Lin(3*1024, 1024), batch_norm(1024), ReLU(), Lin(1024, 1024))
        self.edge_mlp = Seq(Lin(3*(in_dim//div), in_dim//div), ReLU(), Lin(in_dim//div, in_dim//div))
    def forward(self, src, dest, edge_attr, u=None, batch=None):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self, div=1, in_dim=1024):
        super(NodeModel, self).__init__()
        batch_norm = torch.nn.BatchNorm1d
        dropout = torch.nn.Dropout
        # self.node_mlp_1 = Seq(Lin(2*1024, 1024), dropout(0.2), ReLU(), Lin(1024, 1024))
        # self.node_mlp_2 = Seq(Lin(2*1024, 1024), dropout(0.2), ReLU(), Lin(1024, 1024))

        # self.node_mlp_1 = Seq(Lin(2*1024, 1024), batch_norm(1024), ReLU(), Lin(1024, 1024))
        # self.node_mlp_2 = Seq(Lin(2*1024, 1024), batch_norm(1024), ReLU(), Lin(1024, 1024))

        self.node_mlp_1 = Seq(Lin(2*(in_dim//div), in_dim//div), ReLU(), Lin(in_dim//div, in_dim//div))
        self.node_mlp_2 = Seq(Lin(2*(in_dim//div), in_dim//div), ReLU(), Lin(in_dim//div, in_dim//div))
        self.dropout = 0.4

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        # print(out.shape)
        out = self.node_mlp_1(out)
        
        # out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        mean = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        # add = scatter_add(out, col, dim=0, dim_size=x.size(0))
        
        max = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]

        out = torch.cat([x, mean], dim=1)
        # out = out + x
        # return 0.7*self.node_mlp_2(out) + 0.3*x
        # return 0.6*self.node_mlp_2(out) + 0.4*x
        return self.node_mlp_2(out) 

class cross_modal_attention(nn.Module):
    def __init__(self, inplanes) -> None:
        super(cross_modal_attention,self).__init__()
        bn_1d = nn.BatchNorm1d
        # conv_nd = nn.Conv2d
        self.in_channels = inplanes
        self.maxpool_2d = nn.MaxPool2d(self.in_channels)

        bn = nn.BatchNorm2d
        
        
        self.theta_1 = nn.Sequential(nn.Linear(self.in_channels, self.in_channels//2),
                                        bn_1d(self.in_channels//2),
                                        nn.ReLU(),
                                        nn.Linear(self.in_channels//2, self.in_channels//2, bias=True))

        self.theta_2 = nn.Sequential(nn.Linear(self.in_channels, self.in_channels//2),
                                        bn_1d(self.in_channels//2),
                                        nn.ReLU(),
                                        nn.Linear(self.in_channels//2, self.in_channels//2, bias=True))


        self.softmax = torch.nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.theta_1[0].weight)
        nn.init.xavier_uniform_(self.theta_2[0].weight)
        # nn.init.xavier_uniform_(self.op[0].weight)

    def attention_util(self,image, query, is_GM=False):
        attention_feats = torch.bmm(query.unsqueeze(1), image.permute(0,2,1))
        return attention_feats

    def forward(self, image_feats, text_vector, num_props, is_GM=False):
        # print(image_feats.shape)
        img_feats = self.theta_1(image_feats.view(-1, 1024))
        # batch_size, n_channels, n_dim = img_feats.shape
        n_channels=num_props
        batch_size = text_vector.shape[0]
        
        text_vector = self.theta_2(text_vector.squeeze(1))
        img_feats = img_feats.view(batch_size, n_channels, -1)
        attention = self.attention_util(img_feats, text_vector, is_GM)

        attention = attention.view(batch_size, n_channels) # 64 before
        # attention = attention
        
        
        # image_feats = image_feats*attention.unsqueeze(-1)

        return attention

class MyNodeModel(torch.nn.Module):
    def __init__(self):
        super(MyNodeModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(2*1024, 1024), ReLU(), Lin(1024, 1024))
        self.node_mlp_2 = Seq(Lin(2*1024, 1024), ReLU(), Lin(1024, 1024))
        self.attention = cross_modal_attention(inplanes=1024)

    def forward(self, x, edge_index, edge_attr, u=None, batch=None, queries=None, num_props=None):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        
        att_1 = self.attention(x.view(queries[0].shape[0], num_props, 1024), queries[0], num_props).view(-1)
        att_2 = self.attention(x.view(queries[0].shape[0], num_props, 1024), queries[1], num_props).view(-1)
        
        # attention = (att_1+att_2)/2
        
        attention = att_1 + att_2
        
        # attention = torch.cat([att_1.unsqueeze(-1), att_2.unsqueeze(-1)], dim=-1)
        # attention,_ = torch.max(attention, dim=-1)
        # attention = torch.nn.functional.relu(attention) + 0.001
        attention = torch.nn.functional.tanh(attention)
        # attention = torch.nn.functional.selu(attention)
        
        row, col = edge_index
        attention = attention[col]
        
        out = torch.cat([x[row], edge_attr], dim=1)
        
        
        
        out = self.node_mlp_1(out)
        out_atten = out*attention.unsqueeze(-1)
        # out = out+out_atten
        out = out_atten     
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)

class MyMetaLayer(torch.nn.Module):
    r"""A meta layer for building any kind of graph network, inspired by the
    `"Relational Inductive Biases, Deep Learning, and Graph Networks"
    <https://arxiv.org/abs/1806.01261>`_ paper.

    A graph network takes a graph as input and returns an updated graph as
    output (with same connectivity).
    The input graph has node features :obj:`x`, edge features :obj:`edge_attr`
    as well as global-level features :obj:`u`.
    The output graph has the same structure, but updated features.

    Edge features, node features as well as global features are updated by
    calling the modules :obj:`edge_model`, :obj:`node_model` and
    :obj:`global_model`, respectively.

    To allow for batch-wise graph processing, all callable functions take an
    additional argument :obj:`batch`, which determines the assignment of
    edges or nodes to their specific graphs.

    Args:
        edge_model (Module, optional): A callable which updates a graph's edge
            features based on its source and target node features, its current
            edge features and its global features. (default: :obj:`None`)
        node_model (Module, optional): A callable which updates a graph's node
            features based on its current node features, its graph
            connectivity, its edge features and its global features.
            (default: :obj:`None`)
        global_model (Module, optional): A callable which updates a graph's
            global features based on its node features, its graph connectivity,
            its edge features and its current global features.

    .. code-block:: python

        from torch.nn import Sequential as Seq, Linear as Lin, ReLU
        from torch_scatter import scatter_mean
        from torch_geometric.nn import MetaLayer

        class EdgeModel(torch.nn.Module):
            def __init__(self):
                super(EdgeModel, self).__init__()
                self.edge_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

            def forward(self, src, dest, edge_attr, u, batch):
                # src, dest: [E, F_x], where E is the number of edges.
                # edge_attr: [E, F_e]
                # u: [B, F_u], where B is the number of graphs.
                # batch: [E] with max entry B - 1.
                out = torch.cat([src, dest, edge_attr, u[batch]], 1)
                return self.edge_mlp(out)

        class NodeModel(torch.nn.Module):
            def __init__(self):
                super(NodeModel, self).__init__()
                self.node_mlp_1 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
                self.node_mlp_2 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

            def forward(self, x, edge_index, edge_attr, u, batch):
                # x: [N, F_x], where N is the number of nodes.
                # edge_index: [2, E] with max entry N - 1.
                # edge_attr: [E, F_e]
                # u: [B, F_u]
                # batch: [N] with max entry B - 1.
                row, col = edge_index
                out = torch.cat([x[row], edge_attr], dim=1)
                out = self.node_mlp_1(out)
                out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
                out = torch.cat([x, out, u[batch]], dim=1)
                return self.node_mlp_2(out)

        class GlobalModel(torch.nn.Module):
            def __init__(self):
                super(GlobalModel, self).__init__()
                self.global_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

            def forward(self, x, edge_index, edge_attr, u, batch):
                # x: [N, F_x], where N is the number of nodes.
                # edge_index: [2, E] with max entry N - 1.
                # edge_attr: [E, F_e]
                # u: [B, F_u]
                # batch: [N] with max entry B - 1.
                out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
                return self.global_mlp(out)

        op = MetaLayer(EdgeModel(), NodeModel(), GlobalModel())
        x, edge_attr, u = op(x, edge_index, edge_attr, u, batch)
    """
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super(MyMetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()


    def forward(
            self, x: Tensor, edge_index: Tensor,
            edge_attr: Optional[Tensor] = None, u: Optional[Tensor] = None,
            batch: Optional[Tensor] = None, queries: Optional[Tensor] = None, num_props: Optional[Tensor] = None,) -> Tuple[Tensor, Tensor, Tensor]:
        """"""
        row = edge_index[0]
        col = edge_index[1]

        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr, u,
                                        batch if batch is None else batch[row])

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, batch, queries=queries, num_props=num_props)

        if self.global_model is not None:
            u = self.global_model(x, edge_index, edge_attr, u, batch)

        return x, edge_attr, u


    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                '    global_model={}\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model, self.global_model)

# x, edge_attr, u = op(x, edge_index, edge_attr, u, batch)