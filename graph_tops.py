import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(in_channels, 128, heads=8, dropout=0.6) 
        self.conv2 = GATv2Conv(128 * 128, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, input_ids,
            position_ids,
            token_type_ids,
            inputs_embeds,
            past_key_values_length):
        x, edge_index = input_ids[0], input_ids[1]
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class SageGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SageGAT, self).__init__()
        self.conv1 = SAGEConv(in_channels, 128) 
        self.conv2 = SAGEConv(128 * 128, out_channels )

    def forward(self, input_ids,
            position_ids,
            token_type_ids,
            inputs_embeds,
            past_key_values_length):
        x, edge_index = input_ids[0], input_ids[1]
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)