import os

from graph_dataset import GraphDataset
from torch_geometric.data import DataLoader
from graph_model import GraphNet
import torch

from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    dataset = GraphDataset(root='d:/Work/research/data/hammer', model_name='hollow_1')
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    test_dataset = GraphDataset(root='d:/Work/research/data/hammer', model_name='hollow_2')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = GraphNet(input_dim_node=1, input_dim_element=2, hidden_dim=8, output_dim=1, num_layers=5, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # model.train()

    logdir = 'd:/Work/research/data/hammer/logs'
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    for epoch in range(100):
        model.train()
        avg_loss = 0
        for batch in loader:
            # use R2 as accuracy
            optimizer.zero_grad()
            out = model(batch)
            loss = torch.nn.functional.mse_loss(out, batch.y)
            avg_loss += loss.item()
            # accuracy = 1 - loss
            loss.backward()
            optimizer.step()

        avg_loss /= len(loader)
        print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, avg_loss))
        writer.add_scalar('Loss/train', avg_loss, epoch)

        if epoch % 10 == 0:
            avg_loss = 0
            model.eval()
            for batch in test_loader:
                out = model(batch)
                loss = torch.nn.functional.mse_loss(out, batch.y)
                avg_loss += loss.item()
            avg_loss /= len(test_loader)
            print('Epoch: {:03d}, Test Loss: {:.5f}'.format(epoch, avg_loss))
            writer.add_scalar('Loss/test', avg_loss, epoch)
            torch.save(model.state_dict(), os.path.join(logdir, 'model_{}.pth'.format(epoch)))  

    writer.close()



