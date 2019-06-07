from torch.utils.data import DataLoader
from torch import nn, optim
from fcrn import FCRN
import torch
from utils import MetricLogger
import time
import datetime
from nyu_depth import NYUDepth

if __name__ == '__main__':
    path = 'data/NYU_DEPTH'
    batch_size = 16
    epochs = 50
    device = torch.device('cuda:0')
    print_every = 1
    
    # data loader
    dataset = NYUDepth(path, 'train')
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4)
    
    
    # load model
    model = FCRN(False)
    model = model.to(device)
    
    # train
    model.train()
    
    # optimizer
    optimizer = optim.Adam(model.parameters())
    
    criterion = nn.MSELoss()
    # training
    metric_logger = MetricLogger()
    
    end = time.perf_counter()
    max_iters = epochs * len(dataloader)
    
    print('Start training')
    for epoch in range(epochs):
        start = end
        for i, data in enumerate(dataloader):
            i += 1
            data = [x.to(device) for x in data]
            image, depth, cloud = data
            pred = model(image)
            loss = criterion(pred, depth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # bookkeeping
            end = time.perf_counter()
            metric_logger.update(loss=loss.item())
            metric_logger.update(batch_time=end-start)
            
            if i % print_every == 0:
                # Compute eta. global step: starting from 1
                global_step = epoch * len(dataset) + i
                seconds = (max_iters - global_step) * metric_logger['batch_time'].global_avg
                eta = datetime.timedelta(seconds=seconds)
                # to display: eta, epoch, iteration, loss, batch_time
                display_dict = {
                    'eta': eta,
                    'epoch': epoch,
                    'iter': i,
                    'loss': metric_logger['loss'].median,
                    'batch_time': metric_logger['batch_time'].median
                }
                display_str = []
                for key in display_dict:
                    display_str.append('{}: {}'.format(key, display_dict[key]))
                    
                print(', '.join(display_str))
            
        
    
    
