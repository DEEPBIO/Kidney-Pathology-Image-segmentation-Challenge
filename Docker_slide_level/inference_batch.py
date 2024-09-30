import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm



def prepare_batches(patches, batch_size=128):
    tensors, coords = zip(*patches)
    dataset = TensorDataset(torch.stack(tensors))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), coords

def batch_inference(model, patches, batch_size, device='cuda'):
    model.eval()
    dataloader, coords = prepare_batches(patches, batch_size)
    all_outputs = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch[0].to(device)
            outputs = model(inputs)
            all_outputs.extend(outputs.cpu())
    return all_outputs, coords

