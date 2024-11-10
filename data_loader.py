import pickle
from constants import BATCH_SIZE, CONTEXT_SIZE, DEVICE
import torch

class DataLoader:

    def __init__(self, data_file, state_file=None):

        # load the data
        data_tensor = torch.load(data_file, weights_only=True)
        n = int(0.9*len(data_tensor))
        self.state_dict = {
            'train': {
                'data': data_tensor[:n],
                'epochs': 0,
                'pointer': 0,
            },
            'val': {
                'data': data_tensor[n:],
                'epochs': 0,
                'pointer': 0,
            }
        }
       
        # Load the checkpoint if provided
        if state_file:
            with open(state_file, 'rb') as outp:
                checkpoint = pickle.load(outp)
                self.state_dict['train']['pointer'] = checkpoint['train_data_pointer']
                self.state_dict['train']['epochs'] = checkpoint['train_epochs']
                self.state_dict['val']['pointer'] = checkpoint['val_data_pointer']
                self.state_dict['val']['epochs'] = checkpoint['val_epochs']
            
    def get_next_batch(self, split):
        '''Gets the next batch of data'''
        data_tensor = self.state_dict[split]['data']
        x_batches = []
        y_batches = []
        for idx in range(BATCH_SIZE):
            pointer = self.state_dict[split]['pointer']
            x_batch = data_tensor[pointer : pointer + CONTEXT_SIZE]
            y_batch = data_tensor[pointer + 1 : pointer + CONTEXT_SIZE + 1]
            self.state_dict[split]['pointer'] += CONTEXT_SIZE
            if y_batch.shape[0] != CONTEXT_SIZE:
                y_batch = data_tensor[-CONTEXT_SIZE:]
                x_batch = data_tensor[-CONTEXT_SIZE-1:-1]
                self.state_dict[split]['pointer'] = 0
                self.state_dict[split]['epochs'] += 1
            x_batches.append(x_batch)
            y_batches.append(y_batch)
        xb = torch.stack(x_batches).to(DEVICE)
        yb = torch.stack(y_batches).to(DEVICE)
        return xb, yb

    def save_state(self, filename):
        '''Saves current training state to disk to continue.'''
        checkpoint = {
            'train_data_pointer': self.state_dict['train']['pointer'],
            'train_epochs': self.state_dict['train']['epochs'],
            'val_data_pointer': self.state_dict['val']['pointer'],
            'val_epochs': self.state_dict['val']['epochs']
        }
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(checkpoint, outp, pickle.HIGHEST_PROTOCOL)
        

