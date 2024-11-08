import pickle
from constants import BATCH_SIZE, CONTEXT_SIZE
import torch

class DataLoader:

    def __init__(self, data_file, state_file=None):
        self.data_tensor = torch.load(data_file)
        self.data_pointer = 0
        self.epochs = 0
        # Load the checkpoint if provided
        if state_file:
            with open(state_file, 'rb') as outp:
                checkpoint = pickle.load(outp)
                self.data_pointer = checkpoint['data_pointer']
                self.epochs = checkpoint['epochs']

            
    def get_next_batch(self):
        '''Gets the next batch of data'''
        batches = []
        for idx in range(BATCH_SIZE):
            batch = self.data_tensor[self.data_pointer : self.data_pointer + CONTEXT_SIZE]
            self.data_pointer += CONTEXT_SIZE
            if batch.shape[0] != CONTEXT_SIZE:
                batch = self.data_tensor[-CONTEXT_SIZE:]
                self.data_pointer = 0
                self.epochs += 1
            batches.append(batch)
        return torch.stack(batches)

    def save_state(self, filename):
        '''Saves current training state to disk to continue.'''
        checkpoint = {'data_pointer':self.data_pointer, 'epochs':self.epochs}
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(checkpoint, outp, pickle.HIGHEST_PROTOCOL)
        

