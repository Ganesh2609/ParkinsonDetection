import os
import torch
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import Wav2Vec2Processor


class Wav2Vec2PretrainDataset(Dataset):

    def __init__(self, csv_path:str, root:str, age_filter:list=None, sampling_rate:int=16000, max_length:int=41216):
        self.df = pd.read_csv(csv_path)
        if age_filter:
            self.df = self.df[self.df['age'].isin(age_filter)]
        self.audio_files = [os.path.join(root, filename) for filename in self.df['filename']]
        self.max_length = max_length
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.audio_files)
    
    def collate_fn(batch, processor):
        max_len = max(item['Input'].shape[0] for item in batch)
        input_values = []
        attention_masks = []
        for item in batch:
            input_values.append(item['Input'])
            attention_masks.append(item['Mask'])   
        input_values = processor.pad({"input_values": input_values}, padding="max_length", max_length=max_len, return_tensors="pt")
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        return {
            'Input': input_values.input_values,
            'Mask': attention_masks
        }

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        audio_path = self.audio_files[idx]
        
        waveform, sr = torchaudio.load(audio_path)

        if sr != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)(waveform)

        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)

        if waveform.shape[0] > self.max_length:
            start = torch.randint(0, waveform.shape[0] - self.max_length, (1,)).item()
            waveform = waveform[start: start + self.max_length]
            attn_mask = torch.ones(self.max_length, dtype=torch.long)
        else:
            pad_len = self.max_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            attn_mask = torch.cat([torch.ones(waveform.shape[0] - pad_len, dtype=torch.long), 
                                   torch.zeros(pad_len, dtype=torch.long)])

        input_values = self.processor(waveform, sampling_rate=self.sampling_rate, return_tensors="pt").input_values
        return {
            'Input': input_values.squeeze(0), 
            'Mask': attn_mask
        }


def get_data_loaders(csv_path, root1, root2=None, batch_size:int=32, num_workers:int=12, prefetch_factor:int=2):
    
    age_filter = [i for i in range(55, 90)]

    dataset1 = Wav2Vec2PretrainDataset(csv_path=csv_path, root=root1, age_filter=age_filter)
    dataset2 = Wav2Vec2PretrainDataset(csv_path=csv_path, root=root2, age_filter=age_filter)
    train_dataset = ConcatDataset([dataset1, dataset2])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True, 
        prefetch_factor=prefetch_factor,
        collate_fn=lambda batch: Wav2Vec2PretrainDataset.collate_fn(batch, dataset1.processor)
    )

    return train_loader




# import os
# import torch
# import torchaudio
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
# from transformers import Wav2Vec2Processor


# class Wav2Vec2PretrainDataset(Dataset):

#     def __init__(self, root:str, sampling_rate:int=16000, max_length:int=41216):
#         self.audio_files = [os.path.join(root, i) for i in os.listdir(root)]
#         self.max_length = max_length
#         self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
#         self.sampling_rate = sampling_rate

#     def __len__(self):
#         return len(self.audio_files)
    
#     def collate_fn(batch, processor):
#         max_len = max(wav.shape[1] for wav in batch)
#         inputs = processor([wav.numpy() for wav in batch], sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=max_len)
#         return inputs.input_values 

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.item()
#         audio_path = self.audio_files[idx]
#         waveform, sr = torchaudio.load(audio_path)

#         if sr != self.sampling_rate:
#             waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)(waveform)

#         waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)

#         if waveform.shape[0] > self.max_length:
#             start = torch.randint(0, waveform.shape[0] - self.max_length, (1,)).item()
#             waveform = waveform[start: start + self.max_length]
#             attn_mask = torch.ones(self.max_length, dtype=torch.long)
#         else:
#             pad_len = self.max_length - waveform.shape[0]
#             waveform = torch.nn.functional.pad(waveform, (0, pad_len))
#             attn_mask = torch.cat([torch.ones(waveform.shape[0] - pad_len), torch.zeros(pad_len)]).long()

#         input_values = self.processor(waveform, sampling_rate=self.sampling_rate, return_tensors="pt").input_values
#         return {
#             'Input' : input_values.squeeze(0), 
#             #'Input' : input_values.squeeze(0).type(torch.float16), 
#             'Mask' : attn_mask
#         }




# def get_data_loaders(root1, root2, batch_size:int=32, num_workers:int=12, prefetch_factor:int=2):

#     dataset1 = Wav2Vec2PretrainDataset(root=root1)
#     dataset2 = Wav2Vec2PretrainDataset(root=root2)
#     train_dataset = ConcatDataset([dataset1, dataset2])

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)

#     return train_loader




# import os
# import torch
# import torchaudio
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
# from transformers import Wav2Vec2Processor


# class Wav2Vec2PretrainDataset(Dataset):

#     def __init__(self, root:str, sampling_rate:int=16000):
#         self.audio_files = [os.path.join(root, i) for i in os.listdir(root)]
#         print(self.audio_files)
#         self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
#         self.sampling_rate = sampling_rate

#     def __len__(self):
#         return len(self.audio_files)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.item()
#         audio_path = self.audio_files[idx]
#         waveform, sr = torchaudio.load(audio_path)
#         if sr != self.sampling_rate:
#             waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)(waveform)
#         return waveform.squeeze(0)


# def collate_fn(batch, processor):
#     max_len = max(wav.shape[0] for wav in batch)
#     inputs = processor([wav.numpy() for wav in batch], sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=max_len)
#     return inputs.input_values 

# def get_data_loaders(root1, root2, batch_size:int=32, num_workers:int=12, prefetch_factor:int=2):

#     dataset1 = Wav2Vec2PretrainDataset(root=root1)
#     dataset2 = Wav2Vec2PretrainDataset(root=root2)
#     train_dataset = ConcatDataset([dataset1, dataset2])

#     processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, processor) )

#     return train_loader
