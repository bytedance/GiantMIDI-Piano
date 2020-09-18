"""This piano solo detection module is trained by Bochen Li in Feb. 2020, and 
then is cleaned up by Qiuqiang Kong in Jul. 2020.
"""
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Hyper-parameters
SR = 32000
FRAME_LEN = 2048

if SR == 32000:
    FRAME_HOP = 500

CH_NUM = 1
USE_DB = False
OFFSET = 1.0

N_FFT = FRAME_LEN
WIN = np.sqrt(np.hanning(N_FFT))
DIM_F = int(FRAME_LEN / 2)
DIM_F = 256
DIM_T = 64
DIM_T_HOP = 64


def read_audio_stereo(filename):
    wav, _ = librosa.core.load(filename, sr=SR, mono=None)
    if wav.ndim == 1:
        wav = np.tile(wav[..., None], (1,2))
    else:
        wav = wav.T
    return wav

    
def wav2spec_mono(wav):
    spec = librosa.core.stft(y=wav, n_fft=N_FFT, 
                             hop_length=FRAME_HOP, win_length=FRAME_LEN,
                             window=WIN, center='True', pad_mode='constant')
    mag, pha = librosa.core.magphase(spec)
    if USE_DB:
        mag = librosa.core.amplitude_to_db(S=(mag+OFFSET))
    ang = np.angle(pha)
    mag = mag[:DIM_F, :]
    ang = ang[:DIM_F, :]
    mag = mag[None, ...]
    ang = ang[None, ...]
    return mag, ang

def spec2wav_mono(mag, ang):
    if USE_DB:
        mag = librosa.core.db_to_amplitude(S_db=mag) - OFFSET
    pha = np.exp(1j * ang)
    spec = mag * pha
    if DIM_F % 2 == 0:
        tmp = np.zeros((1, spec.shape[-1]))
        spec = np.concatenate((spec, tmp), axis=0)

    wav = librosa.core.istft(stft_matrix=spec, 
                             hop_length=FRAME_HOP,
                             win_length=FRAME_LEN,
                             window=WIN, center='True')
    return wav


def wav2spec(wav):
    """
    input:  mono shape=(n,) or stereo shape=(n,2)
    output: mag, ang
            mono shape=(1,F,T) or stereo shape=(2,F,T)
    """
    
    if wav.ndim == 1:
        
        mag, ang = wav2spec_mono(wav)
        
    else:
        
        mag1, ang1 = wav2spec_mono(wav[:, 0])
        mag2, ang2 = wav2spec_mono(wav[:, 1])
        mag = np.concatenate((mag1, mag2), axis=0)
        ang = np.concatenate((ang1, ang2), axis=0)
    
    return mag, ang

def spec2wav(mag, ang):
    if mag.shape[0] == 1:
        mag = mag[0,...]
        ang = ang[0,...]
        wav = spec2wav_mono(mag, ang)
    else:
        wav1 = spec2wav_mono(mag[0,...], ang[0,...])
        wav2 = spec2wav_mono(mag[1,...], ang[1,...])
        wav = np.concatenate( (wav1[...,None], wav2[...,None]), axis=-1 )
        
    return wav


class ConvBlock(nn.Module):
    def __init__(self, in_plane, out_plane, droprate=0.0):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_plane, out_plane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_plane)
        self.relu = nn.ReLU(inplace=True)
        
        self.droprate = droprate
        
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return out
    

class PianoSoloDetector(object):
    def __init__(self):
        """Piano solo detector."""
        self.model = PianoDetection()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model.load('resources/piano_solo_model_32k.pth')

    def predict(self, wav):
        """Predict the probabilities of piano solo on 1-second segments.
        """
        rms = np.sqrt(np.mean(wav ** 2))
        wav = wav / rms / 20
        duration = len(wav) / SR

        n_seg = int(duration / 1.00)
        
        mag_segs = []
        batch_size = 32

        all_probs = []
        zero_locts = []

        for i in np.arange(n_seg):
            wav_seg = wav[i * SR : (i + 1) * SR + 1000]
            
            if np.sqrt(np.mean(wav_seg**2)) < 0.001:
                zero_locts.append(i)
            
            mag, ang = wav2spec(wav_seg)
            mag = mag[..., :DIM_T]
            
            mag_segs.append(mag)

            if len(mag_segs) == batch_size or i == n_seg - 1:
                probs = self.predict_seg(np.array(mag_segs))
                all_probs.append(probs)
                mag_segs = []

        all_probs = np.concatenate(all_probs)
        zero_locts = np.array(zero_locts)
        
        if len(zero_locts) > 0:
            all_probs[zero_locts] = 0

        return all_probs

    def predict_seg(self, mag_seg):
        """Predict the probability of piano solo on each segment.

        Args:
          mag_seg: (batch_size, 1, F, T)

        Returns:
          probs: (batch_size,)
        """
        x = np.transpose(mag_seg, (0, 1, 3, 2))
        y = self.model.predict_on_batch(x)  # (batch_size, classes_num)
        probs = y[:, 1]
        return probs


class PianoDetection(nn.Module):
    def __init__(self):
        super(PianoDetection, self).__init__()
        
        self.net = CNN()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
        if torch.cuda.is_available():
            self.criterion = nn.CrossEntropyLoss().cuda()
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.net(x)
        return x
    
    def _convert(self, x):
        x_var = []
        x_var = Variable(torch.FloatTensor(x))

        if torch.cuda.is_available():
            x_var = x_var.cuda()
        
        return x_var
    
    def _convert_int(self, x):
        x_var = []
        x_var = Variable(torch.LongTensor(x))

        if torch.cuda.is_available():
            x_var = x_var.cuda()
        
        return x_var

    def train_on_batch(self, x, t):
        self.train()
        x = self._convert(x)
        t = self._convert_int(t)
        y = self.forward(x=x)
        self.optimizer.zero_grad()
        loss = self.criterion(y, t)
        loss.backward()
        self.optimizer.step()
        return loss.data.cpu().numpy()
    
    
    def eval_on_batch(self, x, t):
        self.eval()
        x = self._convert(x)
        t = self._convert_int(t)
        y = self.forward(x)
        loss = self.criterion(y, t)
        return loss.data.cpu().numpy()
    
    def predict_on_batch(self, x):
        self.eval()
        x = self._convert(x)
        y = self.forward(x)
        y = F.softmax(y, dim=1)
        return y.data.cpu().numpy()
    
    def adjust_learning_rate(self, epoch):
        lr = self.lr * (0.8 ** np.floor(epoch / 5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def save(self, filename):
        torch.save(self.state_dict(), filename+".pth")
    
    def load(self, filename):
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(filename))
        else:
            self.load_state_dict(torch.load(filename, map_location='cpu'))

    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.cnn1 = ConvBlock(1, 32)
        self.cnn2 = ConvBlock(32, 64)
        self.cnn3 = ConvBlock(64, 64)
        self.cnn4 = ConvBlock(64, 32)
        self.fn1 = nn.Linear(2048, 50)
        self.fn2 = nn.Linear(50, 2)
        
    def forward(self, x):
        x = self.cnn1(x)
        x = F.avg_pool2d(x, 2)
        
        x = self.cnn2(x)
        x = F.avg_pool2d(x, 2)
        
        x = self.cnn3(x)
        x = F.avg_pool2d(x, 2)
        
        x = self.cnn4(x)
        x = F.avg_pool2d(x, 2)
        
        x_dim = x.shape[1] * x.shape[2] * x.shape[3]
        x = x.view(-1, x_dim)
        
        x = self.fn1(x)
        x = F.relu(x)
        
        x = self.fn2(x)
        
        return x