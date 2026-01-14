from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import torch
from torch import nn
from utils import *
import torch.nn.functional as F
import constriction
import numpy as np

def grad_scale(x, scale):
    return (x - x * scale).detach() + x * scale

def ste(x):
    return (x.round() - x).detach() + x

class FakeQuantizationHalf(torch.autograd.Function):
    """performs fake quantization for half precision"""

    @staticmethod
    def forward(_, x):
        return x.half().float()

    @staticmethod
    def backward(_, grad_output):
        return grad_output

class FakeQuantizationFP12(nn.Module):

    def __init__(self):
        super().__init__()
        self.E = 5
        self.M = 6
        self.bias = (1 << (self.E - 1)) - 1  # 15


    def _quantize_to_float(self, x: torch.Tensor) -> torch.Tensor:
        E = self.E
        M = self.M
        bias = self.bias
        if not torch.is_floating_point(x):
            x = x.float()
        absx = x.abs()
        zero_mask = absx == 0
        mant, exp = torch.frexp(absx)
        mant = mant * 2.0
        exp = exp - 1
        exp_min = 1 - bias
        exp_max = bias
        exp_clamped = exp.clamp(exp_min, exp_max)
        step = float(1 << M)
        mant_code = torch.round((mant - 1.0) * step)  # 0..2^M-1
        mant_code = mant_code.clamp(0, step - 1)
        mant_q = 1.0 + mant_code / step
        sign = torch.sign(x) 
        x_q = torch.ldexp(mant_q, exp_clamped) * sign
        x_q = torch.where(zero_mask, torch.zeros_like(x_q), x_q)

        return x_q


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self._quantize_to_float(x)
        return x + (x_q - x).detach()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.IntTensor:
        E = self.E
        M = self.M
        bias = self.bias
        if not torch.is_floating_point(x):
            x = x.float()
        absx = x.abs()
        zero_mask = absx == 0
        mant, exp = torch.frexp(absx)
        mant = mant * 2.0
        exp = exp - 1
        exp_min = 1 - bias
        exp_max = bias
        exp_clamped = exp.clamp(exp_min, exp_max)
        step = float(1 << M)      
        mant_code = torch.round((mant - 1.0) * step) 
        mant_code = mant_code.clamp(0, step - 1).to(torch.int32)
        exp_code = (exp_clamped + bias).to(torch.int32)
        exp_code = exp_code.clamp(0, (1 << E) - 1)
        sign_bit = (x < 0).to(torch.int32)
        code = (sign_bit << 11) | (exp_code << 6) | mant_code
        code = torch.where(zero_mask, torch.zeros_like(code), code)

        return code.to(torch.int16)

    @torch.no_grad()
    def decode(self, code: torch.IntTensor) -> torch.Tensor:
        E = self.E
        M = self.M
        bias = self.bias
        step = float(1 << M)
        code = code.to(torch.int32)
        zero_mask = (code == 0)
        sign_bit = (code >> 11) & 0x1       # 1 bit
        exp_code = (code >> 6) & 0x1F       # 5 bit
        mant_code = code & 0x3F             # 6 bit
        ones = torch.ones_like(code, dtype=torch.float32)
        sign = torch.where(sign_bit == 0, ones, -ones)
        exp_unbiased = exp_code.to(torch.int32) - bias
        mant_q = 1.0 + mant_code.to(torch.float32) / step
        x = torch.ldexp(mant_q, exp_unbiased) * sign
        x = torch.where(zero_mask,
                        torch.zeros_like(x, dtype=torch.float32),
                        x)
        return x


class UniformQuantizer(nn.Module):
    def __init__(self, signed=False, bits=8, learned=False, num_channels=1, entropy_type="none", weight=0.001):
        super().__init__()
        if signed:
            self.qmin = -2**(bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1
        self.learned = learned
        self.entropy_type = entropy_type
        if self.learned:
            self.scale = nn.Parameter(torch.ones(num_channels)/self.qmax, requires_grad=True)
            self.beta = nn.Parameter(torch.ones(num_channels)/self.qmax, requires_grad=True)
        self.weight = weight

    def _init_data(self, tensor):
        device = tensor.device
        t_min, t_max = tensor.min(dim=0)[0], tensor.max(dim=0)[0]
        scale = (t_max - t_min) / (self.qmax-self.qmin)
        self.beta.data = t_min.to(device)
        self.scale.data = scale.to(device)

    def forward(self, x):
        if self.learned:
            grad = 1.0 / ((self.qmax * x.numel()) ** 0.5)
            s_scale = grad_scale(self.scale, grad)
            beta_scale = grad_scale(self.beta, grad)
            s_scale, beta_scale = self.scale, self.beta
            code = ((x - beta_scale) / s_scale).clamp(self.qmin, self.qmax)
            quant = ste(code)
            dequant = quant * s_scale + beta_scale
        else:
            code = (x * self.qmax).clamp(self.qmin, self.qmax)
            quant = ste(code)
            dequant = quant / self.qmax

        bits, entropy_loss = 0, 0
        if not self.training:
            num_points, num_channels = x.shape
            bits = self.size(quant)
        return dequant, entropy_loss*self.weight, bits

    def size(self, quant):
        index_bits = 0
        compressed, histogram_table, unique = compress_matrix_flatten_categorical(quant.int().flatten().tolist())
        index_bits += get_np_size(compressed) * 8
        index_bits += get_np_size(histogram_table) * 8
        index_bits += get_np_size(unique) * 8 
        index_bits += self.scale.numel()*torch.finfo(self.scale.dtype).bits
        index_bits += self.beta.numel()*torch.finfo(self.beta.dtype).bits
        return index_bits

    def compress(self, x):
        code = ((x - self.beta) / self.scale).clamp(self.qmin, self.qmax)
        return code.round(), code.round()* self.scale + self.beta

    def decompress(self, x):
        return x * self.scale + self.beta

class VectorQuantizer(nn.Module):
    def __init__(self, num_quantizers=1, codebook_dim=1, codebook_size=64, kmeans_iters=10, vector_type="vector"):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.vector_type = vector_type
        if self.num_quantizers == 1:
            if self.vector_type == "vector":
                self.quantizer = VectorQuantize(dim=codebook_dim, codebook_size=codebook_size, decay = 0.8, commitment_weight = 1., kmeans_init = True, 
                    kmeans_iters = kmeans_iters)#learnable_codebook=True, ema_update = False, orthogonal_reg_weight =1.)
        else:
            if self.vector_type == "vector":
                self.quantizer = ResidualVQ(dim=codebook_dim, codebook_size=codebook_size, num_quantizers=num_quantizers, decay = 0.8, commitment_weight = 1., kmeans_init = True, 
                    kmeans_iters = kmeans_iters) #learnable_codebook=True, ema_update=False, orthogonal_reg_weight=0., in_place_codebook_optimizer=torch.optim.Adam)

    def forward(self, x):
        if self.training:
            x, _, l_vq = self.quantizer(x)
            l_vq = torch.sum(l_vq)
            return x, l_vq, 0
        else:
            num_points, num_channels = x.shape
            x, embed_index, l_vq = self.quantizer(x)
            l_vq = torch.sum(l_vq)
            bits = self.size(embed_index)
            # unit_bit = bits / num_points / num_channels
            return x, l_vq, bits

    def size(self, embed_index):
        if self.num_quantizers == 1:
            if self.vector_type == "vector":
                codebook_bits = self.quantizer._codebook.embed.numel()*torch.finfo(self.quantizer._codebook.embed.dtype).bits
            elif self.vector_type == "ste":
                codebook_bits = self.quantizer.embedding.weight.data.numel()*torch.finfo(self.quantizer.embedding.weight.data.dtype).bits
            index_bits = 0
            compressed, histogram_table, unique = compress_matrix_flatten_categorical(embed_index.int().flatten().tolist())
            index_bits += get_np_size(compressed) * 8
            index_bits += get_np_size(histogram_table) * 8
            index_bits += get_np_size(unique) * 8  
        else:
            codebook_bits, index_bits = 0, 0
            for quantizer_index, layer in enumerate(self.quantizer.layers):
                if self.vector_type == "vector":
                    codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
                elif self.vector_type == "ste":
                    codebook_bits += layer.embedding.weight.data.numel()*torch.finfo(layer.embedding.weight.data.dtype).bits
            compressed, histogram_table, unique = compress_matrix_flatten_categorical(embed_index.int().flatten().tolist())
            index_bits += get_np_size(compressed) * 8
            index_bits += get_np_size(histogram_table) * 8
            index_bits += get_np_size(unique) * 8  
        total_bits = codebook_bits + index_bits
        #print("vq:", embed_index.shape, codebook_bits, index_bits)
        return total_bits

    def compress(self, x):
        x, embed_index, _ = self.quantizer(x)
        return x, embed_index

    def decompress(self, embed_index):
        recon = 0
        for i,layer in enumerate(self.quantizer.layers):
            recon += layer._codebook.embed[0, embed_index[:, i]]
        return recon



def compress_matrix_flatten_categorical(matrix, return_table=False):
    '''
    :param matrix: np.array
    :return compressed, symtable
    '''
    matrix = np.array(matrix) #matrix.flatten()
    
    # Special case: matrix with only one value
    if matrix.size == 0:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    
    unique, unique_indices, unique_inverse, unique_counts = np.unique(matrix, return_index=True, return_inverse=True, return_counts=True, axis=None)
    
    # Special case: only one unique value
    if len(unique) == 1:
        # Return minimal compression info for single value
        # Just need to store the value and count, no need for entropy coding
        return np.array([unique[0]], dtype=np.uint8), np.array([len(matrix)], dtype=np.int32), unique.astype(np.uint8)
    
    min_value = np.min(unique)
    max_value = np.max(unique)
    unique = unique.astype(judege_type(min_value, max_value))
    message = unique_inverse.astype(np.int32)
    probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
    entropy_model = constriction.stream.model.Categorical(probabilities)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message, entropy_model)
    compressed = encoder.get_compressed()
    return compressed, unique_counts, unique

# def compress_matrix_flatten_categorical(matrix, return_table=False):
#     '''
#     :param matrix: np.array
#     :return compressed, symtable
#     '''
#     matrix = np.array(matrix) #matrix.flatten()
#     unique, unique_indices, unique_inverse, unique_counts = np.unique(matrix, return_index=True, return_inverse=True, return_counts=True, axis=None)
#     min_value = np.min(unique)
#     max_value = np.max(unique)
#     unique = unique.astype(judege_type(min_value, max_value))
#     message = unique_inverse.astype(np.int32)
#     probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
#     entropy_model = constriction.stream.model.Categorical(probabilities)
#     encoder = constriction.stream.stack.AnsCoder()
#     encoder.encode_reverse(message, entropy_model)
#     compressed = encoder.get_compressed()
#     return compressed, unique_counts, unique

def decompress_matrix_flatten_categorical(compressed, unique_counts, quant_symbol, symbol_length, symbol_shape):
    '''
    :param matrix: np.array
    :return compressed, symtable
    '''
    # Check if we have valid data for probability distribution
    if len(unique_counts) == 0 or np.sum(unique_counts) == 0:
        # Return empty results for invalid input
        return np.array([], dtype=np.int32)
    
    # Special case: single value (compressed contains the value, unique_counts contains the count)
    if len(unique_counts) == 1 and len(compressed) == 1:
        # Return array filled with the single value
        return np.full(symbol_length, compressed[0], dtype=np.int32).reshape(symbol_shape)
    
    probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
    
    # Additional validation for probabilities
    if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
        return np.array([], dtype=np.int32)
    
    entropy_model = constriction.stream.model.Categorical(probabilities)
    decoder = constriction.stream.stack.AnsCoder(compressed)
    decoded = decoder.decode(entropy_model, symbol_length)
    decoded = quant_symbol[decoded].reshape(symbol_shape)#.astype(np.int32)
    return decoded

# def decompress_matrix_flatten_categorical(compressed, unique_counts, quant_symbol, symbol_length, symbol_shape):
#     '''
#     :param matrix: np.array
#     :return compressed, symtable
#     '''
#     probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
#     entropy_model = constriction.stream.model.Categorical(probabilities)
#     decoder = constriction.stream.stack.AnsCoder(compressed)
#     decoded = decoder.decode(entropy_model, symbol_length)
#     decoded = quant_symbol[decoded].reshape(symbol_shape)#.astype(np.int32)
#     return decoded


def judege_type(min, max):
    if min>=0:
        if max<=256:
            return np.uint8
        elif max<=65535:
            return np.uint16
        else:
            return np.uint32
    else:
        if max<128 and min>=-128:
            return np.int8
        elif max<32768 and min>=-32768:
            return np.int16
        else:
            return np.int32
        
def get_np_size(x):
    return x.size * x.itemsize
