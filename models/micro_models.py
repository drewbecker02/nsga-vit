import sys
import sys
sys.path.insert(0, '/home/ajb46717/workDir/projects/nsgaformer')
from models.micro_operations import *
from misc.utils import drop_path

from torch.nn import LayerNorm, Conv2d

import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchvision.ops.misc import ConvNormActivation, MLP, Conv2dNormActivation



DEFAULT_PADDINGS = {
    'none': 0,
    'skip_connect': 0,
    'conv_1x1': 0,
    'conv_3x1': 1,
    'sep_conv_3x1': 1,
    'sep_conv_5x1': 2,
    'sep_conv_7x1': 3,
    'sep_conv_9x1': 4,
    'sep_conv_11x1': 5,
    'mulit_attend_8': 0,
    'mulit_attend_16': 0,
    'mulit_attend_4': 0,
}



################################################################################################################
class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        

class ViTCell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, 
                 # reduction, reduction_prev, 
                 SE=False, drop_prob=0.0, final_combiner= 'concat'):
        super(ViTCell, self).__init__()
        print(C_prev_prev, C_prev, C)
        self.drop_prob = drop_prob
        self.se_layer = None
        self.C = C #Input Channels
        
        #Inherited from NSGA-Net
        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        
        #Layer Normalization
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.ln1 = norm_layer(C)
        
        #add or concat
        self.final_combiner = final_combiner
        
        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat
        
        self._compile(C, op_names, indices, concat, norm_layer)
   

    def _compile(self, C, op_names, indices, concat, norm_layer):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) //2
        self._concat = concat
        if self.final_combiner == 'concat':
            self.multiplier = len(concat)
        else:
            self.multiplier = 1
            
        #layer norm for increased dimension size
        self.ln2 = norm_layer(C*self.multiplier)
        
        
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 1
            if not isinstance(OPS[name], nn.MultiheadAttention):
                op = OPS[name](C, stride, True)
            else:
                op = partial(OPS[name](C, stride, True))
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        #permute for 1x1 preprocessing
        s0 = self.preprocess0(s0.permute(0,2,1))
        s1 = self.preprocess1(s1.permute(0,2,1))
        
        s0 = s0.permute(0,2,1)
        s1 = s1.permute(0,2,1)
        #added norm layer
        s0 = self.ln1(s0)
        s1 = self.ln1(s1)
        states = [s0, s1]
        
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
           
            if isinstance(op1, nn.MultiheadAttention):
#                if self.training and drop_prob > 0: #attention-level dropout
#                    op1 = op1(dropout=drop_prob)
                h1, _ = op1(h1, h1, h1)
            else: #conv layer or identity
                h1 = h1.permute(0,2,1)
                h1 = op1(h1)
                h1 = h1.permute(0,2,1)
            if isinstance(op2, nn.MultiheadAttention):
#                if self.training and drop_prob > 0: #attention-level dropout
#                    op2 = op2(dropout=drop_prob)
                h2, _ = op2(h2, h2, h2)
            else:
                h2 = h2.permute(0,2,1)
                op2(h2)
                h2 = h2.permute(0,2,1)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity) and not isinstance(op1, nn.MultiheadAttention):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity) and not isinstance(op2, nn.MultiheadAttention):
                    h2 = drop_path(h2, drop_prob)
                    
            #Padding      
            if h1.shape[1] <  h2.shape[1]:
                h1 = F.pad(h1, (0, 0, 1, 1))
            elif h1.shape[1] >  h2.shape[1]:
                h2 = F.pad(h2, (0, 0, 1, 1))

            s = h1 + h2
            # print(s.shape)
            states += [s]
            # print(self._concat)
            # print(self._steps)
        # print(len(states))
        
        
        if self.final_combiner == 'concat':  
            x = torch.cat([states[i] for i in self._concat], dim=2)
        elif self.final_combiner == 'add' and len(self._concat) >=2:
            x = states[self._concat[0]] + states[self._concat[1]]
            for i in self._concat[2:]:
                x += states[i]
        else:
            x = states[self._concat[0]]
        # print(out.shape)
        return x
        

class ViTNetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype, SE=False, drop_prob = 0,
                 patchify = True, final_combiner= 'concat', stem_layers = 3):
        self.drop_prob = drop_prob
        super(ViTNetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.patchify = patchify
        
        #input embedding
        self.C = C
        
        #SET PATCH SIZE
        image_size = 32
        patch_size = 4
        stem_multiplier = 1
        self.image_size = image_size
        self.patch_size = patch_size
        self.stem_multiplier = stem_multiplier
        C_curr = stem_multiplier * C
        
        #INITIALIZE STEM
        if self.patchify:
            self.stem = Conv2d(
                    in_channels=3, #Should be 3 for initial
                    out_channels=C,
                    kernel_size=patch_size, #patchify stem
                    stride=patch_size,)
            
            # Init the patchify stem weights FROM PYTORCH
            fan_in = 3 * patch_size**2
            nn.init.trunc_normal_(self.stem.weight, std=math.sqrt(1 / fan_in))

        else: #convolution stem (NOT USED UNLESS SPECIFIED)
            last_step: nn.Module = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=1)
            nn.init.normal_( 
                last_step.weight, mean=0.0, std=math.sqrt(2.0 / C_curr)
            )
            self.stem = nn.Sequential()
            C_curr = C // ((stem_layers -1)* 2)
            C_prev = C // ((stem_layers -1)* 2)
            for i in range(stem_layers - 1): # 1x1 conv is always final layer
                if i == 0: 
                    self.stem.add_module(
                        f"conv_bn_relu_{i}",
                        ConvNormActivation(
                            in_channels=3,
                            out_channels=C_curr,
                            kernel_size=3,
                            stride=2,
                            padding=1
                        )
                    )
                else:
                    self.stem.add_module(
                        f"conv_bn_relu_{i}",
                        ConvNormActivation(
                            in_channels=C_prev,
                            out_channels=C_curr,
                            kernel_size=3,
                            stride=2,
                            padding=1
                        )
                    )
                C_prev, C_curr = C_curr, C_curr*2
            self.stem.append(last_step)
            
        #create sequence length
        C_curr = C #just to be sure 
        seq_length = (image_size // patch_size) ** 2
        seq_length += 1
        self.seq_length = seq_length
        
        # add a class token akin to language models
        self.class_token = nn.Parameter(torch.zeros(1, 1, C_curr))
        
       # Add learnable position embedding
        self.pos_embedding = nn.Parameter(torch.empty(1, self.seq_length, C_curr,).normal_(std=0.02)) 
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C_curr # inputs from the stem same size as first cell
        
        self.cells = nn.ModuleList()
        
        for i in range(layers):
            
            cell = ViTCell(genotype, C_prev_prev, C_prev, C_curr, SE=SE)
            self.cells += [cell]
            # C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
        print ('Linear hidden dim??:', C_prev)
            
        
        # self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        # nn.init.zeros_(self.classifier.weight)
        # nn.init.zeros_(self.classifier.bias)



    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size #4
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        # n_h = 8
        n_w = w // p
        # n_w = 8
        
        # print("SHAPE GEFORE STEM", x.shape)
        
        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.stem(x) # should be ([128, 16, 8, 8])
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, n_h * n_w))
        # print("SHAPE AFTER STEM", x.shape)
        
        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        x = x.reshape(n, self.C, n_h * n_w)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        return x


    def forward(self, input):
        logits_aux = None
        x = input
        s0 = s1 = self._process_input(x)
        n = s0.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        # print(s0.shape)
        s0 = torch.cat([batch_class_token, s0], dim=1)
        s1 = torch.cat([batch_class_token, s1], dim=1)
    
        for i, cell in enumerate(self.cells):
            
            s0, s1 = s1, cell(s0, s1, self.drop_prob)
#         s1 = s1.permute(0, 2, 1)
#         s0 = s0.permute(0, 2, 1)
        
        # out = self.global_pooling(s1)
        out = s1[:, 0]
        # print(out.view(out.size(0), -1))
        # print ("OUT SHAPE", out.shape)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux

class ViTNetworkImageNet(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype, drop_prob = 0, patchify=True,
                 final_combiner= 'add'):
        super(ViTNetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.patchify = patchify
        image_size = 224
        patch_size = 16
        stem_layers = 5
        self.image_size = image_size
        self.patch_size = patch_size
        self.drop_prob= drop_prob
        if self.patchify:
            self.stem = nn.Sequential(
                nn.Conv2d(3, C, kernel_size=patch_size, stride=patch_size, bias=False),
            )    
        else: #convolution stem (NOT USED UNLESS SPECIFIED)
            last_step: nn.Module = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=1)
            nn.init.normal_( 
                last_step.weight, mean=0.0, std=math.sqrt(2.0 / C_curr)
            )
            self.stem = nn.Sequential()
            C_curr = C // ((stem_layers -1)* 2)
            C_prev = C // ((stem_layers -1)* 2)
            for i in range(stem_layers - 1): # 1x1 conv is always final layer
                if i == 0: 
                    self.stem.add_module(
                        f"conv_bn_relu_{i}",
                        ConvNormActivation(
                            in_channels=3,
                            out_channels=C_curr,
                            kernel_size=3,
                            stride=2,
                            padding=1
                        )
                    )
                else:
                    self.stem.add_module(
                        f"conv_bn_relu_{i}",
                        ConvNormActivation(
                            in_channels=C_prev,
                            out_channels=C_curr,
                            kernel_size=3,
                            stride=2,
                            padding=1
                        )
                    )
                C_prev, C_curr = C_curr, C_curr*2
            self.stem.append(last_step)
        #create sequence length
        C_curr = C #just to be sure 
        seq_length = (image_size // patch_size) ** 2
        seq_length += 1
        self.seq_length = seq_length
        self.class_token = nn.Parameter(torch.zeros(1, 1, C))
        
        #Init the "previous 2 channels" these come from the convolution stem
        self.pos_embedding = nn.Parameter(torch.empty(1, self.seq_length, C,).normal_(std=0.02)) 
        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        for i in range(layers):
            cell = ViTCell(genotype, C_prev_prev, C_prev, C_curr)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
#             if i == 2 * layers // 3:
#                 C_to_auxiliary = C_prev

#         if auxiliary:
#             self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
#         self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size #4
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        # n_h = 8
        n_w = w // p
        # n_w = 8
        
        # print("SHAPE GEFORE STEM", x.shape)
        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.stem(x) # should be ([128, 16, 8, 8])
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, n_h * n_w))
        # print("SHAPE AFTER STEM", x.shape)
        
        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        x = x.reshape(n, -1, n_h * n_w)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        return x
    
    def forward(self, input):
        logits_aux = None
        x = input
        s0 = s1 = self._process_input(x)
        # print(s0.shape)
        n = s0.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        # print(s0.shape)
        s0 = torch.cat([batch_class_token, s0], dim=1)
        s1 = torch.cat([batch_class_token, s1], dim=1)
    
        for i, cell in enumerate(self.cells):
            
            s0, s1 = s1, cell(s0, s1, self.drop_prob)
#         s1 = s1.permute(0, 2, 1)
#         s0 = s0.permute(0, 2, 1)
        
        # out = self.global_pooling(s1)
        out = s1[:, 0]
        # print(out.view(out.size(0), -1))
        # print ("OUT SHAPE", out.shape)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux

    

class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, 
                 reduction, reduction_prev, 
                 SE=False, drop_prob=0.0):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)
        self.drop_prob = drop_prob
        self.se_layer = None

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

        if SE:
            self.se_layer = SELayer(channel=self.multiplier * C)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)//2

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]

        if self.se_layer is None:
            return torch.cat([states[i] for i in self._concat], dim=1)
        else:
            return self.se_layer(torch.cat([states[i] for i in self._concat], dim=1))


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

#this is the model used in the vanilla search

class NetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype, SE=False, drop_prob = 0):
        self.drop_prob = drop_prob
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, SE=SE)

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux

class PyramidNetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype, increment=4, SE=False):
        super(PyramidNetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                # C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, SE=SE)

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

            C_curr += increment

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.droprate)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux

    
    
class NetworkImageNet(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.droprate)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


if __name__ == '__main__':
    import misc.utils as utils
    import models.micro_genotypes as genotypes

    genome = genotypes.ViT
    # model = AlterPyramidNetworkCIFAR(30, 10, 20, True, genome, 6, SE=False)
    #model = PyramidNetworkCIFAR(48, 10, 20, True, genome, 22, SE=True)
    model = ViTNetworkCIFAR(24, 10, 2, True, genome)
    # model = GradPyramidNetworkCIFAR(34, 10, 20, True, genome, 4)
    model.droprate = 0.0

    # calculate number of trainable parameters
    print("param size = {}MB".format(utils.count_parameters_in_MB(model)))
