from collections import namedtuple
import sys
sys.path.insert(0, '~/workDir/projects/nsgaformer')

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype_trans = namedtuple('Genotype', 'normal normal_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'conv_1x1',
    'conv_3x1',
    'sep_conv_3x1',
    'sep_conv_5x1',
    'sep_conv_7x1',
    'sep_conv_9x1',
    'sep_conv_11x1',
    'multi_attend_8',
    'multi_attend_16',
    'multi_attend_4',
    'ffn',
    'gelu',
]

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS = Genotype(
    normal=[
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('skip_connect', 0),
        ('dil_conv_3x3', 2)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('skip_connect', 2),
        ('max_pool_3x3', 1)
    ],
    reduce_concat=[2, 3, 4, 5]
)

ENAS = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('sep_conv_5x5', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_5x5', 1),
        ('avg_pool_3x3', 0)
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_3x3', 1),
        ('avg_pool_3x3', 1),
        ('avg_pool_3x3', 1),
        ('sep_conv_3x3', 1),
        ('sep_conv_5x5', 4),
        ('avg_pool_3x3', 1),
        ('sep_conv_3x3', 5),
        ('sep_conv_5x5', 0)
    ],
    reduce_concat=[2, 3, 6]
)

NSGANet = Genotype(
    normal=[
        ('skip_connect', 0),
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 0),
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 1),
        ('sep_conv_3x3', 3),
        ('max_pool_3x3', 1),
        ('sep_conv_5x5', 3),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0)
    ],
    normal_concat=[2, 4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('dil_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('dil_conv_5x5', 1),
        ('skip_connect', 2),
        ('avg_pool_3x3', 1),
        ('dil_conv_5x5', 1),
        ('dil_conv_3x3', 1)
    ],
    reduce_concat=[3, 4, 5, 6]
)

ViT= Genotype_trans(
    # normal=[
    #     ('multi_attend_8', 0),
    #     ('skip_connect', 1),
    #     ('conv_1x1', 2),
    #     ('skip_connect', 2),
    #     ('multi_attend_8', 3),
    #     ('skip_connect', 3),
    #     ('conv_1x1', 4),
    #     ('skip_connect', 4),
    # ],
    # normal_concat=[5]
    normal=[
        # ('skip_connect', 1),
        # ('skip_connect', 1),
        ('multi_attend_8', 1),
        ('skip_connect', 1),
        ('ffn', 2),
        ('skip_connect', 2),
        ('multi_attend_8', 3),
        ('skip_connect', 3),
        ('ffn', 4),
        ('skip_connect', 4),
    ],
    normal_concat=[5]
)

EvolvedEncoder= Genotype_trans(

    normal=[
        # ('skip_connect', 1),
        # ('skip_connect', 1),
        ('gelu', 1),
        ('skip_connect', 1),
        ('conv_1x1', 2),
        ('conv_3x1', 2),
        ('sep_conv_9x1', 3),
        ('skip_connect', 2),
        ('multi_attend_8', 4),
        ('skip_connect', 4),
        ('ffn', 5),
        ('skip_connect', 5),
    ],
    normal_concat=[6]
)
NSGA_ViT_A = Genotype_trans(normal=[('multi_attend_4', 1), ('conv_1x1', 0), ('ffn', 1), ('multi_attend_16', 1), ('conv_3x1', 1), ('multi_attend_4', 1), ('sep_conv_11x1', 3), ('multi_attend_4', 3), ('conv_3x1', 4), ('multi_attend_8', 4)], normal_concat=[2, 5, 6])

NSGA_ViT_A2 = Genotype_trans(normal=[('conv_3x1', 1), ('skip_connect', 1), ('multi_attend_16', 1), ('ffn', 1), ('sep_conv_11x1', 2), ('skip_connect', 2), ('conv_3x1', 4), ('multi_attend_16', 2), ('ffn', 0), ('conv_3x1', 1)], normal_concat=[3, 5, 6]) #valid_acc 84.4

NSGA_ViT = Genotype_trans(normal=[('conv_3x1', 1), ('skip_connect', 1), ('sep_conv_11x1', 2), ('conv_1x1', 2), ('ffn', 3), ('skip_connect', 1), ('multi_attend_16', 4), ('skip_connect', 4), ('ffn', 5), ('skip_connect', 5)], normal_concat=[6])


NSGA_ViT_B2= Genotype_trans(normal=[('sep_conv_9x1', 1), ('skip_connect', 0), ('ffn', 2), ('skip_connect', 1), ('multi_attend_4', 3), ('skip_connect', 3), ('ffn', 4), ('skip_connect', 4), ('conv_3x1', 5), ('conv_3x1', 5)], normal_concat=[6]) #valid acc 82