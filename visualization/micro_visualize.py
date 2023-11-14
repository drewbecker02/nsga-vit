import sys
# sys.path.insert(0, '/path/to/nsga-net')
sys.path.insert(0, '/home/ajb46717/workDir/projects/nsgaformer')

import os
from models import micro_genotypes as genotypes
from graphviz import Digraph

# op_labels = {
#     'none',
#     'max_pool_3x3',
#     'avg_pool_3x3',
#     'skip_connect',
#     'sep_conv_3x3',
#     'sep_conv_5x5',
#     'dil_conv_3x3',
#     'dil_conv_5x5',
#     'conv_1x1',
#     'multi_attend_8',
#     'mulit_attend_16',
#     'mulit_attend_4'
# }

op_labels = {
    'none': 'none',
    'avg_pool_3x3': 'avg 3x3',
    'max_pool_3x3': 'max 3x3',
    'skip_connect': 'identity',
    'sep_conv_3x3': 'sep 3x3',
    'sep_conv_5x5': 'sep 5x5',
    'sep_conv_7x7': 'sep 7x7',
    'dil_conv_3x3': 'dil 3x3',
    'dil_conv_5x5': 'dil 5x5',
    'conv_7x1_1x7': '7x1 1x7',
    'multi_attend_4': 'multi_attend_4',
    'multi_attend_8': 'multi_attend_8',
    'multi_attend_16': 'multi_attend_16',
    'conv_1x1':'conv_1x1',
    'conv_3x1':'conv_3x1',
    'sep_conv_3x1': 'sep_conv_3x1',
    'sep_conv_5x1': 'sep_conv_5x1',
    'sep_conv_7x1': 'sep_conv_7x1',
    'sep_conv_9x1': 'sep_conv_9x1',
    'sep_conv_11x1': 'sep_conv_11x1',
    'ffn': 'ffn',
    'gelu': 'gelu'
}


def plot(genotype_tup, filename, file_type='pdf', view=True):
    genotype = genotype_tup[0]
    concat = genotype_tup[1]
    g = Digraph(
        format=file_type,
        # graph_attr=dict(margin='0.2', nodesep='0.1', ranksep='0.3'),
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot',
    )
    g.body.extend(['rankdir=LR'])

    g.node("h[i-1]", fillcolor='darkseagreen2')
    g.node("h[i]", fillcolor='darkseagreen2')

    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), label="add", fillcolor='lightblue')

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype[k]
            # g.node(str(steps+k+1), label=op_labels[op], fillcolor='yellow')
            if j == 0:
                u = "h[i-1]"
            elif j == 1:
                u = "h[i]"
            else:
                u = str(j - 2)
            v = str(i)
            # g.edge(u, str(steps+k+1), fillcolor="gray")
            # g.edge(str(steps+k+1), v, fillcolor="gray")
            g.edge(u, v, label=op_labels[op], fillcolor="gray")

    g.node("h[i+1]", label="concat", fillcolor='lightpink')

    for i in range(steps):
        if int(i + 2) in concat:
            g.edge(str(i), "h[i+1]", fillcolor="gray")

    g.node("output", label="h[i+1]", fillcolor='palegoldenrod')
    g.edge("h[i+1]", "output", fillcolor="gray")

    # g.attr(rank='same')

    g.render(filename, view=False)

#    os.remove(filename)


if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #   print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    #   sys.exit(1)

    genotype_name = sys.argv[1]
    try:
        genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)
    
    plot([genotype.normal, genotype.normal_concat], "normal", file_type='png')
#    plot([genotype.reduce, genotype.reduce_concat], "reduction")


