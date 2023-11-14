# NSGA-ViT
Code accompanying the paper. All codes assume running from root directory. Please update the sys path at the beginning of the codes before running.


![overview](https://github.com/drewbecker02/nsga-vit/blob/beta/img/overview_redraw.png  "Overview of NSGA-Net")

## Requirements
``` 
Python >= 3.6.8, PyTorch >= 1.0.1.post2, torchvision >= 0.2.2, pymoo == 0.3.0
```

## Results on CIFAR-10
![cifar10_pareto](https://github.com/drewbecker02/nsga-vit/blob/master/img/conf_matrix.png "confusion matrix")


``` shell
python validation/test.py --net_type micro --arch NSGANet --init_channels 34 --filter_increment 4 --auxiliary --model_path weights.pt
```
- Expected result: *2.22%* test error rate with *2.20M* model parameters, *550M* Multiply-Adds ([*weights.pt*](https://drive.google.com/open?id=1it_aFoez-U7SkxSuRPYWDVFg8kZwE7E7)). 

``` shell
python validation/test.py --net_type micro --arch NSGANet --init_channels 36 --filter_increment 6 --SE --auxiliary --model_path weights.pt
```
- Expected result: *2.02%* test error rate with *4.05M* model parameters, *817M* Multiply-Adds ([*weights.pt*](https://drive.google.com/open?id=1kLXzKxQ7dazjmANTvgSoeMPHWwYKiOtm)). 

## Pretrained models on CIFAR-10
``` shell
python validation/test.py --arch NSGA_ViT --init_channels 512 --layers 6 --model_path NSGA-ViT-200/weights.pt
```
- Expected result: *8.8%* test error rate with *44.2M* model parameters,([*weights.pt*](https://drive.google.com/open?id=1CMtSg1l2V5p0HcRxtBsD8syayTtS9QAu)). 

## Architecture validation
To validate the results by training from scratch, run
``` 
python validation/train.py --net_type micro --arch NSGA_ViT --layers 6 --init_channels 512  --cutout  --batch_size 128 --droprate 0.2 --epochs 200 --weight_decay 0.3
```
You may need to adjust the batch_size depending on your GPU memory. 

For customized macro search space architectures, change `genome` and `channels` option in `train.py`. 

For customized micro search space architectures, specify your architecture in `models/micro_genotypes.py` and use `--arch` flag to pass the name. 


## Architecture search 
To run architecture search:
``` shell
# micro search space
python search/evolution_search.py --search_space micro --init_channels 256 --layers 3 --epochs 20 --n_offspring 40 --n_gens 50
```

If you would like to run asynchronous and parallelize each architecture's back-propagation training, set `--n_offspring` to `1`. The algorithm will run in *steady-state* mode, in which the population is updated as soon as one new architecture candidate is evaludated. It works reasonably well in single-objective case, a similar strategy is used in [here](https://arxiv.org/abs/1802.01548).  

## Visualization
To visualize the architectures:
``` shell
python visualization/micro_visualize.py NSGANet            # micro search space architectures
```
For customized architecture, first define the architecture in `models/*_genotypes.py`, then substitute `NSGANet` with the name of your customized architecture. 

<!-- ## Citations
If you find the code useful for your research, please consider citing our works
``` 
@article{nsganet,
  title={NSGA-NET: a multi-objective genetic algorithm for neural architecture search},
  author={Lu, Zhichao and Whalen, Ian and Boddeti, Vishnu and Dhebar, Yashesh and Deb, Kalyanmoy and Goodman, Erik and  Banzhaf, Wolfgang},
  booktitle={GECCO-2019},
  year={2018}
}
``` -->

## Acknowledgement 
Code heavily inspired and modified from [pymoo](https://github.com/msu-coinlab/pymoo), [DARTS](https://github.com/quark0/darts#requirements), [pytorch-cifar10](https://github.com/kuangliu/pytorch-cifar) and [NSGANet](https://github.com/ianwhale/nsga-net). 
