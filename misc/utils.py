import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
# from torch.nn.functional import one_hot
# from torch.utils._pytree import tree_flatten, tree_unflatten
# from torchvision.utils import _parse_labels_getter, has_any, is_pure_tensor, query_chw, query_size


# class _BaseMixUpCutMix(Transform):
#     def __init__(self, *, alpha: float = 1.0, num_classes: int, labels_getter="default") -> None:
#         super().__init__()
#         self.alpha = float(alpha)
#         self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

#         self.num_classes = num_classes

#         self._labels_getter = _parse_labels_getter(labels_getter)

#     def forward(self, *inputs):
#         inputs = inputs if len(inputs) > 1 else inputs[0]
#         flat_inputs, spec = tree_flatten(inputs)
#         needs_transform_list = self._needs_transform_list(flat_inputs)

#         if has_any(flat_inputs, PIL.Image.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask):
#             raise ValueError(f"{type(self).__name__}() does not support PIL images, bounding boxes and masks.")

#         labels = self._labels_getter(inputs)
#         if not isinstance(labels, torch.Tensor):
#             raise ValueError(f"The labels must be a tensor, but got {type(labels)} instead.")
#         elif labels.ndim != 1:
#             raise ValueError(
#                 f"labels tensor should be of shape (batch_size,) " f"but got shape {labels.shape} instead."
#             )

#         params = {
#             "labels": labels,
#             "batch_size": labels.shape[0],
#             **self._get_params(
#                 [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
#             ),
#         }

#         # By default, the labels will be False inside needs_transform_list, since they are a torch.Tensor coming
#         # after an image or video. However, we need to handle them in _transform, so we make sure to set them to True
#         needs_transform_list[next(idx for idx, inpt in enumerate(flat_inputs) if inpt is labels)] = True
#         flat_outputs = [
#             self._transform(inpt, params) if needs_transform else inpt
#             for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
#         ]

#         return tree_unflatten(flat_outputs, spec)

#     def _check_image_or_video(self, inpt: torch.Tensor, *, batch_size: int):
#         expected_num_dims = 5 if isinstance(inpt, tv_tensors.Video) else 4
#         if inpt.ndim != expected_num_dims:
#             raise ValueError(
#                 f"Expected a batched input with {expected_num_dims} dims, but got {inpt.ndim} dimensions instead."
#             )
#         if inpt.shape[0] != batch_size:
#             raise ValueError(
#                 f"The batch size of the image or video does not match the batch size of the labels: "
#                 f"{inpt.shape[0]} != {batch_size}."
#             )

#     def _mixup_label(self, label: torch.Tensor, *, lam: float) -> torch.Tensor:
#         label = one_hot(label, num_classes=self.num_classes)
#         if not label.dtype.is_floating_point:
#             label = label.float()
#         return label.roll(1, 0).mul_(1.0 - lam).add_(label.mul(lam))

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# def accuracy(output, target, topk=(1,)):
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].float().sum(0)
#         # correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch
"""
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    train_transform.transforms.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):

    n_params_from_auxiliary_head = np.sum(np.prod(v.size()) for name, v in model.named_parameters()) - \
                                   np.sum(np.prod(v.size()) for name, v in model.named_parameters()
                                          if "auxiliary" not in name)
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (n_params_trainable - n_params_from_auxiliary_head) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)
        
def load_checkpoint(save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    checkpoint = torch.load(filename)
    return checkpoint
    

def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1,).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)