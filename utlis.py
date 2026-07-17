from __future__ import print_function
import numpy as np
import os
import os.path
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as patches
import matplotlib.image as mpimg
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as data
from torchvision import transforms
from typing import Callable
from timeit import default_timer as timer
import math
import abc
from typing import Union
import tqdm
from sklearn.decomposition import PCA
from matplotlib import colors
import cv2
import skimage.exposure
from numpy.random import default_rng


device = ('cuda' if torch.cuda.is_available() else 'cpu')


def read_file(file_path, number_array, number_array_batch):
    data = np.genfromtxt(file_path, delimiter=',')
    data = data[1:, 2:]
    number_beam = number_array - number_array_batch + 1
    scanning_length = int(data.size/(number_beam*data.shape[1]))
    
    cscan_data = np.zeros((number_beam, scanning_length, data.shape[1]))
    for j in range(scanning_length):
        cscan_data[:, j, :] = data[j*number_beam:(j+1)*number_beam, :]

    return cscan_data


def plot_3d(data):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    fig.subplots_adjust(right = 1)
    colors = plt.cm.plasma(data)
    ax.voxels(data, facecolors = colors)
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    m = cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    m.set_array([])
    ax.set_xlabel("Phase Array Beam ID")
    ax.set_ylabel("Scanning Legnth")
    ax.set_zlabel("Thickness")
    ax.invert_zaxis()
    plt.show()


def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


def feature_map_permute(input):
    s = input.data.shape
    l = len(s)

    # permute feature channel to the last:
    # NxCxL --> NxLxC
    if l == 2:
        x = input # NxC
    elif l == 3:
        x = input.permute(0, 2, 1)
    elif l == 4:
        x = input.permute(0, 2, 3, 1)
    elif l == 5:
        x = input.permute(0, 2, 3, 4, 1)
    else:
        x = []
        print('wrong feature map size')
    x = x.contiguous()
    # NxLxC --> (NxL)xC
    x = x.view(-1, s[1])
    return x


class EntropyLoss(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        b = x * torch.log(x + self.eps)
        b = -1.0 * b.sum(dim=1)
        b = b.mean()
        return b
    

class EntropyLossEncap(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLossEncap, self).__init__()
        self.eps = eps
        self.entropy_loss = EntropyLoss(eps)

    def forward(self, input):
        score = feature_map_permute(input)
        ent_loss_val = self.entropy_loss(score)
        return ent_loss_val
    

class MeanAbsoluteRelativeError(nn.Module):
    def __init__(self, eps = 1e-12):
        super(MeanAbsoluteRelativeError, self).__init__()
        self.eps = eps

    def forward(self, pred, true):
        mare = torch.div(torch.abs(true - pred), torch.abs(true + self.eps))
        mare = mare.sum(dim=-1)
        mare = mare.mean()
        return mare
    

class SoftThresholding(nn.Module):
    def __init__(self):
        super(SoftThresholding, self).__init__()

    def forward(self, input, lambd):
        output = torch.sign(input) * nn.functional.relu(torch.abs(input) - lambd)
        return output


class L21Norm(nn.Module):
    def __init__(self):
        super(L21Norm, self).__init__()

    def forward(self, pred, truth):
        input = pred - truth
        input = torch.transpose(input.squeeze(), 0, 1)
        output = torch.mean(torch.norm(input, dim=0))
        return output
    

class RelativeL21Norm(nn.Module):
    def __init__(self, eps=1e-6):
        super(RelativeL21Norm, self).__init__()
        self.eps = eps

    def forward(self, pred, truth):
        input = (pred - truth) / (truth + self.eps)
        input = torch.transpose(input.squeeze(), 0, 1)
        output = torch.mean(torch.norm(input, dim=0))
        return output


class MVTEC(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 category='carpet', resize=None, interpolation=2, train_defect=False, clean=False):
        # 0: InterpolationMode.NEAREST,  1: InterpolationMode.LANCZOS
        # 2: InterpolationMode.BILINEAR, 3: InterpolationMode.BICUBIC
        # 4: InterpolationMode.BOX,      5: InterpolationMode.HAMMING
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = resize
        self.interpolation = interpolation
        self.train_defect = train_defect
        self.clean = clean

        def read_img(file):
            img = mpimg.imread(file)
            if img.shape[2] == 4:
                img = img[:, :, 0:3]
            if file[-3:] != 'bmp':
                img = img * 255
            return img.astype(np.uint8)

        # load images for training
        if self.train:
            self.train_data = []
            self.train_labels = []
            self.train_indices = []
            cwd = os.getcwd()
            if self.train_defect:
                trainFolder = self.root + '/' + category + '/train/defect/'
            else:
                trainFolder = self.root + '/' + category + '/train/good/'
            os.chdir(trainFolder)
            filenames = [f.name for f in os.scandir()]
            for file in filenames:
                self.train_data.append(read_img(file))
                self.train_labels.append(1)
                self.train_indices.append(int(file[:-4]))
            os.chdir(cwd)

            if self.clean and not self.train_defect:
                # replace anomalous training images with their pre-corruption
                # background, recovering the uncontaminated training set
                backgroundFolder = self.root + '/' + category + '/train/defect_background/'
                assert os.path.isdir(backgroundFolder), \
                    f'No defect_background folder for category {category} — clean reconstruction unavailable.'
                index_pos = {idx: i for i, idx in enumerate(self.train_indices)}
                os.chdir(backgroundFolder)
                for file in [f.name for f in os.scandir()]:
                    self.train_data[index_pos[int(file[:-4])]] = read_img(file)
                os.chdir(cwd)

            self.train_data = np.array(self.train_data)
        else:
        # load images for testing
            self.test_data = []
            self.test_labels = []
            self.test_indices = []
            
            cwd = os.getcwd()
            testFolder = self.root + '/' + category + '/test/'
            os.chdir(testFolder)
            subfolders = [sf.name for sf in os.scandir() if sf.is_dir()]
            cwsd = os.getcwd()
            
            # for every subfolder in test folder
            for subfolder in subfolders:
                label = 0
                if subfolder == 'good':
                    label = 1
                testSubfolder = './' + subfolder + '/'
                os.chdir(testSubfolder)
                filenames = [f.name for f in os.scandir()]
                for file in filenames:
                    img = mpimg.imread(file)
                    if img.shape[2] == 4:
                        img = img[:, :, 0:3]
                    if file[-3:] != 'bmp':
                        img = img * 255
                    img = img.astype(np.uint8)
                    self.test_data.append(img)
                    self.test_labels.append(label)
                    self.test_indices.append(int(file[:-4]))
                os.chdir(cwsd)
            os.chdir(cwd)
                
            self.test_data = np.array(self.test_data)
                
    def __getitem__(self, index):
        if self.train:
            img, target, idx = self.train_data[index], self.train_labels[index], self.train_indices[index]
        else:
            img, target, idx = self.test_data[index], self.test_labels[index], self.test_indices[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        # resizing image
        if self.resize is not None:
            resizeTransf = transforms.Resize(self.resize, self.interpolation)
            img = resizeTransf(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, idx

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    

def _get_corner_min_array(f_mat: np.ndarray, i: int, j: int) -> float:
    if i > 0 and j > 0:
        a = min(f_mat[i - 1, j - 1],
                f_mat[i, j - 1],
                f_mat[i - 1, j])
    elif i == 0 and j == 0:
        a = f_mat[i, j]
    elif i == 0:
        a = f_mat[i, j - 1]
    else:  # j == 0:
        a = f_mat[i - 1, j]
    return a


def _bresenham_pairs(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Generates the diagonal coordinates

    Parameters
    ----------
    x0 : int
        Origin x value
    y0 : int
        Origin y value
    x1 : int
        Target x value
    y1 : int
        Target y value

    Returns
    -------
    np.ndarray
        Array with the diagonal coordinates
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dim = max(dx, dy)
    pairs = np.zeros((dim, 2), dtype=np.int64)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx // 2
        for i in range(dx):
            pairs[i, 0] = x
            pairs[i, 1] = y
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        for i in range(dy):
            pairs[i, 0] = x
            pairs[i, 1] = y
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return pairs


def _fast_distance_matrix(p, q, diag, dist_func):
    n_diag = diag.shape[0]
    diag_max = 0.0
    i_min = 0
    j_min = 0
    p_count = p.shape[0]
    q_count = q.shape[0]

    # Create the distance array
    dist = np.full((p_count, q_count), np.inf, dtype=np.float64)

    # Fill in the diagonal with the seed distance values
    for k in range(n_diag):
        i0 = diag[k, 0]
        j0 = diag[k, 1]
        d = dist_func(p[i0], q[j0])
        diag_max = max(diag_max, d)
        dist[i0, j0] = d

    for k in range(n_diag - 1):
        i0 = diag[k, 0]
        j0 = diag[k, 1]
        p_i0 = p[i0]
        q_j0 = q[j0]

        for i in range(i0 + 1, p_count):
            if np.isinf(dist[i, j0]):
                d = dist_func(p[i], q_j0)
                if d < diag_max or i < i_min:
                    dist[i, j0] = d
                else:
                    break
            else:
                break
        i_min = i

        for j in range(j0 + 1, q_count):
            if np.isinf(dist[i0, j]):
                d = dist_func(p_i0, q[j])
                if d < diag_max or j < j_min:
                    dist[i0, j] = d
                else:
                    break
            else:
                break
        j_min = j
    return dist


def _fast_frechet_matrix(dist: np.ndarray, diag: np.ndarray,
                         p: np.ndarray, q: np.ndarray) -> np.ndarray:

    for k in range(diag.shape[0]):
        i0 = diag[k, 0]
        j0 = diag[k, 1]

        for i in range(i0, p.shape[0]):
            if np.isfinite(dist[i, j0]):
                c = _get_corner_min_array(dist, i, j0)
                if c > dist[i, j0]:
                    dist[i, j0] = c
            else:
                break

        # Add 1 to j0 to avoid recalculating the diagonal
        for j in range(j0 + 1, q.shape[0]):
            if np.isfinite(dist[i0, j]):
                c = _get_corner_min_array(dist, i0, j)
                if c > dist[i0, j]:
                    dist[i0, j] = c
            else:
                break
    return dist


def _fdfd_matrix(p: np.ndarray, q: np.ndarray,
                 dist_func: Callable[[np.array, np.array], float]) -> float:
    diagonal = _bresenham_pairs(0, 0, p.shape[0], q.shape[0])
    ca = _fast_distance_matrix(p, q, diagonal, dist_func)
    ca = _fast_frechet_matrix(ca, diagonal, p, q)
    return ca


class FastDiscreteFrechetMatrix(object):
    def __init__(self, dist_func):
        self.times = []
        self.dist_func = dist_func
        self.ca = np.zeros((1, 1))
        # JIT the numba code
        self.distance(np.array([[0.0, 0.0], [1.0, 1.0]]),
                      np.array([[0.0, 0.0], [1.0, 1.0]]))

    def timed_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        start = timer()
        diagonal = _bresenham_pairs(0, 0, p.shape[0], q.shape[0])
        self.times.append(timer() - start)

        start = timer()
        ca = _fast_distance_matrix(p, q, diagonal, self.dist_func)
        self.times.append(timer() - start)

        start = timer()
        ca = _fast_frechet_matrix(ca, diagonal, p, q)
        self.times.append(timer() - start)

        self.ca = ca
        return ca[p.shape[0]-1, q.shape[0]-1]

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        ca = _fdfd_matrix(p, q, self.dist_func)
        self.ca = ca
        return ca[p.shape[0]-1, q.shape[0]-1]


def haversine(p: np.ndarray, q: np.ndarray) -> float:
    d = q - p
    a = math.sin(d[0]/2.0)**2 + math.cos(p[0]) * math.cos(q[0]) \
        * math.sin(d[1]/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return c


def earth_haversine(p: np.ndarray, q: np.ndarray) -> float:
    earth_radius = 6378137.0
    return haversine(np.radians(p), np.radians(q)) * earth_radius


def euclidean(p: np.ndarray, q: np.ndarray) -> float:
    d = p - q
    return math.sqrt(np.dot(d, d))


def memory_item_aggregation(memory_bank, l=2):
    r = int(np.ceil(l / 2))
    s = memory_bank.shape
    memory_bank_agg = torch.ones(s[0], s[1], int(s[-1]*(2*r+1))).to(device)
    for i, memory_instance in enumerate(memory_bank):
        for j, _ in enumerate(memory_instance):
            if j < r:
                memory_bank_agg[i, j, :] = torch.cat((memory_instance[0, :].tile((r-j, 1)), memory_instance[0:j+r+1, :]), 0).view(-1)
            elif j >= s[1]-r:
                memory_bank_agg[i, j, :] = torch.cat((memory_instance[j-r:, :], memory_instance[-1, :].tile((j-s[1]+r+1, 1))), 0).view(-1)
            else:
                memory_bank_agg[i, j, :] = memory_instance[j-r:j+r+1, :].view(-1)

    memory_bank_agg = memory_bank_agg.view(-1, memory_bank_agg.shape[-1])
    # memory_bank_agg = memory_bank_agg[~torch.all(memory_bank_agg == torch.zeros(c), dim=1)]
    return memory_bank_agg

'''
def MemoryBankSearch(model, decoder, training, testing, dist=euclidean, r=20):
    _, _, hidden_features_training = model(training)
    _, _, hidden_features_testing = model(testing)

    hidden_features_training = hidden_features_training.permute(0, 2, 1)
    hidden_features_testing = hidden_features_testing.permute(0, 2, 1)
    hidden_features_training_raw = hidden_features_training.view(-1, hidden_features_training.shape[-1])
    hidden_features_training_agg = MemoryItemAggregation(hidden_features_training)

    fdfdm = FastDiscreteFrechetMatrix(dist)
    t = np.linspace(0, testing.shape[2]-1, num=testing.shape[2])
    anomaly_score_hidden = np.zeros(testing.shape[0])
    anomaly_score_recon = np.zeros(testing.shape[0])
    for i, hidden_feature_testing in enumerate(hidden_features_testing):
        hidden_feature_testing_agg = MemoryItemAggregation(hidden_feature_testing.unsqueeze(0))
        feature_distances = np.zeros(hidden_feature_testing_agg.shape[0])
        hidden_feature_memory = torch.zeros(hidden_feature_testing.shape).to(device)
        for j, hidden_rep_testing in enumerate(hidden_feature_testing_agg):
            rep_distances = np.zeros(hidden_features_training_agg.shape[0])
            for k, hidden_rep_training in enumerate(hidden_features_training_agg):
                rep_distances[k] = 1 - F.cosine_similarity(hidden_rep_testing.view(1, -1), hidden_rep_training.view(1, -1)).item()

            min_indices = np.argpartition(rep_distances, r)
            feature_distances[j] = np.max([np.mean(rep_distances[min_indices[:r]]), 0])  # Average over r smallest distances
            hidden_feature_memory[j, :] = torch.mean(hidden_features_training_raw[min_indices[:r], :], dim=0)

        # Distance in hidden        
        anomaly_score_hidden[i] = np.mean(feature_distances)

        # Frechet distance in recon
        hidden_feature_memory = hidden_feature_memory.permute(1, 0).unsqueeze(0)
        recon_memory = decoder(hidden_feature_memory)
        ascan_recon_memory = np.stack((t, np.squeeze(recon_memory.detach().cpu().numpy())), axis=-1)
        ascan_testing = np.stack((t, np.squeeze(testing[i, :].detach().cpu().numpy())), axis=-1)   
        anomaly_score_recon[i] = fdfdm.distance(ascan_recon_memory, ascan_testing)

    return anomaly_score_hidden, anomaly_score_recon
'''

def memory_processing(model, decoder, features_training, testing, interface_start, interface_end, r, R, dist=euclidean):
    _, _, _, hidden_features_testing = model(testing)
    fdfdm = FastDiscreteFrechetMatrix(dist)
    # t = np.linspace(0, testing.shape[2]-1, num=testing.shape[2])
    t = np.linspace(0, interface_end-interface_start-1, num=interface_end-interface_start)
    anomaly_score_hidden = np.zeros(testing.shape[0])
    anomaly_score_recon_Frechet = np.zeros(testing.shape[0])
    anomaly_score_recon_Euclidean = np.zeros(testing.shape[0])

    for i, hidden_feature_testing in enumerate(hidden_features_testing):
        feature_distances = np.zeros(features_training.shape[0])
        for j, hidden_feature_training in enumerate(features_training):
            hidden_feature_testing = hidden_feature_testing.transpose(0, 1).reshape(1, -1)
            hidden_feature_training = hidden_feature_training.transpose(0, 1).reshape(1, -1)
            feature_distances[j] = 1 - F.cosine_similarity(hidden_feature_testing, hidden_feature_training).item()
            # diff = hidden_feature_testing[:, r:] - hidden_feature_training[:, r:]
            # feature_distances[j] = torch.norm(diff, p=2, dim=0).sum().item()
            
        # Distance in hidden space
        min_indices = np.argpartition(feature_distances, R)
        anomaly_score_hidden_value = np.median(feature_distances[min_indices[:R]])
        anomaly_score_hidden[i] = np.max([anomaly_score_hidden_value, 0])

        # Frechet & Euclidean distance
        anomaly_score_hidden_value_index = np.where(feature_distances==anomaly_score_hidden_value)[0][0]
        hidden_feature_memory = features_training[anomaly_score_hidden_value_index].unsqueeze(0)
        recon_memory = decoder(hidden_feature_memory)
        ascan_recon_memory = np.stack((t, np.squeeze(recon_memory[:, :, interface_start:interface_end].detach().cpu().numpy())), axis=-1)
        ascan_testing = np.stack((t, np.squeeze(testing[i, :, interface_start:interface_end].detach().cpu().numpy())), axis=-1)   
        anomaly_score_recon_Frechet[i] = fdfdm.distance(ascan_recon_memory, ascan_testing)
        anomaly_score_recon_Euclidean[i] = np.linalg.norm(ascan_recon_memory-ascan_testing, 'fro')

    return anomaly_score_hidden, anomaly_score_recon_Frechet, anomaly_score_recon_Euclidean


class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)


class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)


class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)


def coreset_pca_visualization(data, coreset_indices):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
    plt.scatter(principalComponents[coreset_indices, 0], principalComponents[coreset_indices, 1], marker='*')
    plt.legend(['Whole', 'Coreset'])
    plt.show()


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))


def peak_signal_to_noise_ratio(y_true, y_pred, data_range=255):
    '''For 8-bit images'''
    return 10 * np.log10(data_range**2 / np.mean((y_true - y_pred)**2))


def structural_similarity(y_true, y_pred, data_range=255):
    '''For 8-bit images'''
    y_true_flat, y_pred_flat = y_true.flatten(), y_pred.flatten()
    mu_x, sigma_x_sq = np.mean(y_true_flat), np.var(y_true_flat, ddof=1)
    mu_y, sigma_y_sq = np.mean(y_pred_flat), np.var(y_pred_flat, ddof=1)
    sigma_xy = (1/(np.size(y_true_flat)-1)) * np.dot(y_true_flat-mu_x, y_pred_flat-mu_y)
    c1, c2 = (0.01*data_range)**2, (0.03*data_range)**2
    A1 = 2 * mu_x * mu_y + c1
    A2 = 2 * sigma_xy + c2
    B1 = mu_x**2 + mu_y**2 + c1
    B2 = sigma_x_sq + sigma_y_sq + c2
    D = B1 * B2
    return (A1 * A2) / D


def plot_result(X, L, L_truth, S, E, S_mask, G, metric):
    _, axs = plt.subplots(2, 4)
    axs[0, 0].imshow((X * 127.5 + 127.5).astype(np.uint8))
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Input X')
    # axs[0, 1].imshow((L_truth * 127.5 + 127.5).astype(np.uint8))
    axs[0, 1].imshow(L_truth)
    axs[0, 1].axis('off')
    axs[0, 1].set_title('Background L')
    axs[0, 2].imshow((L * 127.5 + 127.5).astype(np.uint8)) 
    axs[0, 2].axis('off')
    axs[0, 2].set_title('Restored L\nPSNR: ' + str(round(metric[2], 3)) + ' & SSIM: ' + str(round(metric[3], 3)))
    with plt.style.context('classic'):
        axs[0, 3].imshow(S)
        axs[0, 3].axis('off')
        axs[0, 3].set_title('Anomaly |S|')
        norm1 = colors.Normalize(0, np.max(S))
        plt.colorbar(cm.ScalarMappable(norm=norm1), ax=axs[0, 3])
        axs[1, 0].imshow(E, vmin=0, vmax=np.max(S))
        axs[1, 0].axis('off')
        axs[1, 0].set_title('Noise |E|')
        # norm2 = colors.Normalize(0, np.max(E))
        plt.colorbar(cm.ScalarMappable(norm=norm1), ax=axs[1, 0])
    with plt.style.context('grayscale'):
        axs[1, 1].imshow(G)
        axs[1, 1].axis('off')
        axs[1, 1].set_title('Grouth Truth |S|')
        axs[1, 2].imshow(S_mask)
        axs[1, 2].axis('off')
        # axs[1, 2].set_title('Anomaly Mask |S| > Thre')
        axs[1, 2].set_title('Thresholded |S|\nDice: ' + str(round(metric[0], 3)) + ' & AUROC: ' + str(round(metric[1], 3)))
        norm3 = colors.Normalize(0, 1)
        plt.colorbar(cm.ScalarMappable(norm=norm3), ax=axs[1, 2])
    axs[1, 3].axis('off')
    plt.show()


class RPCA_gpu:
    """ low-rank and sparse matrix decomposition via RPCA [1] with CUDA capabilities """
    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = torch.zeros_like(self.D)
        self.Y = torch.zeros_like(self.D)
        self.mu = mu or (np.prod(self.D.shape) / (4 * self.norm_p(self.D, 2))).item()
        self.mu_inv = 1 / self.mu
        self.lmbda = lmbda or 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def norm_p(M, p):
        return torch.sum(torch.pow(M, p))

    @staticmethod
    def shrink(M, tau):
        return torch.sign(M) * F.relu(torch.abs(M) - tau)  # hack to save memory

    def svd_threshold(self, M, tau):
        U, s, V = torch.svd(M, some=True)
        return torch.mm(U, torch.mm(torch.diag(self.shrink(s, tau)), V.t()))

    def fit(self, tol=None, max_iter=10000, iter_print=100):
        i, err = 0, np.inf
        Sk, Yk, Lk = self.S, self.Y, torch.zeros_like(self.D)
        _tol = tol or 1e-7 * self.norm_p(torch.abs(self.D), 2)
        while err > _tol and i < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.norm_p(torch.abs(self.D - Lk - Sk), 2) / self.norm_p(self.D, 2)
            i += 1
            # if (i % iter_print) == 0 or i == 1 or i > max_iter or err <= _tol:
                # print(f'Iteration: {i}; Error: {err:0.4e}')
        self.L, self.S = Lk, Sk
        return Lk, Sk


def svt_gpu(X, mask, tau=None, delta=None, eps=1e-2, max_iter=1000, iter_print=5):
    """ matrix completion via singular value thresholding [2] with CUDA capabilties """
    Z = torch.zeros_like(X)
    tau = tau or 5 * np.sum(X.shape) / 2
    delta = delta or (1.2 * np.prod(X.shape) / torch.sum(mask)).item()
    for i in range(max_iter):
        U, s, V = torch.svd(Z, some=True)
        s = F.relu(s - tau)  # hack to save memory
        A = U @ torch.diag(s) @ V.t()
        Z += delta * mask * (X - A)
        error = (torch.norm(mask * (X - A)) / torch.norm(mask * X)).item()
        if i % iter_print == 0: print(f'Iteration: {i}; Error: {error:.4e}')
        if error < eps: break
    return A


def creat_random_blobs(seedval, threshold, H, W):
    rng = default_rng(seed=seedval)
    noise = rng.integers(0, 255, (H, W), np.uint8, True)
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)
    thresh = cv2.threshold(stretch, threshold, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.merge([mask, mask, mask])
    return mask


def find_minimum_bounding_box(M):
    '''M: {0,1}^(H x W) - 2D binary mask'''
    indices = np.where(M == 1.)
    y_min = np.min(indices[0])
    y_max = np.max(indices[0])
    x_min = np.min(indices[1])
    x_max = np.max(indices[1])
    return [y_min, y_max, x_min, x_max]


def generate_bounding_boxes(mask, interpolation_points):
    '''interplation_points: [0, ..., 1] - 1D'''
    img_size = mask.shape[0]
    min_box_ind = find_minimum_bounding_box(mask)
    y_min, y_max = min_box_ind[0], min_box_ind[1]
    x_min, x_max = min_box_ind[2], min_box_ind[3]

    top_left_y = [y_min, 0]  # top_right_y
    top_left_x = [x_min, 0]  # bottom_left_x
    top_right_x = [x_max, img_size]  # bottom_right_x
    bottom_left_y = [y_max, img_size]  # bottom_right_y

    new_points = interpolation_points[1:-1]
    top_left_y_new = np.interp(new_points, [0, 1], top_left_y).astype(np.int8)
    top_left_x_new = np.interp(new_points, [0, 1], top_left_x).astype(np.int8)
    top_right_x_new = np.interp(new_points, [0, 1], top_right_x).astype(np.int8)
    bottom_left_y_new = np.interp(new_points, [0, 1], bottom_left_y).astype(np.int8)

    return {'top left & right y': np.concatenate((np.array([y_min]), top_left_y_new, np.array([0]))),
            'top & bottom left x': np.concatenate((np.array([x_min]), top_left_x_new, np.array([0]))),
            'top & bottom right x': np.concatenate((np.array([x_max]), top_right_x_new, np.array([img_size-1]))),
            'bottom left & right y': np.concatenate((np.array([y_max]), bottom_left_y_new, np.array([img_size-1])))}


def DMS_f(mask, interpolation_points, func, Y_true, Y_pred, data_range=255):
    boxes_ind = generate_bounding_boxes(mask, interpolation_points)

    metrics = np.zeros(interpolation_points.shape)
    for i in range(interpolation_points.shape[0]):
        y_min, y_max = boxes_ind['top left & right y'][i], boxes_ind['bottom left & right y'][i]
        x_min, x_max = boxes_ind['top & bottom left x'][i], boxes_ind['top & bottom right x'][i]
        y_true = Y_true[y_min:y_max+1, x_min:x_max+1, :]
        y_pred = Y_pred[y_min:y_max+1, x_min:x_max+1, :]
        metrics[i] = func(y_true, y_pred, data_range)

    dms_f = np.trapz(metrics, dx=interpolation_points[1]-interpolation_points[0])
    return metrics, dms_f


def plot_dms_f(mask, interpolation_points, X, L, L_truth, results):
    b = generate_bounding_boxes(mask, interpolation_points)
    _, axs = plt.subplots(2, 3)
    axs[0, 0].imshow((X * 127.5 + 127.5).astype(np.uint8))
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Input X')
    axs[0, 1].imshow(L_truth)
    axs[0, 1].axis('off')
    axs[0, 1].set_title('Background L')
    for i in range(interpolation_points.shape[0]):
        y_min, y_max, x_min, x_max = b['top left & right y'][i], b['bottom left & right y'][i], b['top & bottom left x'][i], b['top & bottom right x'][i]
        rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=1, edgecolor='b', facecolor='none')
        axs[0, 0].add_patch(rect)
        # axs[0, 2].text(X.shape[1]+2, y_max-1, 'PSNR: ' + str(np.round(results[0][i], 3)) + '\nSSIM: ' + str(np.round(results[2][i], 3)))
    axs[0, 2].imshow((L * 127.5 + 127.5).astype(np.uint8))
    axs[0, 2].axis('off')
    axs[0, 2].set_title('Restored L')
    axs[1, 0].axis('off')
    axs[1, 1].plot(interpolation_points, results[0])
    axs[1, 1].set_title('DMS-PSNR: ' + str(np.round(results[1], 3)))
    axs[1, 2].plot(interpolation_points, results[2])
    axs[1, 2].set_title('DMS-SSIM: ' + str(np.round(results[3], 3)))
    plt.show()

    