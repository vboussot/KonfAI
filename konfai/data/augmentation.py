import importlib
import torch
from abc import ABC, abstractmethod
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from typing import Union
import os
from konfai import DEEP_LEARNING_API_ROOT
from konfai.utils.config import config
from konfai.utils.utils import _getModule
from konfai.utils.dataset import Attribute, data_to_image


def _translate2DMatrix(t: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.cat((torch.eye(2), torch.tensor([[t[0]], [t[1]]])), dim=1), torch.Tensor([[0,0,1]])), dim=0)

def _translate3DMatrix(t: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.cat((torch.eye(3), torch.tensor([[t[0]], [t[1]], [t[2]]])), dim=1), torch.Tensor([[0,0,0,1]])), dim=0)

def _scale2DMatrix(s: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.cat((torch.eye(2)*s, torch.tensor([[0], [0]])), dim=1), torch.tensor([[0, 0, 1]])), dim=0)

def _scale3DMatrix(s: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.cat((torch.eye(3)*s, torch.tensor([[0], [0], [0]])), dim=1), torch.tensor([[0, 0, 0, 1]])), dim=0)

def _rotation3DMatrix(rotation : torch.Tensor, center: Union[torch.Tensor, None] = None) -> torch.Tensor:
    A = torch.tensor([[torch.cos(rotation[2]), -torch.sin(rotation[2]), 0], [torch.sin(rotation[2]), torch.cos(rotation[2]), 0], [0, 0, 1]])
    B = torch.tensor([[torch.cos(rotation[1]), 0, torch.sin(rotation[1])], [0, 1, 0], [-torch.sin(rotation[1]), 0, torch.cos(rotation[1])]])
    C = torch.tensor([[1, 0, 0], [0, torch.cos(rotation[0]), -torch.sin(rotation[0])], [0, torch.sin(rotation[0]), torch.cos(rotation[0])]])
    rotation_matrix = torch.cat((torch.cat((A.mm(B).mm(C), torch.zeros((3, 1))), dim=1), torch.tensor([[0, 0, 0, 1]])), dim=0)
    if center is not None:
        translation_before = torch.eye(4)
        translation_before[:-1, -1] = -center
        rotation_matrix = translation_before.mm(rotation_matrix)
    if center is not None:
        translation_after = torch.eye(4)
        translation_after[:-1, -1] = center
        rotation_matrix = rotation_matrix.mm(translation_after)
    return rotation_matrix

def _rotation2DMatrix(rotation : torch.Tensor, center: Union[torch.Tensor, None] = None) -> torch.Tensor:
    return torch.cat((torch.cat((torch.tensor([[torch.cos(rotation[0]), -torch.sin(rotation[0])], [torch.sin(rotation[0]), torch.cos(rotation[0])]]), torch.zeros((2, 1))), dim=1), torch.tensor([[0, 0, 1]])), dim=0)

class Prob():

    @config()
    def __init__(self, prob: float = 1.0) -> None:
        self.prob = prob

class DataAugmentationsList():

    @config()
    def __init__(self, nb : int = 10, dataAugmentations: dict[str, Prob] = {"default:RandomElastixTransform" : Prob(1)}) -> None:
        self.nb = nb
        self.dataAugmentations : list[DataAugmentation] = []
        self.dataAugmentationsLoader = dataAugmentations

    def load(self, key: str):
        for augmentation, prob in self.dataAugmentationsLoader.items():
            module, name = _getModule(augmentation, "data.augmentation")
            dataAugmentation: DataAugmentation = getattr(importlib.import_module(module), name)(config = None, DL_args="{}.Dataset.augmentations.{}.dataAugmentations".format(DEEP_LEARNING_API_ROOT(), key))
            dataAugmentation.load(prob.prob)
            self.dataAugmentations.append(dataAugmentation)
    
class DataAugmentation(ABC):

    def __init__(self) -> None:
        self.who_index: dict[int, list[int]] = {}
        self.shape_index: dict[int, list[list[int]]] = {}
        self._prob: float = 0

    def load(self, prob: float):
        self._prob = prob

    def state_init(self, index: Union[None, int], shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        if index is not None:
            if index not in self.who_index:
                self.who_index[index] = torch.where(torch.rand(len(shapes)) < self._prob)[0].tolist()
            else:
                return self.shape_index[index]
        else: 
            index = 0
            self.who_index[index] = torch.where(torch.rand(len(shapes)) < self._prob)[0].tolist()
        
        if len(self.who_index[index]) > 0:
            for i, shape in enumerate(self._state_init(index, [shapes[i] for i in self.who_index[index]], [caches_attribute[i] for i in self.who_index[index]])):
                shapes[self.who_index[index][i]] = shape
        self.shape_index[index] = shapes
        return self.shape_index[index]
    
    @abstractmethod
    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        pass

    def __call__(self, index: int, inputs : list[torch.Tensor], device: Union[torch.device, None]) -> list[torch.Tensor]:
        if len(self.who_index[index]) > 0:
            for i, result in enumerate(self._compute(index, [inputs[i] for i in self.who_index[index]], device)):
                inputs[self.who_index[index][i]] = result if device is None else result.cpu()
        return inputs
    
    @abstractmethod
    def _compute(self, index: int, inputs : list[torch.Tensor], device: Union[torch.device, None]) -> list[torch.Tensor]:
        pass

    def inverse(self, index: int, a: int, input : torch.Tensor) -> torch.Tensor:
        if a in self.who_index[index]:
            input = self._inverse(index, a, input)
        return input
        
    @abstractmethod
    def _inverse(self, index: int, a: int, input : torch.Tensor) -> torch.Tensor:
        pass

class EulerTransform(DataAugmentation):

    def __init__(self) -> None:
        super().__init__()
        self.matrix: dict[int, list[torch.Tensor]] = {}

    def _compute(self, index: int, inputs : list[torch.Tensor], device: Union[torch.device, None]) -> list[torch.Tensor]:
        results = []
        for input, matrix in zip(inputs, self.matrix[index]):
            results.append(F.grid_sample(input.unsqueeze(0).type(torch.float32), F.affine_grid(matrix[:, :-1,...], [1]+list(input.shape), align_corners=True).to(input.device), align_corners=True, mode="bilinear", padding_mode="reflection").type(input.dtype).squeeze(0))
        return results
    
    def _inverse(self, index: int, a: int, input : torch.Tensor) -> torch.Tensor:
        return F.grid_sample(input.unsqueeze(0).type(torch.float32), F.affine_grid(self.matrix[index][a].inverse()[:, :-1,...], [1]+list(input.shape), align_corners=True).to(input.device), align_corners=True, mode="bilinear", padding_mode="reflection").type(input.dtype).squeeze(0)

class Translate(EulerTransform):
    
    @config("Translate")
    def __init__(self, t_min: float = -10, t_max = 10, is_int: bool = False):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.is_int = is_int

    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        dim = len(shapes[0])
        func = _translate3DMatrix if dim == 3 else _translate2DMatrix
        translate = torch.rand((len(shapes), dim)) * torch.tensor(self.t_max-self.t_min) + torch.tensor(self.t_min)
        if self.is_int:
            translate = torch.round(translate*100)/100     
        self.matrix[index] = [torch.unsqueeze(func(value), dim=0) for value in translate]
        return shapes

class Rotate(EulerTransform):

    @config("Rotate")
    def __init__(self, a_min: float = 0, a_max: float = 360, is_quarter: bool = False):
        super().__init__()
        self.a_min = a_min
        self.a_max = a_max
        self.is_quarter = is_quarter

    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        dim = len(shapes[0])
        func = _rotation3DMatrix if dim == 3 else _rotation2DMatrix
        angles = []
        
        if self.is_quarter:
            angles = torch.Tensor.repeat(torch.tensor([90,180,270]), 3)
        else:
            angles = torch.rand((len(shapes), dim))*torch.tensor(self.a_max-self.a_min) + torch.tensor(self.a_min)

        self.matrix[index] = [torch.unsqueeze(func(value), dim=0) for value in angles]
        return shapes      

class Scale(EulerTransform):

    @config("Scale")
    def __init__(self, s_std: float = 0.2):
        super().__init__()
        self.s_std = s_std

    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        func = _scale3DMatrix if len(shapes[0]) == 3 else _scale2DMatrix
        scale = torch.Tensor.repeat(torch.exp2(torch.randn((len(shapes))) * self.s_std).unsqueeze(1), [1, len(shapes[0])])
        self.matrix[index] = [torch.unsqueeze(func(value), dim=0) for value in scale]
        return shapes

class Flip(DataAugmentation):

    @config("Flip")
    def __init__(self, f_prob: Union[list[float], None] = [0.33, 0.33 ,0.33]) -> None:
        super().__init__()
        self.f_prob = f_prob
        self.flip: dict[int, list[int]] = {}

    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        prob = torch.rand((len(shapes), len(self.f_prob))) < torch.tensor(self.f_prob)
        dims = torch.tensor([1, 2, 3][:len(self.f_prob)])
        self.flip[index] = [dims[mask].tolist() for mask in prob]
        return shapes

    def _compute(self, index: int, inputs : list[torch.Tensor], device: Union[torch.device, None]) -> list[torch.Tensor]:
        results = []
        for input, flip in zip(inputs, self.flip[index]):
            results.append(torch.flip(input, dims=flip))
        return results
    
    def _inverse(self, index: int, a: int, input : torch.Tensor) -> torch.Tensor:
        return torch.flip(input, dims=self.flip[index][a])
    

class ColorTransform(DataAugmentation):

    @config("ColorTransform")
    def __init__(self) -> None:
        super().__init__()
        self.matrix: dict[int, list[torch.Tensor]] = {}
    
    def _compute(self, index: int, inputs : list[torch.Tensor], device: Union[torch.device, None]) -> list[torch.Tensor]:
        results = []
        for input, matrix in zip(inputs, self.matrix[index]):
            result = input.reshape([*input.shape[:1], int(np.prod(input.shape[1:]))])
            if input.shape[0] == 3:
                matrix = matrix.to(input.device)
                result = matrix[:, :3, :3] @ result.float() + matrix[:, :3, 3:]
            elif input.shape[0] == 1:
                matrix = matrix[:, :3, :].mean(dim=1, keepdims=True).to(input.device)
                result = result.float() * matrix[:, :, :3].sum(dim=2, keepdims=True) + matrix[:, :, 3:]
            else:
                raise ValueError('Image must be RGB (3 channels) or L (1 channel)')
            results.append(result.reshape(input.shape))
        return results
    
    def _inverse(self, index: int, a: int, inputs : torch.Tensor) -> torch.Tensor:
        pass

class Brightness(ColorTransform):

    @config("Brightness")
    def __init__(self, b_std: float) -> None:
        super().__init__()
        self.b_std = b_std

    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        brightness = torch.Tensor.repeat((torch.randn((len(shapes)))*self.b_std).unsqueeze(1), [1, 3])
        self.matrix[index] = [torch.unsqueeze(_translate3DMatrix(value), dim=0) for value in brightness]
        return shapes

class Contrast(ColorTransform):

    @config("Contrast")
    def __init__(self, c_std: float) -> None:
        super().__init__()
        self.c_std = c_std

    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        contrast = torch.exp2(torch.randn((len(shapes))) * self.c_std)
        self.matrix[index] = [torch.unsqueeze(_scale3DMatrix(value), dim=0) for value in contrast]
        return shapes

class LumaFlip(ColorTransform):

    @config("LumaFlip")
    def __init__(self) -> None:
        super().__init__()
        self.v = torch.tensor([1, 1, 1, 0])/torch.sqrt(torch.tensor(3)) 
    
    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        luma = torch.floor(torch.rand([len(shapes), 1, 1]) * 2)
        self.matrix[index] = [torch.unsqueeze((torch.eye(4) - 2 * self.v.ger(self.v) * value), dim=0) for value in luma]
        return shapes

class HUE(ColorTransform):

    @config("HUE")
    def __init__(self, hue_max: float) -> None:
        super().__init__()
        self.hue_max = hue_max
        self.v = torch.tensor([1, 1, 1])/torch.sqrt(torch.tensor(3)) 
    
    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        theta = (torch.rand([len(shapes)]) * 2 - 1) * np.pi * self.hue_max
        self.matrix[index] = [torch.unsqueeze(_rotation3DMatrix(value.repeat(3), self.v), dim=0) for value in theta]
        return shapes
    
class Saturation(ColorTransform):

    @config("Saturation")
    def __init__(self, s_std: float) -> None:
        super().__init__()
        self.s_std = s_std
        self.v = torch.tensor([1, 1, 1, 0])/torch.sqrt(torch.tensor(3)) 

    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        saturation = torch.exp2(torch.randn((len(shapes))) * self.s_std)
        self.matrix[index] = [(self.v.ger(self.v) + (torch.eye(4) - self.v.ger(self.v))).unsqueeze(0) * value for value in saturation]
        return shapes
    
"""class Filter(DataAugmentation):

    def __init__(self) -> None:
        super().__init__()
        wavelets = {
        'haar': [0.7071067811865476, 0.7071067811865476],
        'db1':  [0.7071067811865476, 0.7071067811865476],
        'db2':  [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
        'db3':  [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
        'db4':  [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523],
        'db5':  [0.003335725285001549, -0.012580751999015526, -0.006241490213011705, 0.07757149384006515, -0.03224486958502952, -0.24229488706619015, 0.13842814590110342, 0.7243085284385744, 0.6038292697974729, 0.160102397974125],
        'db6':  [-0.00107730108499558, 0.004777257511010651, 0.0005538422009938016, -0.031582039318031156, 0.02752286553001629, 0.09750160558707936, -0.12976686756709563, -0.22626469396516913, 0.3152503517092432, 0.7511339080215775, 0.4946238903983854, 0.11154074335008017],
        'db7':  [0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274, 0.012550998556013784, -0.01657454163101562, -0.03802993693503463, 0.0806126091510659, 0.07130921926705004, -0.22403618499416572, -0.14390600392910627, 0.4697822874053586, 0.7291320908465551, 0.39653931948230575, 0.07785205408506236],
        'db8':  [-0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771, -0.00487035299301066, 0.008746094047015655, 0.013981027917015516, -0.04408825393106472, -0.01736930100202211, 0.128747426620186, 0.00047248457399797254, -0.2840155429624281, -0.015829105256023893, 0.5853546836548691, 0.6756307362980128, 0.3128715909144659, 0.05441584224308161],
        'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
        'sym3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
        'sym4': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
        'sym5': [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936, 0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047, -0.021101834024758855, 0.019538882735286728],
        'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
        'sym7': [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357, 0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628, 0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774, 0.004010244871533663, 0.010268176708511255],
        'sym8': [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605, -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235, 0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506, 0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609],
        }
        Hz_lo = np.asarray(wavelets['sym2'])            # H(z)
        Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size)) # H(-z)
        Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2    # H(z) * H(z^-1) / 2
        Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2    # H(-z) * H(-z^-1) / 2
        Hz_fbank = np.eye(4, 1)                         # Bandpass(H(z), b_i)
        for i in range(1, Hz_fbank.shape[0]):
            Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)]).reshape(Hz_fbank.shape[0], -1)[:, :-1]
            Hz_fbank = scipy.signal.convolve(Hz_fbank, [Hz_lo2])
            Hz_fbank[i, (Hz_fbank.shape[1] - Hz_hi2.size) // 2 : (Hz_fbank.shape[1] + Hz_hi2.size) // 2] += Hz_hi2
        self.Hz_fbank = torch.as_tensor(Hz_fbank, dtype=torch.float32)
        
        self.imgfilter_bands = [1,1,1,1]
        self.imgfilter_std = 1


    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:

        return shapes

    def _compute(self, index: int, inputs : list[torch.Tensor]) -> list[torch.Tensor]:
        num_bands = self.Hz_fbank.shape[0]
        assert len(self.imgfilter_bands) == num_bands
        expected_power =torch.tensor([10, 1, 1, 1]) / 13 # Expected power spectrum (1/f).
        for input in inputs:
            batch_size = input.shape[0]
            num_channels = input.shape[1]
            # Apply amplification for each band with probability (imgfilter * strength * band_strength).
            g = torch.ones([batch_size, num_bands]) # Global gain vector (identity).
            for i, band_strength in enumerate(self.imgfilter_bands):
                t_i = torch.exp2(torch.randn([batch_size]) * self.imgfilter_std)
                t = torch.ones([batch_size, num_bands])                  # Temporary gain vector.
                t[:, i] = t_i                                                           # Replace i'th element.
                t = t / (expected_power * t.square()).sum(dim=-1, keepdims=True).sqrt() # Normalize power.
                g = g * t                                                               # Accumulate into global gain.

            # Construct combined amplification filter.
            Hz_prime = g @ self.Hz_fbank                                    # [batch, tap]
            Hz_prime = Hz_prime.unsqueeze(1).repeat([1, num_channels, 1])   # [batch, channels, tap]
            Hz_prime = Hz_prime.reshape([batch_size * num_channels, 1, -1]) # [batch * channels, 1, tap]

            # Apply filter.
            p = self.Hz_fbank.shape[1] // 2
            images = images.reshape([1, batch_size * num_channels, height, width])
            images = torch.nn.functional.pad(input=images, pad=[p,p,p,p], mode='reflect')
            images = conv2d_gradfix.conv2d(input=images, weight=Hz_prime.unsqueeze(2), groups=batch_size*num_channels)
            images = conv2d_gradfix.conv2d(input=images, weight=Hz_prime.unsqueeze(3), groups=batch_size*num_channels)
            images = images.reshape([batch_size, num_channels, height, width])


    def _inverse(self, index: int, a: int, input : torch.Tensor) -> torch.Tensor:
        pass """ 

class Noise(DataAugmentation):

    @config("Noise")
    def __init__(self, n_std: float, noise_step: int=1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        super().__init__()
        self.n_std = n_std
        self.noise_step = noise_step

        self.ts: dict[int, list[torch.Tensor]] = {}
        self.betas = torch.linspace(beta_start, beta_end, noise_step)
        self.betas = Noise.enforce_zero_terminal_snr(self.betas)
        self.alphas = 1 - self.betas
        self.alpha_hat = torch.concat((torch.ones(1), torch.cumprod(self.alphas, dim=0)))
        self.max_T = 0

        self.C = 1
        self.n = 4
        self.d = 0.25
        self._prob = 1

    def enforce_zero_terminal_snr(betas: torch.Tensor):
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
        alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
        alphas_bar = alphas_bar_sqrt ** 2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        return betas
    
    def load(self, prob: float):
        self.max_T = prob*self.noise_step

    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        if int(self.max_T) == 0:
            self.ts[index] = [0 for _ in shapes]
        else: 
            self.ts[index] = [torch.randint(0, int(self.max_T), (1,)) for _ in shapes]
        return shapes
    
    def _compute(self, index: int, inputs : list[torch.Tensor], device: Union[torch.device, None]) -> list[torch.Tensor]:
        results = []
        for input, t in zip(inputs, self.ts[index]):
            alpha_hat_t = self.alpha_hat[t].to(input.device).reshape(*[1 for _ in range(len(input.shape))])
            results.append(alpha_hat_t.sqrt() * input + (1 - alpha_hat_t).sqrt() * torch.randn_like(input.float()).to(input.device)*self.n_std)
        return results

    def _inverse(self, index: int, a: int, input : torch.Tensor) -> torch.Tensor:
        pass

class CutOUT(DataAugmentation):

    @config("CutOUT")
    def __init__(self, c_prob: float, cutout_size: int, value: float) -> None:
        super().__init__()
        self.c_prob = c_prob
        self.cutout_size = cutout_size
        self.centers: dict[int, list[torch.Tensor]] = {}
        self.value = value

    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        self.centers[index] = [torch.rand((3) if len(shape) == 3 else (2)) for shape in shapes]
        return shapes
    
    def _compute(self, index: int, inputs : list[torch.Tensor], device: Union[torch.device, None]) -> list[torch.Tensor]:
        results = []
        for input, center in zip(inputs, self.centers[index]):
            masks = []
            for i, w in enumerate(input.shape[1:]):
                re = [1]*i+[-1]+[1]*(len(input.shape[1:])-i-1)
                masks.append((((torch.arange(w).reshape(re) + 0.5) / w - center[i].reshape([1, 1])).abs() >= torch.tensor(self.cutout_size).reshape([1, 1])/ 2))
            result = masks[0]
            for mask in masks[1:]:
                result = torch.logical_or(result, mask)
            result = result.unsqueeze(0).repeat([input.shape[0], *[1 for _ in range(len(input.shape)-1)]])
            results.append(torch.where(result.to(input.device) == 1, input, torch.tensor(self.value).to(input.device)))
        return results
    
    def _inverse(self, index: int, a: int, input : torch.Tensor) -> torch.Tensor:
        pass

class Elastix(DataAugmentation):

    @config("Elastix")
    def __init__(self, grid_spacing: int = 16, max_displacement: int = 16) -> None:
        super().__init__()
        self.grid_spacing = grid_spacing
        self.max_displacement = max_displacement
        self.displacement_fields: dict[int, list[torch.Tensor]] = {}
        self.displacement_fields_true: dict[int, list[torch.Tensor]] = {}
    
    @staticmethod
    def _formatLoc(new_locs, shape):
        for i in range(len(shape)):
            new_locs[..., i] = 2 * (new_locs[..., i] / (shape[i] - 1) - 0.5)
        new_locs = new_locs[..., [i for i in reversed(range(len(shape)))]]
        return new_locs

    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        print("Compute Displacement Field for index {}".format(index))
        self.displacement_fields[index] = []
        self.displacement_fields_true[index] = []
        for i, (shape, cache_attribute) in enumerate(zip(shapes, caches_attribute)):
            shape = shape
            dim = len(shape)
            if "Spacing" not in cache_attribute:
                spacing = np.array([1.0 for _ in range(dim)])
            else:
                spacing = cache_attribute.get_np_array("Spacing")
            
            grid_physical_spacing = [self.grid_spacing]*dim
            image_physical_size = [size*spacing for size, spacing in zip(shape, spacing)]
            mesh_size = [int(image_size/grid_spacing + 0.5) for image_size,grid_spacing in zip(image_physical_size, grid_physical_spacing)]
            if "Spacing" not in cache_attribute:
                cache_attribute["Spacing"] = np.array([1.0 for _ in range(dim)])
            if "Origin" not in cache_attribute:
                cache_attribute["Origin"] = np.array([1.0 for _ in range(dim)])
            if "Direction" not in cache_attribute:
                cache_attribute["Direction"] = np.eye(dim).flatten()
            
            ref_image = data_to_image(np.expand_dims(np.zeros(shape), 0), cache_attribute)

            bspline_transform = sitk.BSplineTransformInitializer(image1 = ref_image, transformDomainMeshSize = mesh_size, order=3)
            displacement_filter = sitk.TransformToDisplacementFieldFilter()
            displacement_filter.SetReferenceImage(ref_image)
            
            vectors = [torch.arange(0, s) for s in shape]
            grids = torch.meshgrid(vectors, indexing='ij')
            grid = torch.stack(grids)
            grid = torch.unsqueeze(grid, 0)
            grid = grid.type(torch.float).permute([0]+[i+2 for i in range(len(shape))] + [1])
        
            control_points = torch.rand(*[size+3 for size in mesh_size], dim)
            control_points -= 0.5
            control_points *= 2*self.max_displacement
            bspline_transform.SetParameters(control_points.flatten().tolist())
            displacement = sitk.GetArrayFromImage(displacement_filter.Execute(bspline_transform))
            self.displacement_fields_true[index].append(displacement)
            new_locs = grid+torch.unsqueeze(torch.from_numpy(displacement), 0).type(torch.float32)
            self.displacement_fields[index].append(Elastix._formatLoc(new_locs, shape))
            print("Compute in progress : {:.2f} %".format((i+1)/len(shapes)*100))
        return shapes
    
    def _compute(self, index: int, inputs : list[torch.Tensor], device: Union[torch.device, None]) -> list[torch.Tensor]:
        results = []
        for input, displacement_field in zip(inputs, self.displacement_fields[index]):
            results.append(F.grid_sample(input.type(torch.float32).unsqueeze(0), displacement_field.to(input.device), align_corners=True, mode="bilinear", padding_mode="border").type(input.dtype).squeeze(0))
        return results
    
    def _inverse(self, index: int, a: int, input : torch.Tensor) -> torch.Tensor:
        pass

class Permute(DataAugmentation):

    @config("Permute")
    def __init__(self, prob_permute: Union[list[float], None] = [0.33 ,0.33]) -> None:
        super().__init__()
        self._permute_dims = torch.tensor([[0, 2, 1, 3], [0, 3, 1, 2]])
        self.prob_permute = prob_permute
        self.permute: dict[int, torch.Tensor] = {}

    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        if len(shapes):
            dim = len(shapes[0])
            assert dim == 3, "The permute augmentation only support 3D images"
            if self.prob_permute:
                assert len(self.prob_permute) == 2, "len of prob_permute must be equal 2"
                self.permute[index] = torch.rand((len(shapes), len(self.prob_permute))) < torch.tensor(self.prob_permute)
            else:
                assert len(shapes) == 2, "The number of augmentation images must be equal to 2"
                self.permute[index] = torch.eye(2, dtype=torch.bool)
            for i, prob in enumerate(self.permute[index]):
                for permute in self._permute_dims[prob]:
                    shapes[i] = [shapes[i][dim-1] for dim in permute[1:]]
        return shapes
    
    def _compute(self, index: int, inputs : list[torch.Tensor], device: Union[torch.device, None]) -> list[torch.Tensor]:
        results = []
        for input, prob in zip(inputs, self.permute[index]):
            res = input
            for permute in self._permute_dims[prob]:
                res = res.permute(tuple(permute))
            results.append(res)
        return results
    
    def _inverse(self, index: int, a: int, input : torch.Tensor) -> torch.Tensor:
        for permute in reversed(self._permute_dims[self.permute[index][a]]):
            input = input.permute(tuple(np.argsort(permute)))
        return input

class Mask(DataAugmentation):

    @config("Mask")
    def __init__(self, mask: str, value: float) -> None:
        super().__init__()
        if mask is not None:
            if os.path.exists(mask):
                self.mask = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(mask)))
            else:
                raise NameError('Mask file not found')
        self.positions: dict[int, list[torch.Tensor]] = {}
        self.value = value

    def _state_init(self, index : int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        self.positions[index] = [torch.rand((3) if len(shape) == 3 else (2))*(torch.tensor([max(s1-s2, 0) for s1, s2 in zip(torch.tensor(shape), torch.tensor(self.mask.shape))])) for shape in shapes]
        return [self.mask.shape for _ in shapes]
    
    def _compute(self, index: int, inputs : list[torch.Tensor], device: Union[torch.device, None]) -> list[torch.Tensor]:
        results = []
        for input, position in zip(inputs, self.positions[index]):
            slices = [slice(None, None)]+[slice(int(s1), int(s1)+s2) for s1, s2 in zip(position, self.mask.shape)]
            padding = []
            for s1, s2 in zip(reversed(input.shape), reversed(self.mask.shape)):
                if s1 < s2:
                    pad = s2-s1
                else:
                    pad = 0
                padding.append(0)
                padding.append(pad)
            value = torch.tensor(0, dtype=torch.uint8) if input.dtype == torch.uint8 else torch.tensor(self.value).to(input.device)
            results.append(torch.where(self.mask.to(input.device) == 1, torch.nn.functional.pad(input, tuple(padding), mode="constant", value=value)[tuple(slices)], value))
        return results
    
    def _inverse(self, index: int, a: int, input : torch.Tensor) -> torch.Tensor:
        pass
