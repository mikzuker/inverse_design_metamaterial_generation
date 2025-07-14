import torch
from pathlib import Path
from conditional_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from conditional_dataset_preparation import create_conditional_model_dataset
from typing import List, Optional


class Diffusion_model():
    def __init__(self, milestone: int,
                 use_film: bool = True,
                 cond_dim: Optional[int] = 10):
        self.model = Unet1D(
            dim = 16,  
            dim_mults = (1, 2, 4, 8),
            channels = 1,
            cond_dim = cond_dim,
            use_film = use_film)
        
        self.diffusion = GaussianDiffusion1D(
            self.model,
            seq_length = 16,    
            timesteps = 1000,
            objective = 'pred_v')

        self.milestone = milestone
        

    def __dataset_create__(self, dataset_path: Path, angles: List[int]):
        training_dataset, conditions = create_conditional_model_dataset(dataset_path, angles)
        dataset = Dataset1D(training_dataset, conditions)
        return dataset

    def __train__(self, 
                  dataset: Dataset1D, 
                  train_batch_size: int, 
                  train_lr: float, 
                  train_num_steps: int, 
                  gradient_accumulate_every: int, 
                  ema_decay: float, 
                  amp: bool,
                  loss_path: Path):

        self.trainer = Trainer1D(
            self.diffusion,
            dataset = dataset,
            train_batch_size = train_batch_size,
            train_lr = train_lr,
            train_num_steps = train_num_steps,
            gradient_accumulate_every = gradient_accumulate_every,
            ema_decay = ema_decay,
            amp = amp)
        
        self.trainer.train(path=loss_path, loss_number = self.milestone)
        self.trainer.save(self.milestone)

    def __sample__(self, batch_size: int, conditional_vec: torch.Tensor):
        self.trainer.load(self.milestone)

        sampled_seq = self.diffusion.sample(batch_size = batch_size, cond = conditional_vec)

        return sampled_seq


if __name__ == "__main__":
    model = Diffusion_model(milestone = 34, use_film = True)
    dataset = model.__dataset_create__(Path('diffusion_model/training_dataset'), 
                                       angles = [0, 10, 20, 40, 60, 80, 100, 120, 140, 160])
    model.__train__(dataset = dataset, 
                    train_batch_size = 1, 
                    train_lr = 8e-4, 
                    train_num_steps = 10000, 
                    gradient_accumulate_every = 2, 
                    ema_decay = 0.995, 
                    amp = True, 
                    loss_path = Path('diffusion_model/training_loss'))
    sampled_seq = model.__sample__(batch_size = 1, conditional_vec = torch.tensor([[0.3, 0.1, 0.05, 0.03, 0.01, 0.04, 0.08, 0.1, 0.09, 0.15]], dtype=torch.float32))
    print(sampled_seq, sampled_seq.shape)