import os
import torch 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from typing import Optional
from logger import TrainingLogger
from tqdm import tqdm


class ModularTrainer:


    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader, 
                 test_loader: Optional[torch.utils.data.DataLoader] = None,
                 loss_fn: Optional[torch.nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scaler: Optional[torch.cuda.amp.GradScaler] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 log_path: Optional[str] = './logs/training.log',
                 num_epochs: Optional[int] = 10,
                 checkpoint_path: Optional[str] = './checkpoints',
                 loss_path: Optional[str] = './loss curves/model_loss.png',
                 verbose: Optional[bool] = True,
                 device: Optional[torch.device] = None) -> None:
        

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(os.path.dirname(loss_path), exist_ok=True)
        
        self.logger = TrainingLogger(log_path=log_path)

        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        self.logger.info(f"Using device: {self.device}")
        
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader


        self.loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.Adam(params=self.model.parameters(), lr=1e-3)
        self.scaler = scaler
        self.scheduler = scheduler
        #self.scheduler = scheduler or torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path
        self.loss_path = loss_path
        self.verbose = verbose
        self.loss_update_step = 50

        self.current_epoch = 1
        self.current_step = 1
        self.best_metric = float('inf')

        self.history = {
            'Training Loss': [],
            'Training Accuracy' : [],
            'Testing Loss': [],
            'Testing Accuracy': []
        }

        self.step_history = {
            'Training Loss': [],
            'Training Accuracy' : [],
            'Testing Loss': [],
            'Testing Accuracy': []
        }


    def update_plot(self) -> None:

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        axs[0, 0].plot(self.step_history['Training Loss'], color='blue', label='Training Loss')
        axs[0, 0].set_title('Training Loss')
        axs[0, 0].set_xlabel('Steps [every 50]')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()

        axs[0, 1].plot(self.step_history['Training Accuracy'], color='green', label='Training Accuracy')
        axs[0, 1].set_title('Training Accuracy')
        axs[0, 1].set_xlabel('Steps [every 50]')
        axs[0, 1].set_ylabel('Accuracy')
        axs[0, 1].legend()

        axs[1, 0].plot(self.step_history['Testing Loss'], color='red', label='Testing Loss')
        axs[1, 0].set_title('Testing Loss')
        axs[1, 0].set_xlabel('Steps [every 50]')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()

        axs[1, 1].plot(self.step_history['Testing Accuracy'], color='green', label='Testing Accuracy')
        axs[1, 1].set_title('Testing Accuracy')
        axs[1, 1].set_xlabel('Steps [every 50]')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.loss_path)
        plt.close(fig)

        return


    def train_epoch(self) -> None:

        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0

        with tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Training)') as t:
            
            for i, batch in t:
                
                inp, label = batch['Input'].to(self.device), batch['Label'].to(self.device)
                
                if self.scaler and torch.cuda.is_available():
                    with torch.amp.autocast(device_type='cuda'):
                        out = self.model(inp)
                        loss = self.loss_fn(out, label)
                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    out = self.model(inp)
                    loss = self.loss_fn(out, label)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                predictions = torch.argmax(out, dim=1) if out.shape[1] > 1 else torch.round(torch.sigmoid(out))
                acc = accuracy_score(label.cpu().numpy(), predictions.detach().cpu().numpy()) * 100
                total_accuracy += acc
                self.current_step += 1

                t.set_postfix({
                    'Batch Loss' : loss.item(),
                    'Batch Accuracy' : acc,
                    'Train Loss' : total_loss/(i+1),
                    'Train Accuracy' : total_accuracy/(i+1)
                })

                if i % self.loss_update_step == 0 and i != 0:
                    self.step_history['Training Loss'].append(total_loss / (i+1))
                    self.step_history['Training Accuracy'].append(total_accuracy / (i+1))
                    self.update_plot()

        train_loss = total_loss / len(self.train_loader)
        train_accuracy = total_accuracy / len(self.train_loader)
        self.history['Training Loss'].append(train_loss)
        self.history['Training Accuracy'].append(train_accuracy)
        #self.update_plot()

        self.logger.info(f"Training loss for epoch {self.current_epoch}: {train_loss}")
        self.logger.info(f"Training accuracy for epoch {self.current_epoch}: {train_accuracy}")

        return
    


    def test_epoch(self) -> None:

        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0

        with tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Testing)') as t:
            
            for i, batch in t:
                
                inp, label = batch['Input'].to(self.device), batch['Label'].to(self.device)
                
                with torch.no_grad():
                    if self.scaler and torch.cuda.is_available():
                        with torch.amp.autocast(device_type='cuda'):
                            out = self.model(inp)
                            loss = self.loss_fn(out, label)
                    else:
                        out = self.model(inp)
                        loss = self.loss_fn(out, label)

                total_loss += loss.item()
                predictions = torch.argmax(out, dim=1) if out.shape[1] > 1 else torch.round(torch.sigmoid(out))
                acc = accuracy_score(label.cpu().numpy(), predictions.cpu().numpy()) * 100
                total_accuracy += acc

                t.set_postfix({
                    'Batch Loss' : loss.item(),
                    'Batch Accuracy' : acc,
                    'Test Loss' : total_loss/(i+1),
                    'Test Accuracy' : total_accuracy/(i+1)
                })

                if i % self.loss_update_step == 0 and i != 0:
                    self.step_history['Testing Loss'].append(total_loss / (i+1))
                    self.step_history['Testing Accuracy'].append(total_accuracy / (i+1))
                    self.update_plot()

        test_loss = total_loss / len(self.test_loader)
        test_accuracy = total_accuracy / len(self.test_loader)
        self.history['Testing Loss'].append(test_loss)
        self.history['Testing Accuracy'].append(test_accuracy)
        #self.update_plot()

        if self.scheduler:
            self.scheduler.step(test_loss)

        if test_loss < self.best_metric:
            self.best_metric = test_loss
            self.save_checkpoint(is_best=True)

        self.logger.info(f"Testing loss for epoch {self.current_epoch}: {test_loss}")
        self.logger.info(f"Testing accuracy for epoch {self.current_epoch}: {test_accuracy}\n")
        if self.scheduler:
            self.logger.info(f"Current Learning rate: {self.scheduler.get_last_lr()}")

        return
    

    def train(self, resume_from: Optional[str]=None) -> None:
        
        if resume_from:
            self.load_checkpoint(resume_from)
            self.logger.log_training_resume(
                epoch=self.current_epoch, 
                global_step=self.current_step, 
                total_epochs=self.num_epochs
            )
        else:
            self.logger.info(f"Starting training for {self.num_epochs} epochs from scratch")
    
        print(f"Starting training from epoch {self.current_epoch} to {self.num_epochs}")
        

        for epoch in range(self.current_epoch, self.num_epochs + 1):

            self.current_epoch = epoch
            self.train_epoch()
            
            if self.test_loader:
                self.test_epoch()
    
            self.save_checkpoint()
        
        return
    
    

    def save_checkpoint(self, is_best:Optional[bool]=False):

        checkpoint = {
            'epoch': self.current_epoch,
            'current_step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_history' : self.step_history,
            'history': self.history,
            'best_metric': self.best_metric
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            path = os.path.join(self.checkpoint_path, 'best_model.pth')
        else:
            path = os.path.join(
                self.checkpoint_path, 
                f'model_epoch_{self.current_epoch}.pth'
            )

        torch.save(checkpoint, path)
        
        if self.verbose:
            save_type = "Best model" if is_best else "Checkpoint"
            self.logger.info(f"{save_type} saved to {path}")


    def load_checkpoint(self, checkpoint:Optional[str]=None, resume_from_best:Optional[bool]=False):
        
        if resume_from_best:
            checkpoint_path = os.path.join(self.checkpoint_path, 'best_model.pth')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        else:
            checkpoint = torch.load(checkpoint)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint.get('epoch') + 1
        self.current_step = checkpoint.get('current_step')
        self.best_metric = checkpoint.get('best_metric')
        
        loaded_history = checkpoint.get('history')
        for key in self.history:
            self.history[key] = loaded_history.get(key, self.history[key])

        loaded_step_history = checkpoint.get('step_history')
        for key in self.step_history:
            self.step_history[key] = loaded_step_history.get(key, self.step_history[key])

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Resumed training from epoch {self.current_epoch}")
        
        return self.current_epoch
