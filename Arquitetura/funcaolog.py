import os
import datetime
import torchtuples as tt
import torch
import logging
from logger_config import setup_logger # biblioteca local


# Diretório base para checkpoints
BASE_DIR = r"D:\cox-models-sudden-death\Arquitetura\logs\checkpoints"
BASE_DIR2 = r"D:\cox-models-sudden-death\Arquitetura\logs"

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(BASE_DIR2, exist_ok=True)

# salvando os pesos
def get_callbacks(pat=10, check=True, nome_modelo="best_model", load=True, rm=False):
    """Retorna lista com callback EarlyStopping configurado para PyCox
    
    Args:
        pat (int, optional): Número de épocas sem melhora antes de parar. Defaults to 5.
        check (bool, optional): Se deve salvar o modelo no melhor ponto. Defaults to True.
        nome_modelo (str, optional): Nome base para o arquivo de checkpoint. Defaults to "best_model".
        load (bool, optional): Se deve carregar os melhores pesos ao final do treinamento. Defaults to True.
        rm (bool, optional): Se deve remover o arquivo de checkpoint após carregar os melhores pesos. Defaults to False.
    """
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_path = os.path.join(BASE_DIR, f"{nome_modelo}_pesos_{timestamp}.pt")
    
    early_cb = tt.callbacks.EarlyStopping(
        patience=pat,
        min_delta=0.005,  # 0.0001,  # Limiar para o que é uma "melhora"
        checkpoint_model=check,
        file_path=checkpoint_path,
        load_best=load,
        rm_file=rm
    )
    
    return early_cb

# =====================================================================================================
class SaveFullModelCallback(tt.callbacks.Callback):
    def __init__(self, nome_modelo="net1d", save_each="epoch"):
        """
        Callback para salvar o modelo durante o treino (rede + otimizador).
        Também registra logs em arquivo separado.
        """
        super().__init__()
        self.nome_modelo = nome_modelo
        self.save_each = save_each
        self.base_dir = r"D:\cox-models-sudden-death\Arquitetura\logs\checkpoints_full"
        os.makedirs(self.base_dir, exist_ok=True)

        # Cria logger específico para checkpoints
        self.logger = setup_logger(
            r"D:\cox-models-sudden-death\Arquitetura\logs",
            "processamento_modelos.log",
            name="processamento_modelos"
        )

        # Contador de épocas manual
        self.current_epoch = 0

    def set_model(self, model):
        """Recebe o modelo do torchtuples (necessário para acessar .net e .optimizer)."""
        self.model = model

    def on_batch_end(self, logs=None):
        if self.save_each == "batch":
            batch = logs.get("batch") if logs else None
            epoch = logs.get("epoch") if logs else self.current_epoch
            self._save_checkpoint(epoch, batch)

    def on_epoch_end(self, logs=None):
        if self.save_each == "epoch":
            epoch = logs.get("epoch") if logs else self.current_epoch
            self._save_checkpoint(epoch, None)
            self.current_epoch += 1

    def _save_checkpoint(self, epoch, batch):
        # 🔹 Nome fixo para sobrescrever sempre
        filename = f"{self.nome_modelo}.pt"
        checkpoint_path = os.path.join(self.base_dir, filename)

        # 🔹 Salvar pesos da rede + otimizador
        checkpoint = {
            "epoch": epoch,
            "batch": batch,
            "model_state": self.model.net.state_dict(),
            "optimizer_state": self.model.optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        # 🔹 Log simplificado (sem diretório completo)
        if batch is not None:
            self.logger.info(f"Modelo salvo - Epoch: {epoch}, Batch: {batch}, Arquivo: {filename}")
        else:
            self.logger.info(f"Modelo salvo - Epoch: {epoch}, Arquivo: {filename}")

        
        
# =====================================================================================================
def main():
    callbacks = get_callbacks()
    return callbacks

if __name__ == "__main__":
    callbacks = main()
    print("Callbacks criados:", callbacks)
    
    callback = SaveFullModelCallback(nome_modelo="meu_modelo", save_each="epoch")
    print("Callback configurado:", callback)


