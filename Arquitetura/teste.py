import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchtuples as tt
from pycox.models import LogisticHazard
from pycox.evaluation import EvalSurv
from sklearn.model_selection import train_test_split


# Agora o diretório de trabalho é a pasta raiz. A importação absoluta funciona diretamente.
from processamento import PassandoDados # biblioteca local
from funcaolog import get_callbacks, SaveFullModelCallback # biblioteca local

print("Importação bem-sucedida!")



tensores = PassandoDados()
tensores.limite_features = 1000
#tensores.limite_labels = 200
tensores.metodo = 'neurokit'


# 4. DATASET E DATALOADER 
class ECGDatasetBatch(Dataset):
    def __init__(self, sinais, durations, events):
        self.durations, self.events = tt.tuplefy(durations, events).to_tensor()
        self.sinais = sinais
    def __len__(self):
        return len(self.durations)
    def __getitem__(self, index):
        if not hasattr(index, '__iter__'): index = [index]
        batch_sinais = self.sinais[index]
        return tt.tuplefy(batch_sinais, (self.durations[index], self.events[index]))
    
# 1. SEUS DADOS INICIAIS COM AS FORMAS CORRETAS
# sinais.size() -> torch.Size([756, 3, 6000])
# tempos.size() -> torch.Size([756])
# eventos.size() -> torch.Size([756])
sinais, tempos, eventos = tensores.convert_to_tensors()

# 2. DIVISÃO TREINO/TESTE (permanece igual)
indices = np.arange(len(sinais))
idx_treino, idx_teste = train_test_split(indices, test_size=0.2, random_state=123)
sinais_treino, sinais_teste = sinais[idx_treino], sinais[idx_teste]
tempos_treino, tempos_teste = tempos[idx_treino], tempos[idx_teste]
eventos_treino, eventos_teste = eventos[idx_treino], eventos[idx_teste]


# Converta os tensores para arrays NumPy ANTES de passar para a função
tempos_treino_np = tempos_treino.numpy()
eventos_treino_np = eventos_treino.numpy()
tempos_teste_np = tempos_teste.numpy()
eventos_teste_np = eventos_teste.numpy()

# 3. TRANSFORMAÇÃO DOS RÓTULOS (permanece igual)
num_durations = 20
labtrans = LogisticHazard.label_transform(num_durations)


# Agora, use os arrays NumPy
alvo_treino = labtrans.fit_transform(tempos_treino_np, eventos_treino_np)
alvo_teste = labtrans.transform(tempos_teste_np, eventos_teste_np)

# alvo_treino = labtrans.fit_transform(tempos_treino, eventos_treino)
# alvo_teste = labtrans.transform(tempos_teste, eventos_teste)



dataset_treino = ECGDatasetBatch(sinais_treino, *alvo_treino)
dataset_teste = ECGDatasetBatch(sinais_teste, *alvo_teste)
batch_size = 128
dl_treino = tt.data.DataLoaderBatch(dataset_treino, batch_size, shuffle=True)
dl_teste = tt.data.DataLoaderBatch(dataset_teste, batch_size, shuffle=False)

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtuples as tt
from pycox.models import LogisticHazard
import matplotlib.pyplot as plt
from pycox.evaluation import EvalSurv
import torchtuples as tt


# ============================================
# 1. ARQUITETURA CNN 1D
# ============================================
class Net1D(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.max_pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 16, 5, 1)
        self.glob_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, out_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = self.glob_avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================
# 2. INICIANDO EXPERIMENTO MLflow
# ============================================

mlflow.set_tracking_uri("file:///D:/cox-models-sudden-death/Arquitetura/logs/mlruns")
mlflow.set_experiment("TESTEecg_cox_experiment")

with mlflow.start_run(run_name="TESTENet1D Cox Model"):
    # -------------------
    # 2.1 Modelo
    # -------------------
    net_1d = Net1D(labtrans.out_features)
    model = LogisticHazard(net_1d, tt.optim.Adam(0.01), duration_index=labtrans.cuts)

    # -------------------
    # 2.2 Callbacks
    # -------------------
    modelo = "TESTE2net1d_model"
    
    save_cb = SaveFullModelCallback(nome_modelo=modelo, save_each="epoch")
    early_cb = get_callbacks()
    callbacks = [early_cb, save_cb]

    # -------------------
    # 2.3 Treinamento
    # -------------------
    epochs = 20
    log = model.fit_dataloader(dl_treino, epochs, callbacks, True, val_dataloader=dl_teste)

    # -------------------
    # 2.4 Logando métricas no MLflow
    # -------------------
    df_log = log.to_pandas()

    for epoch, row in df_log.iterrows():
        mlflow.log_metric("train_loss", row["train_loss"], step=epoch)
        mlflow.log_metric("val_loss", row["val_loss"], step=epoch)

    # -------------------
    # 2.5 Salvando modelo no MLflow
    # -------------------
    
    
    mlflow.pytorch.log_model(net_1d, modelo)
    
    # 2. Registrar no MLflow Model Registry
    mlflow.register_model(
    f"runs:/{mlflow.active_run().info.run_id}/{modelo}",  # o mesmo nome que passou no log_model
    name=modelo)

    

    # -------------------
    # 2.6 Salvando hiperparâmetros
    # -------------------
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", dl_treino.batch_size)

    # -------------------
    # 2.7 Salvando artefatos (logs/checkpoints)
    # -------------------
    mlflow.log_artifact(fr"D:\cox-models-sudden-death\Arquitetura\logs\checkpoints_full\{modelo}.pt")
    mlflow.log_artifact(r"D:\cox-models-sudden-death\Arquitetura\logs\processamento_modelos.log")

    print("✅ Treinamento e log concluídos no MLflow!")
    
    # --- Plot da perda ---
    print("--- Plot da Perda de Treinamento e Validação ---")
    _ = log.plot()
    plt.title("Perda de Treinamento vs. Validação")
    plt.ylabel('Perda (Log-Likelihood Negativo)')
    plt.xlabel('Época')
    plt.savefig("perda_treino_val.png")
    plt.show()

    # Log da figura no MLflow
    mlflow.log_artifact("perda_treino_val.png")

    # --- Avaliação no conjunto de teste ---
    dl_teste_x = tt.data.dataloader_input_only(dl_teste)
    curvas_sobrevida = model.interpolate(10).predict_surv_df(dl_teste_x)
    ev = EvalSurv(curvas_sobrevida, tempos_teste.numpy(), eventos_teste.numpy(), censor_surv='km')

    # C-index
    c_index = ev.concordance_td()
    print(f"C-index (time-dependent): {c_index:.4f}")
    mlflow.log_metric("c_index", c_index)

    # --- Plot das curvas de sobrevivência ---
    curvas_sobrevida.iloc[:, :10].plot()
    plt.ylabel('Probabilidade de Sobrevivência S(t|x)')
    plt.xlabel('Tempo')
    plt.title("Curvas de Sobrevivência (Teste - Primeiros 10 Pacientes)")
    plt.savefig("curvas_sobrevida.png")
    plt.show()

    # Log da figura no MLflow
    mlflow.log_artifact("curvas_sobrevida.png")

