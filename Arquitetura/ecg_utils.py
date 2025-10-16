import wfdb
import os
import glob

# Caminho base para os dados, pode ser importado por outros scripts
PATH_DADOS_HOLTER = r"D:\cox-models-sudden-death\01_Dataset\dados_ECG\Holter_ECG"

def get_lista_pacientes(incluir_apenas=None, excluir=None):
    """
    Busca todos os registros WFDB e filtra a lista opcionalmente.
    
    Args:
        incluir_apenas (list, optional): Se fornecido, retorna apenas os pacientes desta lista.
        excluir (list, optional): Se fornecido, remove estes pacientes da lista final.
    
    Returns:
        list: A lista final de IDs de pacientes a serem processados.
    """
    padrao_busca = os.path.join(PATH_DADOS_HOLTER, 'P0*.hea')
    arquivos_hea = glob.glob(padrao_busca)
    
    # Extrai IDs dos pacientes
    todos_pacientes = [os.path.basename(f).replace('.hea', '') for f in arquivos_hea]
    
    # Aplicar filtros
    if incluir_apenas:
        todos_pacientes = [p for p in todos_pacientes if p in incluir_apenas]
        print(f"✓ Filtrado para {len(todos_pacientes)} pacientes incluídos")
    
    if excluir:
        todos_pacientes = [p for p in todos_pacientes if p not in excluir]
        print(f"✓ Removidos {len(excluir)} pacientes. Restam {len(todos_pacientes)}")
    
    return todos_pacientes



def load_ecg_segment(paciente_id, canal='z', minutos_a_pular=60, duracao_em_minutos=10, sampling_rate=200):
    """
    Função reutilizável para carregar um segmento de ECG de um paciente.
    """
    # Mapeia a letra do canal para o índice numérico
    mapa_canais = {'x': 0, 'y': 1, 'z': 2}
    if canal.lower() not in mapa_canais:
        raise ValueError(f"Canal '{canal}' inválido. Escolha entre 'x', 'y', ou 'z'.")
    canal_idx = mapa_canais[canal.lower()]
    
    record_path = os.path.join(PATH_DADOS_HOLTER, paciente_id)
    
    idx_inicio = minutos_a_pular * 60 * sampling_rate
    idx_fim = idx_inicio + (duracao_em_minutos * 60 * sampling_rate)
    
    try:
        record = wfdb.rdrecord(record_path, sampfrom=idx_inicio, sampto=idx_fim)
        signal = record.p_signal[:, canal_idx]
        print(f"✅ Sinal do paciente {paciente_id} (canal {canal.upper()}) carregado com sucesso.")
        return signal
    except Exception as e:
        print(f"❌ Erro ao carregar o registro {paciente_id}: {e}")
        return None
    
    
    