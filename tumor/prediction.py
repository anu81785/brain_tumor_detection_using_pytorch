import torch
from .model.brain_tumor_detection import CustomResNet50
from .model.preprocessing import preprocess_image  


def load_model():
    num_classes = 17 
    custom_resnet = CustomResNet50(num_classes=num_classes, pretrained=True)  
    model = custom_resnet.get_model()
    model.load_state_dict(torch.load('/home/anusaini/Desktop/Django/brain_tumor_detection/brain_tumor_project/tumor/model/model.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

def process_prediction(prediction):
    class_names=['Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1',
  'Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1C+',
  'Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T2',
  'Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1',
  'Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1C+',
  'Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T2',
  'NORMAL T1',
  'NORMAL T2',
  'Neurocitoma (Central - Intraventricular, Extraventricular) T1',
  'Neurocitoma (Central - Intraventricular, Extraventricular) T1C+',
  'Neurocitoma (Central - Intraventricular, Extraventricular) T2',
  'Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1',
  'Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1C+',
  'Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T2',
  'Schwannoma (Acustico, Vestibular - Trigeminal) T1',
  'Schwannoma (Acustico, Vestibular - Trigeminal) T1C+',
  'Schwannoma (Acustico, Vestibular - Trigeminal) T2']
    predicted_class_idx = prediction.argmax()
    predicted_class_name = class_names[predicted_class_idx]
    return predicted_class_name

