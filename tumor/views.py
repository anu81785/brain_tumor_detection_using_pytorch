import torch
from django.shortcuts import render
from .model.preprocessing import preprocess_image 
from .prediction import load_model
from .prediction import process_prediction

def classify_tumor(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['tumor_image']
        model = load_model()  
        image = preprocess_image(uploaded_file) 
        with torch.no_grad():
            prediction=model(image)
        result = process_prediction(prediction)
        return render(request, 'result.html', {'result': result})
    
    return render(request, 'upload.html')
