# APIDeep_Streamlit
Desarrollo de un API para clasificación de imágenes usando Streamlit. Se presenta el ejemplo con clasificación de imágenes CT Covid, y clasificación de especies de aves


## 1. Preparación del entorno
    $ conda create -n APIDeepStreamlit anaconda python=3.7.7
    $ conda activate APIDeepStreamlit
    $ conda install ipykernel
    $ python -m ipykernel install --user --name APIDeepStreamlit --display-name "APIDeepStreamlit"
    $ conda install tensorflow-gpu==2.1.0 cudatoolkit=10.1
    $ pip install tensorflow==2.1.0
    $ pip install jupyter
    $ pip install keras==2.3.1
    $ pip install numpy scipy Pillow cython matplotlib scikit-image opencv-python h5py imgaug IPython[all] streamlit
    
 ## 2. Ejecutar streamlit
 
    $ streamlit run appCOVID.py

## Resultado

![API web Streamlit + Deep Learning](https://github.com/DavidReveloLuna/APIDeep_Streamlit/blob/master/asssets/Resultado.jpg)

    $ streamlit run appAves.py
    
 ## Resultado
 
 ![API web Streamlit + Deep Learning](https://github.com/DavidReveloLuna/APIDeep_Streamlit/blob/master/asssets/Resultado2.jpg)
