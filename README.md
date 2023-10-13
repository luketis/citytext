This repository contains the code that implements the city text analisys framework proposed in the Master Thesis "Detection, Recognition and Topic Semantic Matching of Street Level City Text For Urban Analysis". 

# Installation

1. Clone this repo: `git clone git@github.com:luketis/citytext.git`
2. Clone EAST: `git clone git@github.com:luketis/EAST.git`
3. Clone STR: `git clone git@github.com:luketis/deep-text-recognition-benchmark.git`
4. Install tensorflow: `pip install tensorflow==1.13.1`
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
5. Install STR dependencies: `pip install lmdb pillow nltk natsort`
6. Install a lower version of protobuf: `pip install protobuf==3.20.*`
7. Rename the folder `deep-text-recognition-benchmark` to `deep_text_recognition_benchmark`
8. Install `pip install osmnx==1.1.2 folium opencv-python matplotlib mapclassify wordcloud openai sentence-transformers`
9. Download EAST pre trained models as instructed in their README and put in folder `EAST/nets/`
10. Download STR pre trained models as instructed in their README and put in folder `deep_text_recognition_benchmark`
11. For property analisys in regions in the city of Sao Paulo, download geosampa from https://geosampa.prefeitura.sp.gov.br/PaginasPublicas/_SBC.aspx under Cadastro/Uso Predominante do Solo.


# Execution

- Coordinate Sampling: run coordinate_sampling.ipynb
- Images Acquisition: See https://www.mapillary.com/developer/api-documentation
- OCR and NLP steps: run pipeline.ipynb
- Analisys: run analysis.ipynb

# Citing
TODO 