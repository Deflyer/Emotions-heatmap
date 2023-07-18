import pandas as pd
import numpy as np
import math
import gdown
import matplotlib.pyplot as plt
import whisperx
import gc
import transformers
import webvtt
import os
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification

class VideoEmotionRecognition:
    def __init__(self, mp4):

        logging.basicConfig(filename="newfile.log",
        format='%(asctime)s %(message)s',filemode='w')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        self.logger.info("Initializing the class")

        self.dataframe = -1
        self.mp4 = mp4

        self.logger.info("Loading the arousal_valence map")

        url1 = 'https://drive.google.com/uc?id=1yJBHU8Zl4MuoQfJqkPgVDwJfYRJDPUAH'
        output = 'emotions_coord.xlsx'
        gdown.download(url1, output, quiet=False)
        self.emotions_coord = pd.read_excel(output)

        self.logger.info("Successfully initialized")


    def transcript(self, method = "whisperx", min_time = 1):
        self.logger.info("Running the whisperx transcription")
        bashCommand = "whisperx --compute_type float32 --output_format vtt " + self.mp4
        os.system(bashCommand)
        self.logger.info("Successfully transcripted")

        self.logger.info("Generating the Dataframe from the vtt")
        self.set_vtt(self.mp4.replace("mp4", "vtt"))

        self.dataframe = self.dataframe[self.dataframe.apply(lambda x: time_diff(x[1], x[0]), axis=1) > min_time]
        self.dataframe.reset_index(inplace = True)

        self.logger.info("Successfully generated the dataframe")

    #Funcao utilizada para gerar uma coluna de strings no Dataframe contendo cada frase traduzida para o inglês
    def __Traduz(self, frase):
        traducao = self.pten_pipeline(frase)
        traducao = list(traducao[0].values())
        return traducao[0]

    #Função para extrair do dicionario retornado pelo goemotions a emoção mais provável e sua probabilidade
    def __Emocao_provavel(self, frase):
        emotion_labels = self.emotion(frase)

        max = emotion_labels[0][0]["score"]
        emocao = emotion_labels[0][0]["label"]

        for dict in emotion_labels[0]:
            if dict["score"] > max:
              max = dict["score"]
              emocao = dict["label"]

        return emocao, max

    def emotion_recognition(self, modality = "transcript", method = "RoBERTa-Go-Emotion"):
        try:
            self.dataframe.size
            self.logger.info("Loading the translator")
            tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/translation-pt-en-t5")

            model = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/translation-pt-en-t5")

            self.pten_pipeline = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

            self.logger.info("Successfully loaded, now applyng")
            self.dataframe['Translation'] = list(self.dataframe[2].apply(self.__Traduz))

            self.logger.info("Loading the classifier")
            tokenizer = RobertaTokenizerFast.from_pretrained("bhadresh-savani/bert-base-go-emotion")
            model = TFRobertaForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-go-emotion",from_pt=True)

            self.emotion = pipeline('sentiment-analysis',
                                model="bhadresh-savani/bert-base-go-emotion",
                                return_all_scores=True)

            self.logger.info("Successfully loaded, now applying")
            resp = list(self.dataframe['Translation'].apply(self.__Emocao_provavel))
            temp = pd.DataFrame.from_records(resp, columns=['label', 'prob'])

            self.dataframe = pd.concat([self.dataframe, temp], axis=1)
            self.logger.info("Dataframe emotions classified")
        except AttributeError:
          raise ValueError("There is no transcription")

    def set_vtt(self, arquivo):
        self.logger.info("Loading dataframe via vtt")
        L = []

        for caption in webvtt.read(arquivo):
            L.append([caption.start,caption.end,str(caption.text)])

        self.dataframe = pd.DataFrame(L)
        self.logger.info("Successfully loaded")

    def get_labels(self, modality = "transcript"):
        return self.dataframe

    def __generate_coord(self,label):
        index = self.emotions_coord.loc[self.emotions_coord['Emotion'] == label].index[0]
        x = self.emotions_coord.iloc[index]['X']
        y = self.emotions_coord.iloc[index]['Y']
        return (x,y)

    def __kde_quartic(self,d,h):
        dn=d/h
        P=(15/16)*(1-dn**2)**2
        return P

    def get_heatmap(self, modality = "transcript"):
        
        self.logger.info("Adding the arousal valence coordinates")
        resp = list(self.dataframe['label'].apply(self.__generate_coord))
        temp = pd.DataFrame.from_records(resp, columns=['x', 'y'])

        self.dataframe = pd.concat([self.dataframe, temp], axis=1)
        self.dataframe = self.dataframe[self.dataframe.label!='neutral']
        self.dataframe = self.dataframe.reset_index(drop=True)

        array_x = self.dataframe['x'].to_numpy()
        x = array_x.tolist()
        array_y = self.dataframe['y'].to_numpy()
        y = array_y.tolist()

        #Definindo tamanho do grid e do raio(h)
        grid_size=0.02
        h=0.5

        #Tomando valores de máximos e mínimos de X e Y.
        x_min=-1
        x_max=1
        y_min=-1
        y_max=1

        #Construindo grid
        x_grid=np.arange(x_min-h,x_max+h,grid_size)
        y_grid=np.arange(y_min-h,y_max+h,grid_size)
        x_mesh,y_mesh=np.meshgrid(x_grid,y_grid)

        #Determinando ponto central do grid
        xc=x_mesh+(grid_size/2)
        yc=y_mesh+(grid_size/2)

        self.logger.info("Generating the intensity list for heatmap")
        intensity_list=[]
        for j in range(len(xc)):
            intensity_row=[]
            for k in range(len(xc[0])):
                kde_value_list=[]
                for i in range(len(x)):
                    #Calculando distância
                    d=math.sqrt((xc[j][k]-x[i])**2+(yc[j][k]-y[i])**2)
                    if d<=h:
                        p=self.__kde_quartic(d,h)
                    else:
                        p=0
                    kde_value_list.append(p)
                #Soma os valores de intensidade
                p_total=sum(kde_value_list)
                intensity_row.append(p_total)
            intensity_list.append(intensity_row)
            
        self.logger.info("Generating the plot")
        #Saída do Heatmap
        plt.figure(figsize=(7,7))

        intensity=np.array(intensity_list)
        plt.pcolormesh(x_mesh,y_mesh,intensity,cmap='YlOrRd') #https://matplotlib.org/stable/tutorials/colors/colormaps.html


        #fig, ax = plt.subplots()

        x_emo = self.emotions_coord.X.to_list()
        y_emo = self.emotions_coord.Y.to_list()
        plt.scatter(x_emo, y_emo)


        for i, row in self.emotions_coord.iterrows():
            plt.annotate(row['Emotion'], (x_emo[i], y_emo[i]))

        plt.xlim(-1, 1)
        plt.ylim(-1,1)

        ax = plt.gca()
        ax.add_patch(plt.Circle((0, 0), 1, color='black', fill=False))
        plt.axvline(x = 0, color = 'black', label = 'Arousal')
        plt.axhline(y = 0, color = 'black', label = 'Valence')

        #plt.colorbar()
        plt.plot(x,y,'x',color='white')

def time_diff(fim, init):

    str_fim = fim.split(":")
    str_init = init.split(":")

    time_fim = 3600*int(str_fim[0]) + 60*int(str_fim[1]) + float(str_fim[2])
    time_init = 3600*int(str_init[0]) + 60*int(str_init[1]) + float(str_init[2])

    return time_fim - time_init