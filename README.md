# Multimodal Emotion Recognition from Videos

Esse projeto consiste na elaboração de um método capaz de extrair emoções em tempo real de um vídeo, organizando-as em um mapa de calor sobre as dimensões de arousal-valence.

Aqui encontra-se o link para o colab com a baseline: [https://colab.research.google.com/drive/1KLLo-aIEw3ZPJSTsSZTkb9QEYWp5nveG?authuser=1#scrollTo=BE7a2oIoO75Q](https://colab.research.google.com/drive/1KLLo-aIEw3ZPJSTsSZTkb9QEYWp5nveG?authuser=1#scrollTo=BE7a2oIoO75Q)

Esse projeto também conta com um pip para ser mais facilmente usado: [pip](https://pypi.org/project/VideoEmotionRecognition/) que pode ser usado via 
pip install git+https://github.com/Deflyer/Multimodal-Emotion-Recognition-from-Videos.git 
ou
pip install VideoEmotionRecognition

Ele contém 4 métodos
<ul>
  <li>transcript
    <ul>
      <li>Transforma um arquivo mp4 em um dataframe do Pandas contendo 
      cada frase do vídeo e suas timestamps</li>
    </ul>
  </li>
  <li>emotion recognition
    <ul>
      <li>Classifica o dataframe do Pandas gerado pelo transcript quanto a suas emoções dependendo de qual
      "modality" for selecionada</li>
      <ul>
        <li>"modality" = "transcript"</li>
        <li>"modality" = "audio"</li>
        <li>"modality" = "video"</li>
        <li>"modality" = "multimodal"</li>
      </ul>
    </ul>
  </li>
  <li>get_labels
    <ul>
      <li>Retorna o dataframe</li>
    </ul>
  </li>
  <li>get_heatmap
    <ul>
      <li>Gera um heatmap com o dataframe desde que ele tenha sido classificado quanto as emoções, com algumas flags é possível mudar a exibição do heatmap</li>
      <ul>
        <li>"animated" = True</li>
          <ul>
            <li>Gera um vídeo contendo as variações no heatmap ao longo do tempo</li>
          </ul>
        <li>"join_video" = True</li>
          <ul>
            <li>Gera um vídeo contendo o vídeo original e o heatmap, com as emoções ao longo do tempo, simultaneamente</li>
          </ul>
      </ul>
    </ul>
  </li>
</ul>
Aqui tem um exemplo de heatmap gerado
<img src="/img/heatmap.png">
Links Importantes

Exemplo de uso do pip [https://colab.research.google.com/drive/1UqzA6bDgtZWGji652a0UjT7ZtDh3Xyi5?authuser=1#scrollTo=CKL-Mm9D18BB](https://colab.research.google.com/drive/1UqzA6bDgtZWGji652a0UjT7ZtDh3Xyi5?authuser=1#scrollTo=CKL-Mm9D18BB)

Apresentação no Canvas explicando o projeto [https://www.canva.com/design/DAFy1Pywr0Q/F6_WBK5Bjz12OHZG8P4V8Q/edit](https://www.canva.com/design/DAFy1Pywr0Q/F6_WBK5Bjz12OHZG8P4V8Q/edit)
https://sol.sbc.org.br/index.php/eniac/article/view/25746/25562

Artigo do ENIAC publicado sobre o assunto elaborado pelo colega Gabriel Natal[https://sol.sbc.org.br/index.php/eniac/article/view/25746/25562](https://sol.sbc.org.br/index.php/eniac/article/view/25746/25562)

Fontes para os modelos utilizados:
Github do whisperx(transcrição) [https://github.com/m-bain/whisperX](https://github.com/m-bain/whisperX)
Hugging Face do tradutor da Unicamp [https://huggingface.co/unicamp-dl/translation-pt-en-t5](https://huggingface.co/unicamp-dl/translation-pt-en-t5)
Site oficial do goemotions (texto) [https://blog.research.google/2021/10/goemotions-dataset-for-fine-grained.html?m=1](https://blog.research.google/2021/10/goemotions-dataset-for-fine-grained.html?m=1)
Github contendo a implementação do classificador HUBERT (áudio) [https://github.com/m3hrdadfi/soxan](https://github.com/m3hrdadfi/soxan)
Github do deepface(vídeo) [https://github.com/serengil/deepface](https://github.com/serengil/deepface)



Este projeto foi apoiado pelo Ministério da Ciência, Tecnologia e Inovações, com recursos
da Lei no 8.248, de 23 de outubro de 1991, no âmbito do PPI-SOFTEX, coordenado pela Softex e
publicado Residência em TIC 13, DOU 01245.010222/2022-44.

Um agradecimento ao projeto da Motorola e ao C4AI por financiarem e fortalecerem pesquisas no âmbito do Centro de IA
