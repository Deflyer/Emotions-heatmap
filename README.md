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
      <li>Gera um heatmap com o dataframe desde que ele tenha sido classificado quanto as emoções</li>
    </ul>
  </li>
</ul>

Exemplo de uso do pip [https://colab.research.google.com/drive/1UqzA6bDgtZWGji652a0UjT7ZtDh3Xyi5?authuser=1#scrollTo=CKL-Mm9D18BB](https://colab.research.google.com/drive/1UqzA6bDgtZWGji652a0UjT7ZtDh3Xyi5?authuser=1#scrollTo=CKL-Mm9D18BB)

Este projeto foi apoiado pelo Ministério da Ciência, Tecnologia e Inovações, com recursos
da Lei no 8.248, de 23 de outubro de 1991, no âmbito do PPI-SOFTEX, coordenado pela Softex e
publicado Residência em TIC 13, DOU 01245.010222/2022-44.

Um agradecimento ao projeto da Motorola e ao C4AI por financiarem e fortalecerem pesquisas no âmbito do Centro de IA
