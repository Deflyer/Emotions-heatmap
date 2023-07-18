# Multimodal Emotion Recognition from Videos

Esse projeto consiste na elaboração de um método capaz de extrair emoções em tempo real de um vídeo, organizando-as em um mapa de calor sobre as dimensões de arousal-valence.

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
      <li>Classificar o dataframe do Pandas gerado pelo transcript quanto a suas emoções</li>
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

Aqui encontra-se o link para o colab com a baseline: [https://colab.research.google.com/drive/1KLLo-aIEw3ZPJSTsSZTkb9QEYWp5nveG?authuser=1#scrollTo=BE7a2oIoO75Q](https://colab.research.google.com/drive/1KLLo-aIEw3ZPJSTsSZTkb9QEYWp5nveG?authuser=1#scrollTo=BE7a2oIoO75Q)